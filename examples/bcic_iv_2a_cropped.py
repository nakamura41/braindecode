import logging
import os.path
import time
from collections import OrderedDict
import sys
import argparse
import itertools

import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import optim
import torch as th
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda):
    ival = [-500, 4000]
    input_time_length = 1125
    max_epochs = 800
    max_increase_epochs = 80
    batch_size = 60
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.2

    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCICompetition4Set2A(
        train_filepath, labels_filename=train_label_filepath)
    test_loader = BCICompetition4Set2A(
        test_filepath, labels_filename=test_label_filepath)
    train_cnt = train_loader.load()
    test_cnt = test_loader.load()

    # Preprocessing

    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    # lets convert to millvolt for numerical stability of next operations
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                       'EOG-central', 'EOG-right'])
    assert len(test_cnt.ch_names) == 22
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - valid_set_fraction)

    set_random_seeds(seed=20190706, cuda=cuda)

    n_classes = 4
    n_chans = int(train_set.X.shape[1])
    if model == 'shallow':
        model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                final_conv_length=30).create_network()
    elif model == 'deep':
        model = Deep4Net(n_chans, n_classes, input_time_length=input_time_length,
                         final_conv_length=2).create_network()

    to_dense_prediction_model(model)
    if cuda:
        model.cuda()

    log.info("Model: \n{:s}".format(str(model)))
    dummy_input = np_to_var(train_set.X[:1, :, :, None])
    print("dummy_input shape:", train_set.X[:1, :, :, None].shape)
    if cuda:
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)

    print("out shape:", out.cpu().data.numpy().shape)

    n_preds_per_input = out.cpu().data.numpy().shape[2]

    optimizer = optim.Adam(model.parameters())

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    stop_criterion = Or([MaxEpochs(max_epochs),
                         NoDecrease('valid_misclass', max_increase_epochs)])

    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(
                    input_time_length=input_time_length), RuntimeMonitor()]

    model_constraint = MaxNormDefaultConstraint()

    loss_function = lambda preds, targets: F.nll_loss(
        th.mean(preds, dim=2, keepdim=False), targets)

    exp = Experiment(model, train_set, valid_set, test_set, iterator=iterator,
                     loss_function=loss_function, optimizer=optimizer,
                     model_constraint=model_constraint,
                     monitors=monitors,
                     stop_criterion=stop_criterion,
                     remember_best_column='valid_misclass',
                     run_after_early_stop=True, cuda=cuda)
    exp.run()
    return exp


def batch_predict(experiment, dataset='test', is_target_cropped=True):
    all_actual = []
    all_predicted = []
    for batch in experiment.iterator.get_batches(experiment.datasets[dataset], shuffle=False):

        feature_vars = np_to_var(batch[0], pin_memory=experiment.pin_memory)
        target_vars = np_to_var(batch[1], pin_memory=experiment.pin_memory)

        if experiment.cuda:
            feature_vars = feature_vars.cuda()
            target_vars = target_vars.cuda()

        outputs = experiment.model(feature_vars)

        if hasattr(outputs, 'cpu'):
            outputs = outputs.cpu().data.numpy()
        else:
            # assume it is iterable
            outputs = [o.cpu().data.numpy() for o in outputs]

        if hasattr(target_vars, 'cpu'):
            target_vars = target_vars.cpu().data.numpy()
        else:
            # assume it is iterable
            target_vars = [o.cpu().data.numpy() for o in target_vars]

        predicted_vars = np.argmax(outputs, axis=1).squeeze()

        all_actual += target_vars.tolist()
        all_predicted += predicted_vars.tolist()

    if is_target_cropped:
        new_all_predicted = []
        for y in all_predicted:
            (values, counts) = np.unique(y, return_counts=True)
            ind = np.argmax(counts)
            new_all_predicted.append(values[ind])
        all_predicted = new_all_predicted

    return all_actual, all_predicted


def plot_confusion_matrix(confusion_matrix, classes, model_name, title, xlabel, ylabel, width=6, height=5,
                          cmap=plt.cm.Blues):
    fig = plt.figure(figsize=(width, height), dpi=80)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # plt.show()
    fig.savefig('result/{}_cm.png'.format(model_name))


def evaluate(experiment, subject_id, model_name):
    test_target_actual, test_target_predicted = batch_predict(experiment, 'test', is_target_cropped=True)
    all_classes = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']

    test_actual = [all_classes[i] for i in test_target_actual]
    test_predicted = [all_classes[i] for i in test_target_predicted]

    kappa_score = metrics.cohen_kappa_score(test_actual, test_predicted, labels=all_classes)
    pd.DataFrame([kappa_score], columns=['kappa_score']).to_csv('result/{}_kappa.csv'.format(model_name), index=False)
    confusion_matrix = metrics.confusion_matrix(test_actual, test_predicted, labels=all_classes)
    pd.DataFrame(confusion_matrix, index=all_classes, columns=all_classes).to_csv('result/{}_cm.csv'.format(model_name))
    performance_score = metrics.classification_report(test_actual, test_predicted, target_names=all_classes,
                                                      output_dict=True)
    pd.DataFrame(performance_score).transpose().to_csv('result/{}_score.csv'.format(model_name))
    plot_confusion_matrix(confusion_matrix, classes=all_classes, model_name=model_name,
                          title='Subject A0{:d} Confusion Matrix'.format(subject_id), xlabel='Predicted Classes',
                          ylabel='True Classes')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = '/home/david/data/BCICIV_2a_gdf/'

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on BCI Competition IV 2A Dataset.")
    parser.add_argument('--subject_id', default=1, type=int)  # 1-9
    parser.add_argument('--low_cut_hz', default=4, type=int)  # 0 or 4
    parser.add_argument('--model', default='shallow', type=str)  # 'shallow' or 'deep'
    parser.add_argument('--cuda', default=True, type=bool)  # True or False
    args = parser.parse_args()

    model_name = '{}_A0{:d}_{}'.format(args.model, args.subject_id, args.low_cut_hz)

    exp = run_exp(data_folder, args.subject_id, args.low_cut_hz, args.model, args.cuda)
    log.info("Last 10 epochs")
    log.info("\n" + str(exp.epochs_df.iloc[-10:]))

    # save model
    filepath = "result/{}_saved_model.pt".format(model_name)
    th.save(exp.model, filepath)
    log.info("Save model to {}".format(filepath))

    evaluate(exp, args.subject_id, model_name)
    log.info("Done")
