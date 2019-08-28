from __future__ import division

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy.stats import ranksums
from scipy import ndimage

import mkl
from SPHARM.lib import classification
from SPHARM.lib.confusion_matrix_pretty_print import plot_confusion_matrix_from_data
from SPHARM.classes.stratified_group_shuffle_split import GroupShuffleSplitStratified
from helper_lib import filelib

import warnings
warnings.simplefilter(action='ignore', category=Warning)
mkl.set_num_threads(1)


def pvalue_to_star(pvalue, sym='*'):

    if pvalue < 0.001:
        return sym*3
    elif pvalue < 0.01:
        return sym*2
    elif pvalue < 0.05:
        return sym
    else:
        return ''


def predict_classes_pairwise(features, classes, groups, samples, C):
    accuracy = pd.DataFrame()
    class_names = np.unique(groups)

    if grouped:
        curaccuracy = classification.predict_group_shuffle_split(features, classes, C=C,
                                                                 nsplits=150,
                                                                 test_size=len(np.unique(classes)),
                                                                 groups=samples,
                                                                 random_state=0)
    else:
        curaccuracy = classification.predict_shuffle_split(features, classes, C=C,
                                                           nsplits=150, test_size=2./7, random_state=0)

    curaccuracy['Comparison'] = class_names[0] + ' vs ' + class_names[1]
    curaccuracy['Pair'] = class_names[0] + ' vs ' + class_names[1]
    accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

    unique_classes = np.unique(classes)
    n_samples = [np.sum(np.where(classes == unique_classes[0], 1, 0)),
                 np.sum(np.where(classes == unique_classes[1], 1, 0))]
    n_samples.sort()
    if n_samples[0] < 0.4 * np.sum(n_samples):
        balanced = False
    else:
        balanced = True

    if grouped:
        unique_groups = np.unique(samples)
        group_classes = []
        for gr in unique_groups:
            group_classes.append(classes[np.where(samples==gr)[0][0]])

        if balanced:
            for i_shuffles in range(15):
                np.random.shuffle(group_classes)
                shuffled_classes = np.zeros_like(classes)
                for igr, gr in enumerate(unique_groups):
                    ind = np.where(samples == gr)
                    shuffled_classes[ind] = group_classes[igr]

                curaccuracy = classification.predict_group_shuffle_split(features, shuffled_classes, C=C,
                                                                         nsplits=10,
                                                                         test_size=len(np.unique(shuffled_classes)),
                                                                         groups=samples,
                                                                         random_state=0)

                curaccuracy['Comparison'] = 'Control'
                curaccuracy['Pair'] = class_names[0] + ' vs ' + class_names[1]
                accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

        else:
            cv = GroupShuffleSplitStratified(n_splits=150, test_size=len(np.unique(classes)), random_state=0)
            for train, test in cv.split(X=features, y=classes, groups=samples):
                train_classes = classes[train]
                unique_train_classes = np.unique(train_classes)
                n_observations = ndimage.sum(np.ones_like(train_classes), train_classes, unique_train_classes)
                predicted_classes = np.ones_like(classes[test])*unique_train_classes[np.argmax(n_observations)]
                curaccuracy = pd.DataFrame({'Accuracy': [np.sum(np.where(classes[test] == predicted_classes, 1, 0))
                                                        / len(predicted_classes)],
                                            'Comparison': 'Control',
                                            'Pair': class_names[0] + ' vs ' + class_names[1]})
                accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

    else:
        shuffled_classes = classes.copy()
        for i_shuffles in range(5):
            np.random.shuffle(shuffled_classes)
            curaccuracy = classification.predict_shuffle_split(features, shuffled_classes, C=C,
                                                               nsplits=10, test_size=2./7, random_state=0)
            curaccuracy['Comparison'] = 'Control'
            curaccuracy['Pair'] = 'Control'
            accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

    return accuracy


def predict_classes(folder_accuracy, folder_predicted, inputfile=None, stat=None, group='Group',
                    id_col='TrackID', static=True, cutoff=None, C=1., grouped=False, **kwargs):
    folders = [folder_accuracy, folder_predicted]
    filelib.make_folders(folders)

    if stat is None:
        stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    if cutoff is not None:
        curstat = stat[stat['degree'] <= cutoff]
    else:
        curstat = stat

    print('Feature extraction')
    features, classes, names, groups, samples = classification.extract_features(curstat,
                                                                                cell_id=id_col,
                                                                                group=group,
                                                                                static=static,
                                                                                **kwargs)[:]

    print('Prediction')
    if grouped:
        accuracy, predicted = classification.predict_classes_loo(features, classes, C=C, groups=samples)
    else:
        accuracy, predicted = classification.predict_classes_st_kfold(features, classes,
                                                                      nsplits=7, random_state=0, C=C)

    predicted['Name'] = names
    predicted[group] = groups

    print('Prediction pairwise')
    accuracy = pd.DataFrame()
    if len(np.unique(classes)) > 2:
        for cl in np.unique(classes):
            ind = np.where(classes != cl)
            if len(samples) > 0:
                cursamples = samples[ind]
            else:
                cursamples = samples
            curaccuracy = predict_classes_pairwise(features[ind], classes[ind], groups[ind], cursamples, C)
            accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

    else:
        accuracy = predict_classes_pairwise(features, classes, groups, samples, C)

    kwargs['Static'] = static
    kwargs['C'] = C
    kwargs['cutoff'] = cutoff

    for i, results in enumerate([accuracy, predicted]):
        outputfile = folders[i]
        for key in kwargs.keys():
            results[key] = kwargs[key]
            outputfile += key + '=' + str(kwargs[key]) + '_'

        results.to_csv(outputfile[:-1] + '.csv', sep='\t')


def plot_confusion_matrix(inputfolder, outputfolder):
    filelib.make_folders([outputfolder])
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])
    for fn in files:
        stat = pd.read_csv(inputfolder + fn, sep='\t', index_col=0)
        classes = stat['Group'].unique()
        cl_frame = pd.DataFrame({'Class name': classes})
        for i in range(len(cl_frame)):
            cl_frame.at[i, 'Class code'] = stat[stat['Group'] == cl_frame.iloc[i]['Class name']]['Actual class'].iloc[0]
            cl_frame.at[i, 'Class name'] = cl_frame.iloc[i]['Class name'].replace('NW=6_PW=3_', '')
        cl_frame = cl_frame.sort_values('Class name')
        cl_frame['New class code'] = np.arange((len(cl_frame)))

        for i in range(len(cl_frame)):
            stat.at[stat[stat['Actual class'] ==
                         cl_frame.iloc[i]['Class code']].index, 'Actual class'] = cl_frame.iloc[i]['Class name']
            stat.at[stat[stat['Predicted class'] ==
                         cl_frame.iloc[i]['Class code']].index, 'Predicted class'] = cl_frame.iloc[i]['Class name']
        plot_confusion_matrix_from_data(stat['Actual class'], stat['Predicted class'],
                                        columns=cl_frame['Class name'], outputfile=outputfolder + fn[:-4] + '.png')
        plt.close()


def plot_accuracy_pairwise(inputfolder, outputfolder):
    filelib.make_folders([outputfolder])
    filelib.combine_statistics(inputfolder)
    stat = pd.DataFrame.from_csv(inputfolder[:-1] + '.csv', sep='\t')
    for i in range(len(stat)):
        stat.at[i, 'Comparison'] = stat.iloc[i]['Comparison'].replace('NW=6_PW=3_', '').replace('FB', 'FR')
    stat['Features'] = ''
    stat.at[stat[stat['Static'] == True].index, 'Features'] = 'Static'
    stat.at[stat[(stat['Static'] == False) & (stat['dynamic_features'] == 'time')].index, 'Features'] = 'Dynamic\n time'
    stat.at[stat[(stat['Static'] == False)
                 & (stat['dynamic_features'] == 'frequency')].index, 'Features'] = 'Dynamic\n frequency'

    stat = stat.sort_values(['Features', 'Comparison'], ascending=False)

    for pair in stat['Pair'].unique():
        pair_stat = stat[stat['Pair'] == pair]

        plt.figure(figsize=(4, 4))
        margins = {'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.13}
        plt.subplots_adjust(**margins)
        sns.boxplot(x='Features', y='Accuracy', hue='Comparison', data=pair_stat, palette='Set1')
        sns.despine()
        plt.xlabel('')
        plt.ylim(0, 1.05)
        ncomparisons = len(pair_stat['Comparison'].unique())

        for ifeatures, feature in enumerate(pair_stat['Features'].unique()):
            print(ifeatures, feature)
            curstat = pair_stat[pair_stat['Features'] == feature]
            control_stat = curstat[curstat['Comparison'] == 'Control']['Accuracy']
            for icomparison, comparison in enumerate(curstat['Comparison'].unique()):
                print(icomparison, comparison)
                if comparison != 'Control':
                    teststat = curstat[curstat['Comparison'] == comparison]['Accuracy']
                    pval = ranksums(control_stat, teststat)[1]
                    boxwidth = 0.8 / ncomparisons
                    xpos = ifeatures - boxwidth * ncomparisons / 2 + boxwidth / 2 + icomparison * boxwidth
                    print(xpos)
                    plt.text(xpos, np.max(teststat) + 0.01, pvalue_to_star(pval), family='sans-serif', fontsize=8,
                             horizontalalignment='center', verticalalignment='bottom', color='black')

        for icomparison, comparison in enumerate(pair_stat['Comparison'].unique()):
            if comparison != 'Control':
                curstat = pair_stat[pair_stat['Comparison'] == comparison]
                control_stat = curstat[curstat['Features'] == 'Static']['Accuracy']
                for ifeatures, feature in enumerate(stat['Features'].unique()):
                    teststat = curstat[curstat['Features'] == feature]['Accuracy']
                    if np.mean(teststat) > 0.55:
                        pval = ranksums(control_stat, teststat)[1]
                        boxwidth = 0.8 / ncomparisons
                        xpos = ifeatures - boxwidth * ncomparisons / 2 + boxwidth / 2 + icomparison * boxwidth
                        plt.text(xpos, np.max(teststat) + 0.06, pvalue_to_star(pval, sym='$'), family='sans-serif',
                                 fontsize=5,
                                 horizontalalignment='center', verticalalignment='bottom', color='black')

                control_stat = curstat[curstat['Features'] == 'Dynamic\n time']['Accuracy']
                teststat = curstat[curstat['Features'] == 'Dynamic\n frequency']['Accuracy']
                ifeatures = 2
                if np.mean(teststat) > 0.55:
                    pval = ranksums(control_stat, teststat)[1]
                    boxwidth = 0.8 / ncomparisons
                    xpos = ifeatures - boxwidth * ncomparisons / 2 + boxwidth / 2 + icomparison * boxwidth
                    plt.text(xpos, np.max(teststat) + 0.1, pvalue_to_star(pval, sym='#'), family='sans-serif',
                             fontsize=5,
                             horizontalalignment='center', verticalalignment='bottom', color='black')

        plt.savefig(outputfolder + 'accuracy_pairwise_comparison_' + pair + '.png', dpi=300)
        plt.savefig(outputfolder + 'accuracy_pairwise_comparison_' + pair + '.svg')
        plt.close()


gridsize = 120

#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        path += 'output/'
        inputfile = path + 'spharm/gridsize=' + str(gridsize) + '.csv'

        if len(path.split('Synthetic')) > 1:
            id_col = 'CellID'
            cutoff = 2
            timelength = 80
            grouped = False
            C=100

        else:
            id_col = 'TrackID'
            cutoff = 10
            timelength = 10
            grouped = True
            C=100

        stat = pd.read_csv(inputfile, sep='\t', index_col=0)

        predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
                        folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=True,
                        dynamic_features=None, timelength=None, one_time_point=True, static_features='amplitude',
                        rotation_invariant=True, C=C, grouped=grouped)

        predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
                        folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=False,
                        dynamic_features='time', timelength=timelength, one_time_point=None, static_features='amplitude',
                        rotation_invariant=True, C=C, grouped=grouped)

        predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
                        folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=False,
                        dynamic_features='frequency', timelength=timelength, one_time_point=None,
                        static_features='amplitude', rotation_invariant=True, C=C, grouped=grouped)

        plot_confusion_matrix(path + 'predicted_classes/', path + 'confusion_matrix/')
        plot_accuracy_pairwise(path + 'prediction_accuracy/', path + 'accuracy_plots/')



