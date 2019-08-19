from __future__ import division

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy.stats import ranksums

import mkl
from SPHARM.lib import classification
from SPHARM.lib.confusion_matrix_pretty_print import plot_confusion_matrix_from_data
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


def predict_classes(folder_accuracy, folder_predicted, inputfile=None, stat=None, group='Group',
                    id_col='TrackID', static=True, cutoff=None, C=1, grouped=False, **kwargs):
    folders = [folder_accuracy, folder_predicted]
    filelib.make_folders(folders)

    if stat is None:
        stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    if cutoff is not None:
        stat = stat[stat['degree'] <= cutoff]

    print('Feature extraction')
    features, classes, names, groups, samples = classification.extract_features(stat,
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
            class_names = np.unique(groups[ind])

            if grouped:
                curaccuracy = classification.predict_group_shuffle_split(features[ind], classes[ind], C=C,
                                                                         nsplits=150, test_size=1, groups=samples[ind],
                                                                         random_state=0)
            else:
                curaccuracy = classification.predict_shuffle_split(features[ind], classes[ind], C=C,
                                                                   nsplits=150, test_size=2./7, random_state=0)

            curaccuracy['Comparison'] = class_names[0] + ' vs ' + class_names[1]
            accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)
    else:
        class_names = np.unique(groups)

        if grouped:
            curaccuracy = classification.predict_group_shuffle_split(features, classes, C=C,
                                                                     nsplits=150, test_size=1, groups=samples,
                                                                     random_state=0)
        else:
            curaccuracy = classification.predict_shuffle_split(features, classes, C=C,
                                                               nsplits=150, test_size=2./7, random_state=0)

        curaccuracy['Comparison'] = class_names[0] + ' vs ' + class_names[1]
        accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

    print('Prediction mock')
    for cl in np.unique(classes):  # total number of comparisons = 150 = 3 classes x 5 iterations x 10 splits
        ind = np.where(classes == cl)

        curfeatures = features[ind]

        if grouped:
            cursamples = samples[ind]
            unique_groups = np.unique(cursamples)
            for i_iter in range(5):
                group_ind = np.random.choice(len(unique_groups), int(len(unique_groups)/2), replace=False)

                curclasses = np.zeros(len(cursamples))
                for i in group_ind:
                    curclasses[np.where(cursamples == unique_groups[i])] = 1

                curaccuracy = classification.predict_group_shuffle_split(curfeatures, curclasses, C=C, nsplits=10,
                                                                         test_size=1, groups=cursamples,
                                                                         random_state=0)
                curaccuracy['Comparison'] = 'Control'
                accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

        else:
            for i_iter in range(5):
                group_ind = np.random.choice(len(curfeatures), int(len(curfeatures)/2), replace=False)

                curclasses = np.zeros(len(curfeatures))
                curclasses[group_ind] = 1

                curaccuracy = classification.predict_shuffle_split(curfeatures, curclasses, C=C, nsplits=10,
                                                                   test_size=2./7, random_state=0)
                curaccuracy['Comparison'] = 'Control'
                accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

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
        # stat = stat.sort_values('Group')
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
        stat.at[i, 'Comparison'] = stat.iloc[i]['Comparison'].replace('NW=6_PW=3_', '')
    stat['Features'] = ''
    stat.at[stat[stat['Static'] == True].index, 'Features'] = 'Static'
    stat.at[stat[(stat['Static'] == False) & (stat['dynamic_features'] == 'time')].index, 'Features'] = 'Dynamic\n time'
    stat.at[stat[(stat['Static'] == False)
                 & (stat['dynamic_features'] == 'frequency')].index, 'Features'] = 'Dynamic\n frequency'

    stat = stat.sort_values(['Features', 'Comparison'], ascending=False)
    plt.figure(figsize=(4, 4))
    margins = {'left': 0.2, 'right': 0.95, 'top': 0.9, 'bottom': 0.13}
    plt.subplots_adjust(**margins)
    sns.boxplot(x='Features', y='Accuracy', hue='Comparison', data=stat, palette='Set1')
    sns.despine()
    plt.xlabel('')
    ncomparisons = len(stat['Comparison'].unique())

    for ifeatures, feature in enumerate(stat['Features'].unique()):
        curstat = stat[stat['Features'] == feature]
        control_stat = curstat[curstat['Comparison'] == 'Control']['Accuracy']
        for icomparison, comparison in enumerate(stat['Comparison'].unique()):
            if comparison != 'Control':
                teststat = curstat[curstat['Comparison'] == comparison]['Accuracy']
                pval = ranksums(control_stat, teststat)[1]
                boxwidth = 0.8/ncomparisons
                xpos = ifeatures - boxwidth*ncomparisons/2 + boxwidth/2 + icomparison*boxwidth
                plt.text(xpos, np.max(teststat)*1.02, pvalue_to_star(pval), family='sans-serif', fontsize=8,
                         horizontalalignment='center', verticalalignment='bottom', color='black')

    for icomparison, comparison in enumerate(stat['Comparison'].unique()):
        if comparison != 'Control':
            curstat = stat[stat['Comparison'] == comparison]
            control_stat = curstat[curstat['Features'] == 'Static']['Accuracy']
            for ifeatures, feature in enumerate(stat['Features'].unique()):
                teststat = curstat[curstat['Features'] == feature]['Accuracy']
                if np.mean(teststat) > 0.55:
                    pval = ranksums(control_stat, teststat)[1]
                    boxwidth = 0.8 / ncomparisons
                    xpos = ifeatures - boxwidth * ncomparisons / 2 + boxwidth / 2 + icomparison * boxwidth
                    plt.text(xpos, np.max(teststat)*1.06, pvalue_to_star(pval, sym='$'), family='sans-serif', fontsize=5,
                             horizontalalignment='center', verticalalignment='bottom', color='black')

            control_stat = curstat[curstat['Features'] == 'Dynamic\n time']['Accuracy']
            teststat = curstat[curstat['Features'] == 'Dynamic\n frequency']['Accuracy']
            ifeatures = 2
            if np.mean(teststat) > 0.55:
                pval = ranksums(control_stat, teststat)[1]
                boxwidth = 0.8/ncomparisons
                xpos = ifeatures - boxwidth*ncomparisons/2 + boxwidth/2 + icomparison*boxwidth
                plt.text(xpos, np.max(teststat)*1.1, pvalue_to_star(pval, sym='#'), family='sans-serif', fontsize=5,
                         horizontalalignment='center', verticalalignment='bottom', color='black')

    plt.savefig(outputfolder + 'accuracy_pairwise_comparison.png', dpi=300)
    plt.savefig(outputfolder + 'accuracy_pairwise_comparison.svg')
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

        cutoff = 50
        C = 10

        if len(path.split('Synthetic')) > 1:
            id_col = 'CellID'
            timelength = 81
            grouped = False

        else:
            id_col = 'TrackID'
            timelength = 10
            grouped = True

        # stat = pd.read_csv(inputfile, sep='\t', index_col=0)

        # predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
        #                 folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=True,
        #                 dynamic_features=None, timelength=None, one_time_point=True, static_features='amplitude',
        #                 rotation_invariant=True, C=C, grouped=grouped)
        #
        # predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
        #                 folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=False,
        #                 dynamic_features='frequency', timelength=timelength, one_time_point=None,
        #                 static_features='amplitude', rotation_invariant=True, C=C, grouped=grouped)
        #
        # predict_classes(stat=stat, folder_accuracy=path + 'prediction_accuracy/',
        #                 folder_predicted=path + 'predicted_classes/', id_col=id_col, cutoff=cutoff, static=False,
        #                 dynamic_features='time', timelength=timelength, one_time_point=None, static_features='amplitude',
        #                 rotation_invariant=True, C=C, grouped=grouped)

        # plot_confusion_matrix(path + 'predicted_classes/', path + 'confusion_matrix/')
        plot_accuracy_pairwise(path + 'prediction_accuracy/', path + 'accuracy_plots/')



