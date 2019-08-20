from __future__ import division

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

import mkl
from SPHARM.lib import classification
from helper_lib import filelib
import helper_lib.parallel as prl

import warnings
warnings.simplefilter(action='ignore', category=Warning)
mkl.set_num_threads(1)


def compare_parameters_parallel(cutoffs, timelengths, rotation_invariant, **kwargs):
    items = []
    if rotation_invariant:
        static_features = ['amplitude']
    else:
        static_features = ['amplitude', 'real_imag']

    for cutoff in cutoffs:

        static = True
        dynamic_features = None
        timelength = None

        for static_feature in static_features:
            items.append([cutoff, static, dynamic_features, timelength,
                          static_feature, rotation_invariant])

        static = False
        for timelength in timelengths:
            for dynamic_features in ['time', 'frequency']:
                for static_feature in static_features:
                    items.append([cutoff, static, dynamic_features, timelength,
                                  static_feature, rotation_invariant])

    if kwargs.pop('debug', False) is True:
        kwargs['item'] = items[0]
        kwargs.pop('max_threads')
        print(kwargs['item'])
        compare_parameters(**kwargs)
    else:
        kwargs['items'] = items
        prl.run_parallel(process=compare_parameters, **kwargs)

    filelib.combine_statistics(kwargs.get('folder_accuracy'))


def compare_parameters(item, inputfile, folder_accuracy, group='Group', parameters=False,
                       id_col='TrackID', grouped=False):
    filelib.make_folders([folder_accuracy])

    cutoff, static, dynamic_features, timelength, static_features, rotation_invariant = item
    params = dict({'Cutoff': cutoff,
                   'Static': static,
                   'Dynamic_features': dynamic_features,
                   'Time length': timelength,
                   'Static_features': static_features,
                   'Rotation_invariant': rotation_invariant})

    if parameters:
        files = os.listdir(inputfile)
    else:
        files = ['']

    for fn in files:
        outputfile = folder_accuracy + fn
        for key in params.keys():
            outputfile += key + '=' + str(params[key]) + '_'
        if not os.path.exists(outputfile[:-1] + '.csv'):
            stat = pd.read_csv(inputfile + fn, sep='\t', index_col=0)
            if cutoff is not None:
                stat = stat[stat['degree'] <= cutoff]
            features, classes, \
            names, groups, samples = classification.extract_features(stat,
                                                                     cell_id=id_col,
                                                                     group=group,
                                                                     static=static,
                                                                     dynamic_features=dynamic_features,
                                                                     timelength=timelength,
                                                                     static_features=static_features,
                                                                     rotation_invariant=rotation_invariant)[:]

            accuracy = pd.DataFrame()
            for C in [0.1, 1., 10., 100., 1000.]:
                if grouped:
                    curaccuracy = classification.predict_group_shuffle_split(features, classes, C=C,
                                                                             nsplits=100, test_size=1, groups=samples,
                                                                             random_state=0)
                else:
                    curaccuracy = classification.predict_shuffle_split(features, classes, C=C,
                                                                       nsplits=100, test_size=2./7, random_state=0)
                curaccuracy['C'] = C
                accuracy = pd.concat([accuracy, curaccuracy], ignore_index=True)

            for key in params.keys():
                accuracy[key] = params[key]

            accuracy.to_csv(outputfile[:-1] + '.csv', sep='\t')


def plot_accuracy(inputfile, outputfolder):
    filelib.make_folders([outputfolder])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat.loc[:, 'Cutoff'] = np.array(stat['Cutoff']).astype(str)
    stat.loc[stat[stat['Cutoff'] == 'nan'].index, 'Cutoff'] = str(60)
    stat.loc[:, 'Cutoff'] = np.array(stat['Cutoff']).astype(float).astype(int)
    stat = stat.sort_values(['Cutoff', 'Time length']).reset_index()
    stat.loc[:, 'Dynamic_features'] = np.array(stat['Dynamic_features']).astype(str)
    stat.loc[stat[stat['Dynamic_features'] == 'nan'].index, 'Dynamic_features'] = 'static'
    if 'Rotation_invariant' in stat.columns:
        stat = stat.assign(features=stat['Dynamic_features'] + '_' + stat['Static_features']
                                    + '_rot_invar=' + stat['Rotation_invariant'].astype(str))
    else:
        stat = stat.assign(features=stat['Dynamic_features'] + '_' + stat['Static_features'])

    for c in stat['C'].unique():
        for features in stat['features'].unique():
            curstat = stat[(stat['features'] == features)&(stat['C'] == c)]
            if str(curstat['Static'].iloc[0]) == 'True':
                if 'One_time_point' in stat.columns:
                    hue = 'One_time_point'
                else:
                    hue = None
            else:
                hue = 'Time length'

            sns.boxplot(x='Cutoff', y='Accuracy', hue=hue, data=curstat)
            plt.savefig(outputfolder + features + '_C=' + str(c) + '.png')
            plt.close()

    stat = stat.sort_values(['C', 'features']).reset_index()
    for cutoff in stat['Cutoff'].unique():
        stat_static = stat[(stat['Static'] == True) & (stat['Cutoff'] == cutoff)]
        stat_dynamic = stat[(stat['Static'] == False) & (stat['Cutoff'] == cutoff)]

        if 'One_time_point' in stat.columns:

            for otp in stat_static['One_time_point'].unique():
                curstat = stat_static[stat_static['One_time_point'] == otp]

                sns.boxplot(x='C', y='Accuracy', hue='features', data=curstat)
                plt.savefig(outputfolder + 'Static_one_time_point=' + str(otp) + '_Cutoff=' + str(cutoff) + '.png')
                plt.close()
        else:
            sns.boxplot(x='C', y='Accuracy', hue='features', data=stat_static)
            plt.savefig(outputfolder + 'Static_Cutoff=' + str(cutoff) + '.png')
            plt.close()

        for tl in stat_dynamic['Time length'].unique():
            curstat = stat_dynamic[stat_dynamic['Time length'] == tl]
            sns.boxplot(x='C', y='Accuracy', hue='features', data=curstat)
            plt.savefig(outputfolder + 'Dynamic_Time_length=' + str(tl) + '_Cutoff=' + str(cutoff) + '.png')
            plt.close()


def plot_accuracy_selected(inputfile, outputfolder):
    filelib.make_folders([outputfolder])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat.loc[:, 'Cutoff'] = np.array(stat['Cutoff']).astype(str)
    stat.loc[stat[stat['Cutoff'] == 'nan'].index, 'Cutoff'] = str(60)
    stat.loc[:, 'Cutoff'] = np.array(stat['Cutoff']).astype(float).astype(int)
    stat = stat.sort_values(['Cutoff', 'Time length']).reset_index()
    stat.loc[:, 'Dynamic features'] = np.array(stat['Dynamic_features']).astype(str)
    stat.loc[stat[stat['Dynamic features'] == 'nan'].index, 'Dynamic features'] = 'static'
    stat = stat[stat['Rotation_invariant'] == True]

    curstat = stat[stat['Static'] == True]

    palette = 'Set1'

    plt.figure(figsize=(4, 4))
    sns.boxplot(x='Cutoff', y='Accuracy', hue='C', data=curstat, palette=palette)
    sns.despine()
    plt.xlabel('$l_{max}$')
    plt.title('Static')
    margins = {'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.13}
    plt.subplots_adjust(**margins)
    plt.savefig(outputfolder + 'Static.png')
    plt.savefig(outputfolder + 'Static.svg')
    plt.close()

    curstat = stat[stat['Static'] == False]
    curstat['Time length'] = curstat['Time length'].astype(int)

    for dyn in curstat['Dynamic_features'].unique():
        plt.figure(figsize=(4, 4))
        sns.boxplot(x='Cutoff', y='Accuracy', hue='C',
                      data=curstat[curstat['Dynamic features'] == dyn], palette=palette)
        sns.despine()
        plt.xlabel('$l_{max}$')
        margins = {'left': 0.15, 'right': 0.95, 'top': 0.9, 'bottom': 0.13}
        plt.subplots_adjust(**margins)
        plt.title('Dynamic features = ' + dyn)
        plt.ylim(0.55, 1.02)
        plt.savefig(outputfolder + 'Dynamic_' + dyn + '.png')
        plt.savefig(outputfolder + 'Dynamic_' + dyn + '.svg')
        plt.close()

    summary = curstat.groupby(['C', 'Dynamic features', 'Cutoff']).mean().reset_index()
    summary = summary.sort_values(['Accuracy'], ascending=False)
    C = summary.iloc[0]['C']
    cutoff = 50
    curstat = curstat[(curstat['C'] == C) & (curstat['Cutoff'] == cutoff)]
    plt.figure(figsize=(3, 4))
    sns.boxplot(x='Time length', y='Accuracy', hue='Dynamic_features', data=curstat, palette=palette)
    sns.despine()
    plt.xlabel('Time length (frames)')
    margins = {'left': 0.22, 'right': 0.95, 'top': 0.9, 'bottom': 0.13}
    plt.subplots_adjust(**margins)
    plt.title('C = ' + str(C) + ', Cutoff = ' + str(cutoff))
    plt.savefig(outputfolder + 'Dynamic_C=' + str(C) + '_cutoff=' + str(cutoff) + '.png')
    plt.savefig(outputfolder + 'Dynamic_C=' + str(C) + '_cutoff=' + str(cutoff) + '.svg')
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

        if len(path.split('Synthetic')) > 1:
            id_col = 'CellID'
            grouped = False
        else:
            id_col = 'TrackID'
            grouped = True

        if len(path.split('parameters')) > 1:
            parameters = True
            inputfile = path + 'spharm/gridsize=' + str(gridsize) + '_parameters/'
        else:
            parameters = False
            inputfile = path + 'spharm/gridsize=' + str(gridsize) + '.csv'

        rotation_invariant = True

        compare_parameters_parallel(inputfile=inputfile,
                                    folder_accuracy=path + 'cross_validation_accuracy/',
                                    cutoffs=[3, 5, 10, 20, 30, 40, 50, None],
                                    timelengths=[5, 10, 20, 30, 50, 81],
                                    max_threads=20,
                                    parameters=parameters,
                                    id_col=id_col,
                                    rotation_invariant=rotation_invariant,
                                    debug=False, grouped=grouped)

        plot_accuracy_selected(path + 'cross_validation_accuracy.csv', path + 'cross_validation_accuracy_plots_selected/')




