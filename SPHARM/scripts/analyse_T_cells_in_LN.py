from __future__ import division

import re
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from SPHARM.lib import spharm
import SPHARM.lib.parallel as prl
import SPHARM.lib.segmentation as sgm
from SPHARM.lib.vrml_parse import combine_with_track_data
from SPHARM.classes.image_stack import ImageStack
from SPHARM.lib import plotting as plt


def metadata_from_filename(filename, group_keys, group_names, sample_keys):
    p = re.compile('[-+]?\d*\.*\d+')
    group = None
    for i, g in enumerate(group_keys):
        if len(filename.split(g)) > 1:
            group = group_names[i]

    sample = None
    for g in sample_keys:
        if len(filename.split(g)) > 1:
            sample = g

    parts = filename.split('Time')
    time = int(p.findall(parts[-1])[0]) + 1
    return group, sample, time


def extract_metadata0(inputfile):

    stat = pd.DataFrame.from_csv(inputfile, sep='\t')
    group_keys = ['PMT3', 'PMT2']
    group_names = ['CMTMR', 'CFSE']
    sample_keys = ['Doc17', 'Doc18']

    # extract group info
    filenames = np.array(stat['Image_name'])
    Time = []

    for i, fn in enumerate(filenames):
        group, sample, time = metadata_from_filename(fn, group_keys, group_names, sample_keys)
        Time.append(time)

    stat['Time'] = Time
    stat.to_csv(inputfile, sep='\t')


def extract_metadata(inputfile, spectrum_file):

    stat = pd.DataFrame.from_csv(inputfile, sep='\t')
    group_keys = ['PMT3', 'PMT2']
    group_names = ['CMTMR', 'CFSE']
    sample_keys = ['Doc17', 'Doc18']

    # extract group info
    filenames = np.array(stat['Name'])
    groups = []
    mutants = []
    samples = []
    Time = []

    for i, fn in enumerate(filenames):
        group, sample, time = metadata_from_filename(fn, group_keys, group_names, sample_keys)
        mutants.append(group)
        print(group, sample)
        groups.append(group + '_' + sample)
        samples.append(sample)
        Time.append(time)

    stat['Group'] = groups
    stat['Time'] = Time
    stat['Sample'] = samples
    stat['Mutant'] = mutants
    stat.to_csv(inputfile, sep='\t')

    # compute frequency spectrum
    if spectrum_file is not None:
        stat['value'] = np.array(stat['real']) + np.array(stat['imag'])*1j
        stat = stat.groupby(['Group', 'Sample', 'Mutant', 'Name', 'Time', 'degree']).sum().reset_index()
        stat['frequency'] = np.sqrt(stat['power'])

        stat.to_csv(spectrum_file, sep='\t')


def combine_with_track_data_batch(inputfolder, trackfolder, outputfolder):
    files = os.listdir(inputfolder)
    trackfiles = os.listdir(trackfolder)
    group_keys = np.array(['PMT3', 'PMT2'])
    group_names = np.array(['CMTMR', 'CFSE'])
    sample_keys = np.array(['Doc17', 'Doc18'])

    for fn in files:
        print(fn)
        group, sample, time = metadata_from_filename(fn, group_keys, group_names, sample_keys)
        for trf in trackfiles:
            if len(trf.split(group)) > 1 and len(trf.split(sample)) > 1:
                extract_metadata0(inputfolder + fn)
                outputfile = outputfolder + sample + '_' + group_keys[np.where(group_names == group)[0][0]] + '/' + fn
                combine_with_track_data(inputfile=inputfolder + fn,
                                        trackfile=trackfolder + trf,
                                        outputfile=outputfile)
                stat = pd.DataFrame.from_csv(outputfile, sep='\t')
                if stat['TrackID'].iloc[0] == -1:
                    os.remove(outputfile)

    for sample in sample_keys:
        for group in group_keys:
            stat_combined = pd.DataFrame()
            files = os.listdir(outputfolder + sample + '_' + group_keys[np.where(group_keys == group)[0][0]] + '/')
            for fn in files:
                stat = pd.read_csv(outputfolder + sample + '_' +
                                   group_keys[np.where(group_keys == group)[0][0]] + '/' + fn, sep='\t', index_col=0)
                stat_combined = pd.concat([stat_combined, stat], ignore_index=True)

            stat_combined.to_csv(outputfolder + sample + '_' +
                                 group_keys[np.where(group_keys == group)[0][0]] + '.csv', sep='\t')


def overlay_tracks(**kwargs):
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../output/RGB/')
    trackfolder = kwargs.get('track_folder')
    filenames = kwargs.get('item')
    voxel_size = np.array(kwargs.get('voxel_size'))

    group_keys = kwargs.get('group_keys')
    group_names = kwargs.get('group_names')
    sample_keys = kwargs.get('sample_keys')

    stacks = []
    for i, fn in enumerate(filenames):
        if len(fn) > 0:
            stacks.append(ImageStack(inputfolder + fn))

    data = np.zeros(stacks[0].data.shape + (3,))
    for i in range(len(stacks)):
        data[:, :, :, i] = stacks[i].data

    stacks[0].data = data

    trackfiles = os.listdir(trackfolder)
    centers = np.zeros(np.array(stacks[0].data.shape)[:-1])
    for fn in filenames:
        if fn != '':
            group, sample, time = metadata_from_filename(fn, group_keys, group_names, sample_keys)
            for trf in trackfiles:
                if len(trf.split(group)) > 1 and len(trf.split(sample)) > 1:
                    trackstat = pd.read_csv(trackfolder + trf, sep='\t')
                    trackstat = trackstat[trackstat['Time'] == time]
                    x = np.int_(np.round_(np.array(trackstat['Position X'])/voxel_size[2]))
                    y = np.int_(np.round_(np.array(trackstat['Position Y'])/voxel_size[1]))
                    z = np.int_(np.round_(np.array(trackstat['Position Z'])/voxel_size[0]))
                    centroids = np.array([z,y,x]).transpose().reshape((len(trackstat), 3))
                    centers[tuple(centroids.transpose())] = 255.

    sigma = 0.25 / voxel_size
    centers = ndimage.gaussian_filter(centers, sigma)
    print(centers.max(), centers.min())

    stacks[0].data[np.where(centers > 0)] = (255, 255, 255)
    stacks[0].save_max_projection(outputfolder + filenames[0])


path = '../../../Data/T_cells_in_LN/'
inputfolder = path + 'input/'

if os.path.exists(path):
    kwargs = {'max_threads': 3, 'mincellrad': 3, 'voxel_size': [4, 0.478, 0.478],
              'channelcodes': ['PMT1', 'PMT2', 'PMT3', 'PMT4'],
              'sigmas': [0.5] * 4, 'percentiles': [100, 99.99, 99.99, 100], 'thresholds': [0, 50, 100, 0]}

    prl.run_parallel(process=sgm.make_metadata_files, inputfolder=inputfolder, **kwargs)


    prl.run_parallel(process=sgm.preprocess, inputfolder=inputfolder,
                     outputfolder=path + 'output/preprocessed/', **kwargs)

    prl.run_parallel(process=plt.plot_maxprojections,
                     inputfolder=path + 'output/preprocessed/',
                     outputfolder=path + 'output/preprocessed_maxproj/')

    prl.run_parallel(process=sgm.mergeRGB, inputfolder=path + 'output/preprocessed/',
                     outputfolder=path + 'output/preprocessed_RGB/', channels=['PMT3', 'PMT2', 'PMT1'])

    prl.run_parallel(process=plt.plot_maxprojections,
                     inputfolder=path + 'output/preprocessed_RGB/',
                     outputfolder=path + 'output/preprocessed_RGB_maxproj/')


    prl.run_parallel(process=sgm.unmix, channels=['PMT3', 'PMT2'],
                     inputfolder=path + 'output/preprocessed/',
                     outputfolder=path + 'output/unmixed/')

    prl.run_parallel(process=plt.plot_maxprojections,
                     inputfolder=path + 'output/unmixed/',
                     outputfolder=path + 'output/unmixed_maxproj/')

    prl.run_parallel(process=sgm.mergeRGB, inputfolder=path + 'output/unmixed/',
                     outputfolder=path + 'output/unmixed_RGB/', channels=['PMT3', 'PMT2', 'PMT1'])

    prl.run_parallel(process=plt.plot_maxprojections,
                     inputfolder=path + 'output/unmixed_RGB/',
                     outputfolder=path + 'output/unmixed_RGB_maxproj/')

    prl.run_parallel(process=sgm.segment,
                     inputfolder=path + 'output/unmixed/',
                     outputfolder=path + 'output/segmented/',
                     channelcodes=['PMT1', 'PMT2', 'PMT3', 'PMT4'],
                     thresholds=[None, 50, 100, None], to_label=False,
                     track_folder=path + 'tracks/')

    prl.run_parallel(process=overlay_tracks, inputfolder=path + 'output/unmixed/',
                     outputfolder=path + 'output/unmixed_RGB_tracks/', channels=['PMT3', 'PMT2', 'PMT1'],
                     track_folder=path + 'tracks/', group_keys=['PMT3', 'PMT2'], group_names=['CMTMR', 'CFSE'],
                     sample_keys=['Doc17', 'Doc18'], **kwargs)

    prl.run_parallel(process=sgm.mergeRGB, inputfolder=path + 'output/segmented/',
                     outputfolder=path + 'output/segmented_RGB/', channels=['PMT3', 'PMT2', 'PMT1'])

    prl.run_parallel(process=plt.plot_maxprojections,
                     inputfolder=path + 'output/segmented_RGB/',
                     outputfolder=path + 'output/segmented_RGB_maxproj/')

    prl.run_parallel(process=sgm.segment,
                     inputfolder=path + 'output/unmixed/',
                     outputfolder=path + 'output/segmented_labeled/',
                     channelcodes=['PMT1', 'PMT2', 'PMT3', 'PMT4'],
                     thresholds=[None, 50, 100, None], to_label=True,
                     track_folder=path + 'tracks/')

    prl.run_parallel(process=sgm.extract_surfaces,
                     inputfolder=path + 'output/segmented_labeled/',
                     outputfolder=path + 'output/surfaces_all/',
                     channelcodes=['PMT2', 'PMT3'], combine=False)

    combine_with_track_data_batch(inputfolder=path + 'output/surfaces_all/', trackfolder=path + 'tracks/',
                                  outputfolder=path + 'output/surfaces_selected/')

    sgm.split_to_surfaces_batch(path + 'output/surfaces_selected/', path + 'output/surfaces/')

    gridsize = 60
    prl.run_parallel(process=spharm.compute_spharm, inputfolder=path + 'output/surfaces/',
                     outputfolder=path + 'output/spharm/', extensions=['csv'],
                     grid_size=gridsize)

    filename = path + 'output/spharm.csv'

    extract_metadata(inputfile=filename,
                     spectrum_file=filename[:-4] + '_frequency_spectrum.csv')

    for group in ['Group', 'Sample', 'Mutant']:

        plt.plot_average_heatmaps(inputfile=filename, outputfolder=path+'output/plots_' + group + '/heatmaps/', group=group)
        plt.plot_average_spectra(inputfile=filename[:-4]+'_frequency_spectrum.csv',
                                  outputfolder=path+'output/plots_' + group + '/frequencies/', value='frequency', group=group)

        plt.plot_effect_size(inputfile=filename[:-4]+'_frequency_spectrum.csv',
                                  outputfolder=path+'output/plots_' + group + '/effect_size/',
                                  value='frequency', group=group)

        plt.plot_pairplots(inputfile=filename[:-4]+'_frequency_spectrum.csv',
                                outputfolder=path+'output/plots_' + group + '/pairplots/', group=group)



