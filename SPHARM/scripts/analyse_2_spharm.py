from __future__ import division

import re
import sys
import numpy as np
import pandas as pd
from SPHARM.lib import spharm
import SPHARM.lib.parallel as prl
from helper_lib import filelib


def extract_metadata_Tcells(inputfile, spectrum_file):

    stat = pd.read_csv(inputfile, sep='\t')

    # extract group info
    filenames = np.array(stat['Name'])
    groups = []
    Time = []
    Sample = []
    p = re.compile('[-+]?\d*\.*\d+')

    for i, fn in enumerate(filenames):
        parts = fn.split('/')
        groups.append(parts[0].split('_')[0])
        parts_sample = parts[1].split('_')
        Sample.append(parts_sample[0] + '_' + parts_sample[1])
        parts = parts[-1].split('Time')
        Time.append(p.findall(parts[-1])[0])

    stat['Group'] = groups
    stat['Time'] = Time
    stat['Sample'] = Sample
    stat.to_csv(inputfile, sep='\t')

    # compute frequency spectrum
    stat['value'] = np.array(stat['real']) + np.array(stat['imag'])*1j
    stat = stat.groupby(['Group', 'Name', 'Time', 'degree', 'Sample']).sum().reset_index()
    stat['frequency'] = np.sqrt(stat['power'])

    stat.to_csv(spectrum_file, sep='\t')


def extract_metadata_synthetic(inputfile, spectrum_file):

    stat = pd.read_csv(inputfile, sep='\t', index_col=0)

    # extract group info
    filenames = np.array(stat['Name'])
    groups = []
    cellID = []
    Time = []
    nw = []
    pw = []
    fb = []
    p = re.compile('\d*\.*\d+')

    for i, fn in enumerate(filenames):
        parts = fn.split('Time')
        nums = p.findall(parts[-2])
        nw.append(nums[-4])
        pw.append(nums[-5])
        fb.append(nums[-1])
        groups.append('NW=' + nums[-4] + '_PW=' + nums[-5] + '_FB=' + nums[-1])
        cellID.append(parts[-2][:-1])
        Time.append(p.findall(parts[-1])[-1])

    cellID = np.array(cellID)
    cellID_unique = np.unique((cellID))
    cellID_num = np.zeros(len(cellID))
    for i in range(len(cellID_unique)):
        cellID_num[np.where(cellID == cellID_unique[i])] = i + 1
    stat['Group'] = groups
    stat['CellID'] = cellID_num
    stat['Time'] = Time
    stat['NWeight'] = nw
    stat['PosWeight'] = pw
    stat['FrontBack'] = fb
    stat.to_csv(inputfile, sep='\t')

    # compute frequency spectrum
    stat['value'] = np.array(stat['real']) + np.array(stat['imag'])*1j
    stat = stat.groupby(['Group', 'Name', 'CellID', 'Time', 'degree',
                         'NWeight', 'PosWeight', 'FrontBack']).sum().reset_index()
    stat['amplitude'] = np.sqrt(stat['power'])

    stat.to_csv(spectrum_file, sep='\t')


def split_parameters(filename, outputfolder):
    filelib.make_folders([outputfolder])
    stat = pd.read_csv(filename, sep='\t', index_col=0)
    for pw in stat['PosWeight'].unique():
        for nw in stat['NWeight'].unique():
            curstat = stat[(stat['PosWeight'] == pw) & (stat['NWeight'] == nw)].reset_index()
            curstat.to_csv(outputfolder + 'PosWeight=' + str(pw) + '_NWeight=' + str(nw) + '.csv', sep='\t')

    for fb in stat['FrontBack'].unique():
        curstat = stat[stat['FrontBack'] == fb].reset_index()
        curstat.to_csv(outputfolder + 'FrontBack=' + str(fb) + '.csv', sep='\t')


kwargs = {'max_threads': 6, 'combined_tracks': True, 'rotate': False}
gridsize = 120


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        path += 'output/'
        if len(args) > 1:
            kwargs['max_threads'] = int(float(args[1]))
        extract_metadata = None

        if len(path.split('T_cells')) > 1:
            extract_metadata = extract_metadata_Tcells

        elif len(path.split('Synthetic')) > 1:
            extract_metadata = extract_metadata_synthetic

        if extract_metadata is not None:
            prl.run_parallel(process=spharm.compute_spharm, inputfolder=path + 'surfaces/',
                             outputfolder=path + 'spharm/gridsize=' + str(gridsize) + '/', extensions=['csv'],
                             grid_size=gridsize, normalize=True, **kwargs)
            filename = path + 'spharm/gridsize=' + str(gridsize) + '.csv'
            extract_metadata(inputfile=filename, spectrum_file=filename[:-4] + '_frequency_spectrum.csv')

            if len(path.split('parameters')) > 1:

                split_parameters(filename=path + 'spharm/gridsize=' + str(gridsize) + '.csv',
                                 outputfolder=path + 'spharm/gridsize=' + str(gridsize) + '_parameters/')

                split_parameters(filename=path + 'spharm/gridsize=' + str(gridsize) + '_frequency_spectrum.csv',
                                 outputfolder=path + 'spharm/gridsize=' + str(gridsize) + '_parameters_frequency_spectrum/')















