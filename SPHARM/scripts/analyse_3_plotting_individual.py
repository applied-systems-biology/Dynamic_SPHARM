from __future__ import division

import mkl
mkl.set_num_threads(1)

import sys

from SPHARM.lib import plotting as plt


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
        else:
            id_col = 'TrackID'

        plt.plot_individual_heatmaps(inputfolder=path + 'spharm/gridsize=' + str(gridsize) + '/',
                                     outputfolder=path + 'plots/individual_heatmaps/gridsize=' + str(gridsize) + '/',
                                     cutoff=5, logscale=True)
        plt.plot_spectra(inputfolder=path + 'spharm/gridsize=' + str(gridsize) + '/',
                         outputfolder=path + 'plots/individual_spectra/gridsize=' + str(gridsize) + '/')

        plt.plot_individual_time_heatmaps(inputfile=path + 'spharm/gridsize=' + str(gridsize) + '.csv',
                                          outputfolder=path + 'plots/individual_time_heatmaps/gridsize='
                                                       + str(gridsize) + '/', logscale=True, cutoff=10, id_col=id_col)
        plt.plot_individual_frequency_heatmaps(inputfile=path + 'spharm/gridsize=' + str(gridsize) + '.csv',
                                               outputfolder=path + 'plots/individual_frequency_heatmaps/gridsize='
                                                            + str(gridsize) + '/', logscale=True, cutoff=10,
                                               id_col=id_col)
        plt.plot_individual_derivative_heatmaps(inputfile=path + 'spharm/gridsize=' + str(gridsize) + '.csv',
                                                outputfolder=path + 'plots/individual_derivative_heatmaps/gridsize='
                                                             + str(gridsize) + '/', cutoff=10, id_col=id_col)









