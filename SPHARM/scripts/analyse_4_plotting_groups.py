from __future__ import division

import mkl
mkl.set_num_threads(1)

import sys
import os

from SPHARM.lib import plotting as plt


def compare_parameters(inputfolder, outputfolder):
    files = os.listdir(inputfolder)
    print(files)
    for fn in files:
        plt.plot_heatmap_difference(inputfile=inputfolder + fn, cutoff=5,
                                    outputfolder=outputfolder +fn[:-4] + '/heatmap_difference/')


def compare_parameters_spectra(inputfolder, outputfolder):
    files = os.listdir(inputfolder)
    print(files)
    for fn in files:
        plt.plot_effect_size(inputfile=inputfolder+fn,
                             outputfolder=outputfolder + fn[:-4] + '/effect_size/', value='amplitude', cutoff=8)

        plt.plot_pairplots(inputfile=inputfolder+fn,
                           outputfolder=outputfolder + fn[:-4] + '/pairplots/', cutoff=5)


gridsize = 120


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        path += 'output/'

        filename = path + 'spharm/gridsize=' + str(gridsize) + '.csv'
        plt.plot_average_heatmaps(inputfile=filename, cutoff=5, logscale=True,
                                  outputfolder=path + 'plots/average_heatmaps/gridsize=' + str(gridsize) + '_')
        plt.plot_average_spectra(inputfile=filename[:-4] + '_frequency_spectrum.csv', cutoff=8,
                                 outputfolder=path + 'plots/average_spectra/gridsize=' + str(gridsize) + '_')
        plt.plot_average_frequency_heatmaps(inputfile=filename, cutoff=10, logscale=True,
                                            outputfolder=path + 'plots/average_frequency_heatmaps/gridsize=' + str(
                                                gridsize) + '_')
        plt.plot_mean_abs_derivative(inputfile=filename, cutoff=10,
                                     outputfolder=path + 'plots/average_abs_derivative/gridsize=' + str(gridsize) + '_')
        plt.plot_inverse_shapes(inputfile=filename,
                                outputfolder=path + 'plots/inverse_shapes/gridsize=' + str(gridsize) + '_')

        if len(path.split('parameters')) > 1:

            compare_parameters_spectra(inputfolder=path + 'spharm/gridsize=' + str(gridsize)
                                                       + '_parameters_frequency_spectrum/',
                                           outputfolder=path+'plots/parameters_comparison/gridsize='+str(gridsize)+'/')

            compare_parameters(inputfolder=path + 'spharm/gridsize=' + str(gridsize) + '_parameters/',
                               outputfolder=path+'plots/parameters_comparison/gridsize='+str(gridsize)+'/')

        else:
            plt.plot_heatmap_difference(inputfile=filename,
                                        outputfolder=path + 'plots/heatmap_difference/gridsize=' + str(gridsize))
            plt.plot_effect_size(inputfile=filename[:-4] + '_frequency_spectrum.csv',
                                 outputfolder=path + 'plots/effect_size/gridsize=' + str(gridsize), value='frequency')
            plt.plot_pairplots(inputfile=filename[:-4] + '_frequency_spectrum.csv',
                               outputfolder=path + 'plots/pairplots/gridsize=' + str(gridsize))



