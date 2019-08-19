from __future__ import division

import numpy as np
import pandas as pd
import pylab as plt

from helper_lib import filelib
from SPHARM.classes.ellipsoid import Ellipsoid


def plot_heatmaps(size, rotation, gridsize, path, value='power', cutoff=None, normalize=False, ri=False):

    filelib.make_folders([path + 'heatmaps/', path + 'frequencies/'])
    ell = Ellipsoid(grid_shape=(gridsize, gridsize), size=size, rotation=rotation)
    sp = ell.compute_spharm(grid_size=gridsize, normalize=normalize, normalization_method='zero-component', ri=ri)
    sp.save_to_csv(path + 'heatmaps/' + ell.name + '.csv')
    sp.heatmap(value=value, cutoff=cutoff).savefig(path + 'heatmaps/' + ell.name + '.png')
    plt.clf()
    sp.frequency_plot(value=value).savefig(path + 'frequencies/' + ell.name + '.png')
    plt.clf()


def test_grid_size_invariance(path):

    value = 'power'
    size = (3, 1, 1)
    rotation = (0, 0)

    for n in range(2, 20):
        gridsize = 2*n
        plot_heatmaps(size, rotation, gridsize, path + 'not_normalized/', value=value, normalize=False)
        plot_heatmaps(size, rotation, gridsize, path + 'normalized/', value=value, normalize=True)


def test_size_invariance(path):

    value = 'power'
    size = (2, 1, 1)
    rotation = (0, 0)
    gridsize = 20
    for k in range(1, 10):
        Size = tuple(np.array(size)*k)
        plot_heatmaps(Size, rotation, gridsize, path + 'not_normalized/', value=value, normalize=False)
        plot_heatmaps(Size, rotation, gridsize, path + 'normalized/', value=value, normalize=True)


def test_rotation_invariance(path, ri=False):

    value = 'amplitude2'
    size = (10, 1, 1)
    gridsize = 50
    n_theta = 4
    n_phi = 4
    for t in range(n_theta):
        for p in range(n_phi):
            rotation = (np.pi/n_theta*t, 2*np.pi/n_phi*p)
            plot_heatmaps(size, rotation, gridsize, path + 'not_normalized/', value=value,
                          normalize=False, cutoff=5, ri=ri)
            plot_heatmaps(size, rotation, gridsize, path + 'normalized/', value=value,
                          normalize=True, cutoff=5, ri=ri)


def test_eccentricity_dependence(path):

    value = 'power'
    rotation = (0, 0)
    gridsize = 16
    for k in range(1, 21):
        size = (1, 1, k)
        plot_heatmaps(size, rotation, gridsize, path + 'prolate/', value=value, normalize=True)

    for k in range(1, 21):
        size = (1, k, k)
        plot_heatmaps(size, rotation, gridsize, path + 'oblate/', value=value, normalize=True)


def prolateness(r):
    r = np.array(r)
    n = len(r)
    ravg = np.mean(r**2)
    pr = 4./9 * (np.prod(r**2 - ravg)/ravg**n + 0.25)
    return pr


def plot_eccentricity_test(inputfile):
    stat = pd.DataFrame.from_csv(inputfile, sep='\t')
    stat = stat.sort_values(['degree', 'Prolateness'])

    for d in stat['degree'].unique():
        curstat = stat[stat['degree'] == d]
        plt.scatter(curstat['Eccentricity'], curstat['power'], c=curstat['Prolateness'], cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_label('Prolateness')
        plt.xlabel('Eccentricty')
        plt.ylabel('Power')
        plt.savefig(inputfile[:-4] + '_eccentricity_deg=%03d.png'%d)
        plt.close()


def test_eccentricity(path, gridsize):

    filelib.make_folders([path])
    sizes = []
    for i in range(1, 21):
        for j in range(1, 21):
            sizes.append((1, i, j))

    stat = pd.DataFrame()
    for size in sizes:
        ell = Ellipsoid(grid_shape=(gridsize, gridsize), size=size, rotation=(0,0))
        sp = ell.compute_spharm(grid_size=gridsize, normalize=True)

        sp.compute_frequency_spectrum()
        curstat = sp.frequency_spectrum
        curstat['Eccentricity'] = np.max(size)/np.min(size)
        curstat['Prolateness'] = prolateness(size)
        stat = pd.concat([stat, curstat], ignore_index=True)

    stat.to_csv(path + 'stat.csv', sep='\t')
    plot_eccentricity_test(path + 'stat.csv')


path = '../../../Data/Ellipsoids/'

import os
if os.path.exists(path):
    # test_grid_size_invariance(path + 'Grid_size_invariance/')
    # test_size_invariance(path + 'Size_invariance/')
    test_rotation_invariance(path + 'Rotation_invariance_RI/', ri=True)
    # test_eccentricity_dependence(path + 'Eccentricity_test/')
    # test_eccentricity(path + 'Eccentricity_test_quantitative/', gridsize=60)




