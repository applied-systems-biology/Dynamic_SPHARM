from __future__ import division

import numpy as np
import pylab as plt
import os

from helper_lib import filelib
from SPHARM.classes.orbital import Orbital


def test_single_orbitals(gridsize, path):
    value = 'amplitude'
    filelib.make_folders([path + 'surfaces/', path + 'heatmaps/', path + 'frequencies/'])
    for m in range(5):
        for n in range(-m, m+1):

            orbital = Orbital(grid_shape=(gridsize, gridsize), m=m, n=n, amplitude=1)
            orbital.name = 'm=' + str(m) + '_n=' + str(n)

            sp = orbital.compute_spharm(grid_size=gridsize)
            sp.heatmap(value=value, cutoff=5).savefig(path + 'heatmaps/' + orbital.name + '.png')
            plt.clf()
            sp.frequency_plot(value=value, cutoff=5).savefig(path + 'frequencies/' + orbital.name + '.png')
            plt.clf()

            orbital.Rgrid = orbital.Rgrid.real
            mesh = orbital.plot_surface(points=False)
            mesh.save(path + 'surfaces/' + orbital.name + '.png', size=(200, 200))


def test_combined_orbitals(gridsize, path):
    value = 'amplitude'
    filelib.make_folders([path + 'surfaces/', path + 'heatmaps/', path + 'frequencies/'])
    combined = [[(0, 0, 1), (2, -1, 0.1)],
                [(0, 0, 1), (2, -1, 0.5), (4, 3, 1)],
                [(0, 0, 1), (4, 3, 0.5)],
                [(0, 0, 1), (1, 0, 1)]]
    for set in combined:
        set = np.array(set)
        m = set[:, 0]
        n = set[:, 1]
        amplitude = set[:, 2]

        orbital = Orbital(grid_shape=(gridsize, gridsize), m=m, n=n, amplitude=amplitude)
        orbital.name = 'm=' + str(m) + '_n=' + str(n) + '_amplitude=' + str(amplitude)

        sp = orbital.compute_spharm(grid_size=gridsize)
        sp.heatmap(value=value, cutoff=5).savefig(path + 'heatmaps/' + orbital.name + '.png')
        plt.clf()
        sp.frequency_plot(value=value).savefig(path + 'frequencies/' + orbital.name + '.png')
        plt.clf()

        orbital.Rgrid = orbital.Rgrid.real
        mesh = orbital.plot_surface(points=False)
        mesh.save(path + 'surfaces/' + orbital.name + '.png', size=(200, 200))

path = '../../../Data/Orbitals/'
if os.path.exists(path):
    test_single_orbitals(50, path + 'Single_orbitals/')
    test_combined_orbitals(50, path + 'Combined_orbitals/')











