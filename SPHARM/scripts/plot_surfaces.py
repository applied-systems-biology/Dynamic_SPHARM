from __future__ import division

import os
import sys
import pandas as pd
from SPHARM.classes.surface import Surface
from helper_lib import filelib
from mayavi import mlab


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'
        path += 'output/'

        track_stat = pd.DataFrame()
        summary_stat = pd.DataFrame()
        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            files = os.listdir(path + 'surfaces/' + gr + '/')
            for fn in files:
                stat = pd.read_csv(path + 'surfaces/' + gr + '/' + fn, sep='\t', index_col=0)
                t1 = stat['Time'].unique()[0]
                stat = stat[stat['Time'] == t1]
                print(fn, len(stat))
                surface = Surface(data=stat)
                mesh = surface.plot_points(scale_factor=0.2)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_init.png', size=(100, 100))
                mlab.clf()

                surface.centrate()
                surface.to_spherical()
                surface.compute_spharm(grid_size=120, normalize=True)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_grid.png', size=(100, 100))
                mlab.clf()

                surface.inverse_spharm(lmax=10)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_inverse_lmax=10.png', size=(100, 100))
                mlab.clf()

                surface.inverse_spharm(lmax=None)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_inverse.png', size=(100, 100))
                mlab.clf()





















