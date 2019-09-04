from __future__ import division

import os
import sys
import pandas as pd
import numpy as np
from SPHARM.classes.surface import Surface
from SPHARM.classes.image_stack import ImageStack
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
        groups = os.listdir(path + 'stacks/')
        for gr in groups:
            print(gr)
            files = os.listdir(path + 'stacks/' + gr + '/')
            for fn in files:
                print(fn)
                stack = ImageStack(path + 'stacks/' + gr + '/' + fn)
                stack.filename = fn
                stack.extract_surfaces(path + 'surfaces/' + gr + '/')

        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            files = os.listdir(path + 'surfaces/' + gr + '/')
            for fn in files:
                print(fn)
                stat = pd.read_csv(path + 'surfaces/' + gr + '/' + fn, sep='\t', index_col=0)
                stat['Time'] = 1
                surface = Surface(data=stat)
                mesh = surface.plot_points(scale_factor=0.5)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_init.png', size=(800, 800))
                mlab.clf()

                surface.centrate()
                surface.to_spherical()
                surface.compute_spharm(grid_size=120, normalize=True)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_grid.png', size=(800, 800))
                mlab.clf()

                surface.inverse_spharm(lmax=10)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_inverse_lmax=10.png', size=(800, 800))
                mlab.clf()

                surface.inverse_spharm(lmax=None)
                mesh = surface.plot_surface(points=False)
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_inverse.png', size=(800, 800))
                mlab.clf()





















