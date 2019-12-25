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
                stack.extract_surfaces(path + 'surfaces/' + gr + '/', voxel_size=0.3)

        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            files = os.listdir(path + 'surfaces/' + gr + '/')
            files.sort()
            for fn in files:
                print(fn)
                surface = Surface(filename=path + 'surfaces/' + gr + '/' + fn, voxel_size=0.3)
                mesh = mlab.points3d(surface.x, surface.y, surface.z, surface.x, scale_mode='none',
                                     scale_factor=0.5, mode='sphere', colormap='jet').scene
                mesh.background = (1, 1, 1)
                mesh.magnification = 10
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_init.png', size=(100, 100))
                mesh = mlab.points3d(surface.x, surface.y, surface.z, surface.x, scale_mode='none',
                                     scale_factor=0.5, mode='sphere', colormap='gray').scene
                mesh.background = (1, 1, 1)
                mesh.magnification = 10

        mlab.clf()
        groups = os.listdir(path + 'surfaces/')
        for gr in groups:
            print(gr)
            files = os.listdir(path + 'surfaces/' + gr + '/')
            files.sort()
            for fn in files:
                print(fn)
                surface = Surface(filename=path + 'surfaces/' + gr + '/' + fn, voxel_size=0.3)
                mesh = mlab.points3d(surface.x, surface.y, surface.z, surface.z, scale_mode='none',
                                     scale_factor=0.5, mode='sphere', colormap='jet',
                                     extent=[-100, 100, -100, 100, -100, 100]).scene
                mesh.background = (1, 1, 1)
                mesh.magnification = 10
                filelib.make_folders([os.path.dirname(path + 'surface_plots/' + gr + '_' + fn[:-4])])
                mesh.save(path + 'surface_plots/' + gr + '_' + fn[:-4] + '_init.png', size=(100, 100))
                mlab.clf()


