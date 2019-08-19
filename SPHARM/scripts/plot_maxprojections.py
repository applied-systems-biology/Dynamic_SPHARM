from __future__ import division

import mkl
mkl.set_num_threads(1)

import sys
import os
import numpy as np

from skimage import io
from helper_lib import filelib
from skimage.exposure import rescale_intensity


#################################
args = sys.argv[1:]
if len(args) > 0:
    path = args[0]
    if path != 'test':
        if not path.endswith('/'):
            path += '/'

        kwargs = {'max_threads': 50, 'voxel_size': 1}
        groups = os.listdir(path + 'raw/')
        for gr in groups:
            samples = os.listdir(path + 'raw/' + gr)
            for sample in samples:
                files = os.listdir(path + 'raw/' + gr + '/' + sample)
                print(path + 'raw/' + gr + '/' + sample + '/' + files[0])
                img = io.imread(path + 'raw/' + gr + '/' + sample + '/' + files[0])
                ind = -1
                for j in range(len(img[0])):
                    if np.max(img[0, j]) > 0:
                        ind = j
                img = img[:, ind]
                per = np.percentile(img, 99.95)
                for i in range(len(img)):
                    img[i][np.where(img[i] > per)] = per
                    maxproj = np.max(img[i], axis=0)
                    filename = path + 'raw_maxprojecion_xy/' + gr + '/' + sample + '/' + files[0][:-4] + '%03d.tif' % i
                    filelib.make_folders([os.path.dirname(filename)])
                    if maxproj.max() > 255:
                        maxproj = rescale_intensity(maxproj, out_range=(0, 255))
                    io.imsave(filename, maxproj.astype(np.uint8))











