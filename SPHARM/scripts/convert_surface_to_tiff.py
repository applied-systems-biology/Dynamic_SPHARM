from __future__ import division

import os
import numpy as np
import pandas as pd
from skimage import io
import warnings

from multiprocessing import Process
import time
import sys


def _print_progress(procdone, totproc, start):
    donepercent = procdone*100/totproc
    elapse = time.time() - start
    tottime = totproc*1.*elapse/procdone
    left = tottime - elapse
    units = 'sec'
    if left > 60:
        left = left/60.
        units = 'min'
        if left > 60:
            left = left/60.
            units = 'hours'

    print('done', procdone, 'of', totproc, '(', donepercent, '% ), approx. time left: ', left, units)


def run_parallel(process, process_name=None, print_progress=True, **kwargs):

    items = kwargs.pop('items', [])
    max_threads = int(round(kwargs.pop('max_threads', 8)))
    if process_name is None:
        process_name = process.func_name

    if print_progress:
        print('Run', process_name)

    procs = []

    totproc = len(items)
    procdone = 0
    start = time.time()

    if print_progress:
        print('Started at ', time.ctime())

    for i, cur_item in enumerate(items):

        while len(procs) >= max_threads:
            time.sleep(0.05)
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)
                    procdone +=1
                    if print_progress:
                        _print_progress(procdone, totproc, start)

        cur_args = kwargs.copy()
        cur_args['item'] = cur_item
        p = Process(target=process, kwargs=cur_args)
        p.start()
        procs.append(p)

    while len(procs) > 0:
        time.sleep(0.05)
        for p in procs:
            if not p.is_alive():
                procs.remove(p)
                procdone += 1
                if print_progress:
                 _print_progress(procdone, totproc, start)

    if print_progress:
        print(process_name, 'done')


def as_stack(x, y, z, minmax=None):
    if minmax is None:
        minmax = np.int_([[z.min(), z.max()],
                          [y.min(), y.max()],
                          [x.min(), x.max()]])
    else:
        minmax = np.int_(np.round_(minmax))

    x = np.int_(x) - minmax[2, 0] + 1
    y = np.int_(y) - minmax[1, 0] + 1
    z = np.int_(z) - minmax[0, 0] + 1

    img = np.zeros([minmax[0, 1] - minmax[0, 0] + 3, minmax[1, 1] - minmax[1, 0] + 3, minmax[2, 1] - minmax[2, 0] + 3])
    img[z, y, x] = 255

    return img


def make_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                pass


_IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG', 'BMP', 'tif', 'TIFF', 'tiff', 'TIF']


def _is_in_extensions(filename, extensions):

    parts = filename.split('.')
    if len(parts) > 1:
        ext = parts[-1]
    else:
        ext = ''

    if ext in extensions or '*' in extensions:
        return True
    else:
        return False


def list_subfolders(inputfolder, cur_subfolder=None, subfolders=None, extensions=None):
    if extensions is None:
        # global IMAGE_EXTENSIONS
        extensions = _IMAGE_EXTENSIONS

    if cur_subfolder is None:
        cur_subfolder = ''

    if subfolders is None:
        subfolders = []

    files = os.listdir(inputfolder + cur_subfolder)

    for path in files:
        if os.path.isdir(inputfolder + cur_subfolder + path):
            if path[-1] != '/':
                path = path + '/'
            subfolders = list_subfolders(inputfolder,  cur_subfolder=cur_subfolder + path, subfolders=subfolders,
                                         extensions=extensions)
        else:
            if _is_in_extensions(path, extensions):
                subfolders.append(cur_subfolder + path)

    return subfolders


###################################################


class Surface(object):

    def __init__(self, filename=None, **kwargs):

        self.x = None  # list of points
        self.y = None  # list of points
        self.z = None  # list of points

        self.filename = filename

        if filename is not None:
            self.read_from_file(filename, **kwargs)

    def read_from_file(self, filename, voxel_size=1):
        if os.path.exists(filename):
            stat = pd.read_csv(filename, sep='\t', index_col=0)

            if 'X' in stat.columns and 'Y' in stat.columns and 'Z' in stat.columns:
                self.x = np.array(stat.X)
                self.y = np.array(stat.Y)
                self.z = np.array(stat.Z)

            else:
                stat = pd.read_csv(filename, sep=',', header=None)
                px = voxel_size
                if 0 in stat.columns and 1 in stat.columns and 2 in stat.columns:
                    self.x = np.array(stat[0]) * px
                    self.y = np.array(stat[1]) * px
                    self.z = np.array(stat[2]) * px

    def save_as_stack(self, filename, voxel_size):
        voxel_size = np.array([voxel_size]).flatten()
        if len(voxel_size) == 1:
            voxel_size = np.ones(3) * voxel_size
        make_folders([os.path.dirname(filename)])
        img = self.as_stack(voxel_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filename, img.astype(np.uint8))
        metadata = pd.Series({'voxel_size_xy': voxel_size[2], 'voxel_size_z': voxel_size[0]})
        metadata.to_csv(filename[:-4] + '.txt', sep='\t')

    def as_stack(self, voxel_size, minmax=None):
        if minmax is not None:
            minmax = [minmax[0] / voxel_size[0],
                      minmax[1] / voxel_size[1],
                      minmax[2] / voxel_size[2]]
        img = as_stack(np.array(self.x) / voxel_size[2],
                          np.array(self.y) / voxel_size[1],
                          np.array(self.z) / voxel_size[0], minmax=minmax)
        return img


###################################################

def convert_to_tiff(**kwargs):
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../stacks/')
    filename = kwargs.get('item')

    surface = Surface(filename=inputfolder + filename)
    surface.save_as_stack(outputfolder + filename + '.tif', voxel_size=kwargs.get('voxel_size'))


def convert_parallel(**kwargs):
    files = list_subfolders(kwargs.get('inputfolder'), extensions=['*'])

    if kwargs.get('debug'):
        kwargs['item'] = files[0]
        kwargs.get('process')(**kwargs)
    else:
        kwargs['items'] = files
        run_parallel(**kwargs)

if __name__ == '__main__':

    args = sys.argv[1:]
    if args[0] != 'test':
        if len(args) == 3:
            convert_parallel(process=convert_to_tiff, inputfolder=args[0],
                             outputfolder=args[1], max_threads=6, voxel_size=float(args[2]))

