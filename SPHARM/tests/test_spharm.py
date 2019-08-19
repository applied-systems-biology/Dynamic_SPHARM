import unittest

import os
import pandas as pd
from ddt import ddt
import shutil
import site

from SPHARM.lib import spharm
import SPHARM.lib.parallel as prl


@ddt
class TestSPHARM(unittest.TestCase):

    def test_spharm_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        prl.run_parallel(process=spharm.compute_spharm, inputfolder=path + 'data/surfaces/',
                         outputfolder='data/test_data/spharm_test/spharm/',
                         grid_size=10, extensions=['csv'], print_progress=False)
        files = os.listdir('data/test_data/spharm_test/spharm/')
        self.assertEqual(len(files), 9)
        data = pd.read_csv('data/test_data/spharm_test/spharm.csv', sep='\t')
        self.assertEqual('Time' in data.columns, True)
        self.assertEqual('TrackID' in data.columns, True)
        shutil.rmtree('data/test_data/')

    def test_spharm_batch2(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        prl.run_parallel(process=spharm.compute_spharm, inputfolder=path + 'data/synthetic_cells/',
                         outputfolder='data/test_data/spharm_test/spharm/', extensions=['*'],
                         grid_size=10, print_progress=False)
        files = os.listdir('data/test_data/spharm_test/spharm/')
        self.assertEqual(len(files), 6)
        data = pd.read_csv('data/test_data/spharm_test/spharm.csv', sep='\t')
        self.assertEqual('Time' in data.columns, True)
        self.assertEqual('TrackID' in data.columns, True)
        shutil.rmtree('data/test_data/')

    def test_convert_surfaces_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        prl.run_parallel(process=spharm.convert_surfaces, inputfolder=path + 'data/synthetic_cells/',
                         outputfolder='data/test_data/synthetic_cells_surfaces/',
                         extensions=['*'], print_progress=False)
        files = os.listdir('data/test_data/synthetic_cells_surfaces/')
        self.assertEqual(len(files), 6)
        shutil.rmtree('data/test_data/')

    def test_convert_to_tiff_batch(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        prl.run_parallel(process=spharm.convert_to_tiff, inputfolder=path + 'data/synthetic_cells/',
                         outputfolder='data/test_data/synthetic_cells_tiff/',
                         extensions=['*'], voxel_size=0.4, print_progress=False, combine=False)
        files = os.listdir('data/test_data/synthetic_cells_tiff/')
        self.assertEqual(len(files), 12)
        shutil.rmtree('data/test_data/')


if __name__ == '__main__':
    unittest.main()













