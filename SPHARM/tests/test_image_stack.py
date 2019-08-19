import unittest

import os
import numpy as np
import shutil
from ddt import ddt, data

from SPHARM.classes.image_stack import ImageStack


@ddt
class TestImageStackClass(unittest.TestCase):

    @data(
        ('Doc17_19-42-58_PMT - PMT [PMT2_ 525-50] _C1_Time Time0.tif', 0, 1),
        ('data/Doc17_19-42-58_PMT - PMT [PMT3_ 593-40] _C2_Time Time18.tif', 18, 2)
    )
    def test_parse_filename(self, case):
        fn, tp, ch = case
        path = 'data/test_data//SPHARM/tests/'
        img = ImageStack(filename=path + fn, load=False)
        self.assertEqual(img.timepoint, tp)
        self.assertEqual(img.channel, ch)

    def test_load_and_save(self):
        fn = 'data/test_data/test_stack.tif'
        img = ImageStack(filename=fn, load=False)
        img.data = np.ones([10, 100, 100])
        img.save(filename=fn)
        self.assertEqual(os.path.exists(fn), True)

        img = ImageStack(filename=fn)
        self.assertIsInstance(img.data, np.ndarray)
        shutil.rmtree('data/test_data/')

    def test_save_maxproj(self):
        fn = 'data/test_data/test_stack.tif'
        img = ImageStack(filename=fn, load=False)
        img.data = np.zeros([10, 100, 100])
        img.data[2:7, 10:-10, 10:-10] = 50
        img.save_max_projection(filename=fn)
        self.assertEqual(os.path.exists(fn), True)
        shutil.rmtree('data/test_data/')

    def test_extract_surfaces(self):
        fn = 'data/test_data/test_stack.tif'
        img = ImageStack(filename=fn, load=False)
        img.data = np.zeros([10, 100, 100])
        img.data[2:7, 10:-10, 10:-10] = 50
        voxel_size = [4, 0.3824, 0.3824]
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=False)
        img.extract_surfaces('data/test_data/surfaces_reconstructed/',
                             voxel_size=voxel_size, reconstruct=True)
        files = os.listdir('data/test_data/surfaces_reconstructed/')
        self.assertEqual(len(files), 1)
        files = os.listdir('data/test_data/surfaces/')
        self.assertEqual(len(files), 1)
        shutil.rmtree('data/test_data/')


if __name__ == '__main__':
    unittest.main()


