import unittest

import os
import numpy as np
import shutil
from ddt import ddt, data

from SPHARM.classes.ellipsoid import Ellipsoid


@ddt
class TestEllipsoidClass(unittest.TestCase):

    @data(
        ([10, 10], [1, 3, 2], (0, 0)),
        ([10, 30], [3, 3, 2], (np.pi/4, 0)),
        ([100, 100], [1, 3, 2], (np.pi/6, np.pi/4))
    )
    def test_generate(self, case):
        grid_shape, size, rotation = case
        ellipsoid = Ellipsoid(grid_shape=grid_shape, size=size, rotation=rotation)
        ellipsoid.profile_xy().save(outputfile='data/test_data/ellipsoid_test/' + ellipsoid.name + '_xy.png')
        ellipsoid.profile_xz().save(outputfile='data/test_data/ellipsoid_test/' + ellipsoid.name + '_xz.png')
        ellipsoid.profile_yz().save(outputfile='data/test_data/ellipsoid_test/' + ellipsoid.name + '_yz.png')
        self.assertEqual(os.path.exists('data/test_data/ellipsoid_test/' + ellipsoid.name + '_xy.png'), True)
        self.assertEqual(os.path.exists('data/test_data/ellipsoid_test/' + ellipsoid.name + '_yz.png'), True)
        self.assertEqual(os.path.exists('data/test_data/ellipsoid_test/' + ellipsoid.name + '_yz.png'), True)
        shutil.rmtree('data/test_data/')

    def test_save_as_stack(self):
        ellipsoid = Ellipsoid(grid_shape=[100, 100], size=[3, 3, 2], rotation=(np.pi/6, np.pi/4))
        ellipsoid.save_as_stack(filename='data/test_data/ellipsoid_test/' + ellipsoid.name + '.tif', voxel_size=0.5)
        self.assertEqual(os.path.exists('data/test_data/ellipsoid_test/' + ellipsoid.name + '.tif'), True)
        self.assertEqual(os.path.exists('data/test_data/ellipsoid_test/' + ellipsoid.name + '.txt'), True)
        shutil.rmtree('data/test_data/')


if __name__ == '__main__':
    unittest.main()













