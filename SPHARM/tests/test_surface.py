import unittest

import numpy as np
import shutil
import os
from ddt import ddt, data
import warnings
import site

from SPHARM.classes.surface import Surface
from SPHARM.classes.image_stack import ImageStack
import SPHARM.lib.transformation as tr


@ddt
class TestSurfaceClass(unittest.TestCase):

    @data(
        '0, 0, -20, 0.216057\n-8, -2, -18, 0.216057\n'
        '-8, 0, -18, 0.216057\n-8, 2, -18, 0.216057\n-6, -4, -18, 0.216057\n'
        '-6, -2, -18, 0.216057\n-6, 0, -18, 0.216057\n-6, 2, -18, 0.216057\n',
        '	X	Y	Z	Name\n'
        '0	0.0	0.0	-10.0	../Data/SyntheticCells/input_test/case1/cells1473770615/ContourCell0_0.216056749\n'
        '1	-4.0	-1.0	-9.0	../Data/SyntheticCells/input_test/case1/cells1473770615/ContourCell0_0.216056749\n'
        '2	-4.0	0.0	-9.0	../Data/SyntheticCells/input_test/case1/cells1473770615/ContourCell0_0.216056749\n'
        '3	-4.0	1.0	-9.0	../Data/SyntheticCells/input_test/case1/cells1473770615/ContourCell0_0.216056749'
    )
    def test_init_and_save(self, coords):
        os.makedirs('data/test_data')
        f = open('data/test_data/surface.txt', 'w')
        f.write(coords)
        f.close()
        surf = Surface(filename='data/test_data/surface.txt')
        for coord in ['x', 'y', 'z']:
            self.assertIsNotNone(surf.__dict__[coord])
        shutil.rmtree('data/test_data/')

    def test_read_surface_and_save_as_stack(self):
        img = ImageStack(filename='', load=False)
        img.data = np.zeros([10, 100, 100])
        img.data[2:7, 10:-10, 10:-10] = 1
        voxel_size = [4, 0.3824, 0.3824]
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=False)
        surf = Surface(filename='data/test_data/surfaces/_Cell00001.csv')
        for coord in ['x', 'y', 'z']:
            self.assertIsNotNone(surf.__dict__[coord])

        surf.save_as_stack(filename='data/test_data/stack.tif', voxel_size=0.5)
        self.assertEqual(os.path.exists('data/test_data/stack.tif'), True)
        self.assertEqual(os.path.exists('data/test_data/stack.txt'), True)
        shutil.rmtree('data/test_data/')

    def test_transforms(self):
        img = ImageStack(filename='', load=False)
        img.data = np.zeros([10, 100, 100])
        img.data[2:7, 10:-10, 10:-10] = 1
        voxel_size = [4, 0.3824, 0.3824]
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=False)
        surf = Surface(filename='data/test_data/surfaces/_Cell00001.csv')
        surf.to_spherical()
        x, y, z = tr.spherical_to_cart(surf.R, surf.theta, surf.phi)
        self.assertAlmostEqual(np.sum(np.abs(surf.x - x)), 0, 7)
        self.assertAlmostEqual(np.sum(np.abs(surf.y - y)), 0, 7)
        self.assertAlmostEqual(np.sum(np.abs(surf.z - z)), 0, 7)
        shutil.rmtree('data/test_data/')

    @data(
        10,
        13,
        25
    )
    def test_interpolate(self, grid_size):
        img = ImageStack(filename='', load=False)
        img.data = np.zeros([10, 100, 100])
        img.data[2:7, 10:-10, 10:-10] = 1
        voxel_size = [4, 0.3824, 0.3824]
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=False)
        surf = Surface(filename='data/test_data/surfaces/_Cell00001.csv')
        surf.centrate()
        surf.to_spherical()
        grid = surf.interpolate(grid_size=grid_size)
        self.assertEqual(len(grid), grid_size)
        shutil.rmtree('data/test_data/')

    def test_spharm_transform(self):
        img = ImageStack(filename='', load=False)
        img.data = np.zeros([100, 100, 100])
        img.data[48:52, 48:52, 48:52] = 1.
        voxel_size = 0.3
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=True)
        surf = Surface(filename='data/test_data/surfaces/_Cell00001.csv')
        surf.centrate()
        surf.to_spherical()
        grid = surf.interpolate(grid_size=10)
        surf.compute_spharm(grid_size=10)
        ngrid = surf.inverse_spharm()
        self.assertAlmostEqual(np.mean(np.abs(ngrid - grid)), 0, 1)
        shutil.rmtree('data/test_data/')

    def test_spharm_transform_norm(self):
        img = ImageStack(filename='', load=False)
        img.data = np.zeros([100, 100, 100])
        img.data[48:52, 48:52, 48:52] = 1.
        voxel_size = 0.3
        img.extract_surfaces('data/test_data/surfaces/',
                             voxel_size=voxel_size, reconstruct=True)
        surf = Surface(filename='data/test_data/surfaces/_Cell00001.csv')
        surf.centrate()
        surf.to_spherical()
        grid = surf.interpolate(grid_size=10)
        surf.compute_spharm(grid_size=10, normalize=True, normalization_method='mean-radius')
        ngrid = surf.inverse_spharm()
        grid = grid / np.mean(grid)
        self.assertAlmostEqual(np.mean(np.abs(ngrid - grid)), 0, 1)
        shutil.rmtree('data/test_data/')

    def test05_plot(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        fn = path + 'data/synthetic_cell.txt'
        surf = Surface(filename=fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = surf.plot_points()
            os.makedirs('data/test_data')
            mesh.save('data/test_data/points_3D.png', size=(200, 200))

        surf.centrate()
        surf.to_spherical()
        surf.Rgrid = surf.interpolate(grid_size=100)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = surf.plot_surface(points=False)
            mesh.save('data/test_data/surface_3D.png', size=(200, 200))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = surf.plot_surface(points=True)
            mesh.save('data/test_data/surface_with_points_3D.png', size=(200, 200))
        self.assertEqual(os.path.exists('data/test_data/surface_3D.png'), True)
        self.assertEqual(os.path.exists('data/test_data/points_3D.png'), True)
        self.assertEqual(os.path.exists('data/test_data/surface_with_points_3D.png'), True)
        shutil.rmtree('data/test_data/')

    def test07_spharm_inverse_less(self):
        path = site.getsitepackages()[0] + '/SPHARM/tests/'
        fn = path + 'data/synthetic_cell.txt'
        surf = Surface(filename=fn)
        surf.centrate()
        surf.to_spherical()
        surf.compute_spharm(grid_size=100)
        surf.inverse_spharm()
        os.makedirs('data/test_data')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surf.plot_surface(points=True).save('data/test_data/surface_inverse_all.png', size=(200, 200))
            surf.inverse_spharm(lmax=30)
            surf.plot_surface(points=True).save('data/test_data/surface_inverse_30.png', size=(200, 200))
            surf.inverse_spharm(lmax=10)
            surf.plot_surface(points=True).save('data/test_data/surface_inverse_10.png', size=(200, 200))
        self.assertEqual(os.path.exists('data/test_data/surface_inverse_all.png'), True)
        self.assertEqual(os.path.exists('data/test_data/surface_inverse_30.png'), True)
        self.assertEqual(os.path.exists('data/test_data/surface_inverse_10.png'), True)
        shutil.rmtree('data/test_data/')


if __name__ == '__main__':
    unittest.main()













