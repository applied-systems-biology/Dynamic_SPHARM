import unittest

import os
import shutil
import numpy as np
from ddt import ddt

from SPHARM.classes.surface import Surface
from SPHARM.classes.moving_surface import MovingSurface


@ddt
class TestMovingSurfaceCass(unittest.TestCase):

    def test_add_spectrum_and_plotting(self):
        ms = MovingSurface()

        surf = Surface(grid=np.ones([10, 10]))
        for i in range(3):
            ms.add_surface(surf, timepoint=i*20)

        surf = Surface(grid=np.ones([10, 10]))
        surf.Rgrid[3:4, 4:8] = 10
        for i in range(2):
            ms.add_surface(surf, timepoint=i*20+60)

        surf = Surface(grid=np.ones([10, 10]))
        for i in range(3):
            ms.add_surface(surf, timepoint=i*20+100)

        self.assertEqual(len(ms.timespectrum.spectra), 0)
        ms.compute_timespectrum(gridsize=10)
        self.assertEqual(len(ms.timespectrum.spectra), 8)

        ms.plot_surfaces('data/test_data/surfaces/')
        files = os.listdir('data/test_data/surfaces/')
        self.assertEqual(len(files), 8)

        ms.plot_max_projections(outputfolder='data/test_data/maxprojections/', voxel_size=0.1)
        files = os.listdir('data/test_data/maxprojections/')
        self.assertEqual(len(files), 24)

        shutil.rmtree('data/test_data/')


if __name__ == '__main__':
    unittest.main()

