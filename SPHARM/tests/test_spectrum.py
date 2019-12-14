import unittest

import os
import numpy as np
import shutil
from ddt import ddt, data

from SPHARM.classes.spectrum import Spectrum


@ddt
class TestSpectrumClass(unittest.TestCase):

    @data(
        np.ones([10, 10]),
        np.ones([10, 20])
    )
    def test_from_surface(self, case):
        sp = Spectrum()
        harm = sp.from_surface(surface=case)
        self.assertEqual(tuple(harm.shape), (2, case.shape[0]/2, case.shape[0]/2))

    @data(
        np.ones([11, 11]),
        np.ones([11, 13]),
        np.ones([10, 12])
    )
    def test_from_surface_errors(self, case):
        sp = Spectrum()
        self.assertRaises(ValueError, sp.from_surface, surface=case)

    @data(
        np.ones([10, 10]),
        np.ones([10, 20])
    )
    def test_from_surface_norm(self, case):
        sp = Spectrum()
        harm = sp.from_surface(surface=case, normalize=True, normalization_method='zero-component')
        self.assertEqual(np.max(harm), 1)

    @data(
        np.ones([10, 10]),
        np.ones([10, 20])
    )
    def test_from_surface_norm2(self, case):
        sp = Spectrum()
        harm = sp.from_surface(surface=case, normalize=True, normalization_method='mean-radius')
        self.assertAlmostEqual(abs(np.max(harm)), 1, 8)

    def test_convertion(self):
        sp = Spectrum()
        harm_shtools = sp.from_surface(surface=np.ones([10, 10]))
        sp.convert_to_csv()
        harm_shtools2 = sp.convert_to_shtools_array()
        self.assertEqual(np.sum(abs(harm_shtools - harm_shtools2)), 0)

    def test_saving(self):
        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        sp.convert_to_csv()
        sp.save_to_csv(filename='data/test_data/spectrum.csv')
        sp2 = Spectrum(filename='data/test_data/spectrum.csv')
        self.assertAlmostEqual(np.sum(abs(sp.harmonics_shtools - sp2.harmonics_shtools)), 0, 15)
        shutil.rmtree('data/test_data/')

    def test_frequency_spectrum(self):
        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        sp.compute_frequency_spectrum()
        self.assertEqual(len(sp.frequency_spectrum), 5)

    def test_heatmap(self):
        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        pl = sp.heatmap()
        os.makedirs('data/test_data')
        pl.savefig('data/test_data/heatmap.png')
        self.assertEqual(os.path.exists('data/test_data/heatmap.png'), True)
        shutil.rmtree('data/test_data/')

    def test_frequency_plot(self):
        sp = Spectrum(name='Example 1')
        sp.from_surface(surface=np.ones([10, 10]))
        pl = sp.frequency_plot()
        os.makedirs('data/test_data')
        pl.savefig('data/test_data/frequency_plot.png')
        self.assertEqual(os.path.exists('data/test_data/frequency_plot.png'), True)
        shutil.rmtree('data/test_data/')

    def test_inverse_transform(self):
        sp = Spectrum()
        surf = np.ones([10, 10])
        sp.from_surface(surface=surf)
        grid = sp.spharm_to_surface()
        self.assertAlmostEqual(np.sum(np.abs(surf - grid)), 0, 7)

    def test_feature_vector(self):
        sp = Spectrum()
        surf = np.ones([10, 10])
        sp.from_surface(surface=surf)
        self.assertEqual(len(sp.return_feature_vector(cutoff=3)), 4)


if __name__ == '__main__':
    unittest.main()

