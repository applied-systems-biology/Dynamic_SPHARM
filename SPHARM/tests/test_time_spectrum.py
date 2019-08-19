import unittest

import os
import shutil
import numpy as np
from ddt import ddt

from SPHARM.classes.spectrum import Spectrum
from SPHARM.classes.time_spectrum import TimeSpectrum


@ddt
class TestTimeSpectrumClass(unittest.TestCase):

    def test_add_spectrum_and_plotting(self):
        tsp = TimeSpectrum()

        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        sp.convert_to_csv()
        for i in range(3):
            tsp.add_spectrum(sp, timepoint=i*20)
        sp = Spectrum()
        surf = np.ones([10, 10])
        surf[3:4, 4:8] = 10
        sp.from_surface(surface=surf)
        sp.convert_to_csv()
        for i in range(2):
            tsp.add_spectrum(sp, timepoint=i*20+60)

        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        sp.convert_to_csv()
        for i in range(3):
            tsp.add_spectrum(sp, timepoint=i*20+100)

        self.assertEqual(len(tsp.spectra), 8)
        tsp.save_to_csv('data/test_data/time_spectrum.csv')
        self.assertEqual(os.path.exists('data/test_data/time_spectrum.csv'), True)
        pl = tsp.time_heatmap()
        pl.savefig('data/test_data/time_heatmap.png')
        self.assertEqual(os.path.exists('data/test_data/time_heatmap.png'), True)

        tsp.compute_derivative()
        pl = tsp.derivative_heatmap()
        pl.savefig('data/test_data/derivative_heatmap.png')
        self.assertEqual(os.path.exists('data/test_data/derivative_heatmap.png'), True)

        pl = tsp.plot_mean_abs_derivative()
        pl.savefig('data/test_data/mean_abs_derivative.png')
        self.assertEqual(os.path.exists('data/test_data/mean_abs_derivative.png'), True)
        shutil.rmtree('data/test_data/')

    def test_fourier(self):
        sp = Spectrum()
        sp.from_surface(surface=np.ones([10, 10]))
        sp.convert_to_csv()
        tsp = TimeSpectrum()
        for i in range(10):
            tsp.add_spectrum(sp)
        tsp.fourier_analysis()
        tsp.save_frequencies_to_csv('data/test_data/time_spectrum_freq.csv')
        self.assertEqual(os.path.exists('data/test_data/time_spectrum_freq.csv'), True)
        pl = tsp.frequency_heatmap()
        pl.savefig('data/test_data/frequency_heatmap.png')
        self.assertEqual(os.path.exists('data/test_data/frequency_heatmap.png'), True)
        shutil.rmtree('data/test_data/')

    def test_feature_vector(self):
        surf = np.ones([10, 10])
        sp = Spectrum()
        sp.from_surface(surface=surf)
        sp.convert_to_csv()

        surf[3:4, 4:8] = 10
        sp2 = Spectrum()
        sp2.from_surface(surface=surf)
        sp2.convert_to_csv()

        tsp = TimeSpectrum()
        tsp.add_spectrum(sp)
        tsp.add_spectrum(sp2)
        tsp.add_spectrum(sp)
        tsp.compute_derivative()
        self.assertEqual(len(tsp.return_feature_vector(cutoff=2)), 3)


if __name__ == '__main__':
    unittest.main()

