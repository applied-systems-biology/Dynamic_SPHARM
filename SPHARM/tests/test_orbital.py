import unittest

import numpy as np
from ddt import ddt, data

from SPHARM.classes.orbital import Orbital


@ddt
class TestOrbitalClass(unittest.TestCase):

    @data(
        ([50, 50], [0], [0], [5]),
        ([50, 50], [2], [0], [5]),
        ([50, 50], [2], [-1], [5]),
        ([50, 50], [2], [2], [5]),
    )
    def test_generate(self, case):
        grid_shape, m, n, amplitude = case
        orbital = Orbital(grid_shape=grid_shape, m=m, n=n, amplitude=1)
        orbital.Rgrid = orbital.Rgrid
        orbital.compute_spharm()
        data = orbital.spharm.harmonics_csv
        data['amplitude'] = np.array(data['amplitude'])/data['amplitude'].max()
        for i in range(len(data)):
            cur_m = data.iloc[i]['degree']
            cur_n = data.iloc[i]['order']
            if cur_m == m[0] and cur_n == n[0]:
                self.assertAlmostEqual((data.iloc[i]['amplitude']), 1, 10)
            else:
                self.assertAlmostEqual((data.iloc[i]['amplitude']), 0, 10)


if __name__ == '__main__':
    unittest.main()













