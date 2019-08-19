import unittest

import numpy as np
from ddt import ddt, data
import SPHARM.lib.transformation as tr


@ddt
class TestTransformation(unittest.TestCase):

    @data(
        ([2, 908, -234, 0.3], [654, -6, -64, 2], [-32, 243, 24, -22]),
    )
    def test_spherical_transform(self, coords):
        x, y, z = coords
        r, theta, phi = tr.cart_to_spherical(x, y, z)
        x1, y1, z1 = tr.spherical_to_cart(r, theta, phi)
        self.assertAlmostEqual(sum(abs(x - x1)), 0, 10)
        self.assertAlmostEqual(sum(abs(y - y1)), 0, 10)
        self.assertAlmostEqual(sum(abs(z - z1)), 0, 10)

    @data(

        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 3)
    )
    def test_rotation_single(self, coords):
        x, y, z = coords
        r, theta, phi = tr.cart_to_spherical(x, y, z)
        x, y, z = tr.rotate_spherical(x, y, z, -theta, -(phi - np.pi))
        self.assertAlmostEqual(x, 0, 10)
        self.assertAlmostEqual(y, 0, 10)
        self.assertGreater(z, 0)

    @data(

        (0, np.pi/2, [[0, 0, -2, 1.9, 0, 0],
                      [1, -0.9, 0, 0, 0, 0],
                      [0, 0, 0, 0, 3, -2.9]]),
    )
    def test_rotation_complex(self, case):
        theta, phi, coord = case
        x = [1, -0.9, 0, 0, 0, 0]
        y = [0, 0, 2, -1.9, 0, 0]
        z = [0, 0, 0, 0, 3, -2.9]
        x0, y0, z0 = coord
        x, y, z = tr.rotate_spherical(x, y, z, theta, phi)
        self.assertAlmostEqual(sum(abs(x - np.array(x0))), 0, 10)
        self.assertAlmostEqual(sum(abs(y - np.array(y0))), 0, 10)
        self.assertAlmostEqual(sum(abs(z - np.array(z0))), 0, 10)


if __name__ == '__main__':
    unittest.main()
