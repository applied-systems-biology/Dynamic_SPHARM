from __future__ import division

import numpy as np
from scipy.special import sph_harm
import SPHARM.lib.transformation as tr

from SPHARM.classes.surface import Surface


class Orbital(Surface):
    """
    Class for a surface of generated from spherical harmonics.
    """
    def __init__(self, grid_shape, m, n, amplitude):
        """
        Initialize a surface from given spherical harmonics.
        
        Parameters
        ----------
        grid_shape : tuple of size 2
            Shape of the 2D grid (phi-theta grid).
        m : scalar or sequence of scalars
            Degree(s) of spherical harmonics to generate. 
            Must be (a) non-negative number(s).
        n : scalar or sequence of scalars
            Order(s) of spherical harmonics to generate.
            Must be of the same length as m.
            Must be in the range [-m; m] of the corresponding degree.        
        amplitude : scalar or sequence of scalars
            Relative amplitude(s) of corresponding harmonics.
            Must be of the same length as m.
        """
        super(Orbital, self).__init__()
        theta = np.linspace(0, np.pi, grid_shape[0], endpoint=False)  # polar angle
        phi = np.linspace(0, 2 * np.pi, grid_shape[1], endpoint=False)  # azimuthal angle
        self.Phi, self.Theta = np.meshgrid(phi, theta)
        self.Rgrid = np.zeros_like(self.Theta)

        self.generate(np.array([m]).flatten(), np.array([n]).flatten(), np.array([amplitude]).flatten())
        self.phi = self.Phi.flatten()
        self.theta = self.Theta.flatten()
        self.R = self.Rgrid.flatten().real
        self.x, self.y, self.z = tr.spherical_to_cart(self.R, self.theta, self.phi)

    def generate(self, m, n, amplitude):
        """
        Generate a surface from given spherical harmonic degree(s), order(s) and amplitude(s).
        
        Parameters
        ----------
        m : scalar or sequence of scalars
            Degree(s) of spherical harmonics to generate. 
            Must be (a) non-negative number(s).
        n : scalar or sequence of scalars
            Order(s) of spherical harmonics to generate.
            Must be of the same length as m.
            Must be in the range [-m; m] of the corresponding degree.        
        amplitude : scalar or sequence of scalars
            Relative amplitude(s) of corresponding harmonics.
            Must be of the same length as m.
        """
        for i in range(len(m)):
            self.Rgrid = self.Rgrid + sph_harm(n[i], m[i], self.Phi, self.Theta)*amplitude[i]


