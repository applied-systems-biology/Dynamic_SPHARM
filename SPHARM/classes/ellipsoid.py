from __future__ import division

import numpy as np

from SPHARM.classes.surface import Surface
from SPHARM.classes.profile import Profile
import SPHARM.lib.transformation as tr


class Ellipsoid(Surface):
    """
    Class for a surface of a ellipsoidal object.
    """
    def __init__(self, grid_shape, size, rotation=None):
        """
        Initialize a surface of ellipsoid with given size and rotation.
        
        Parameters
        ----------
        grid_shape : tuple of size 2
            Shape of the 2D grid of the ellipsoid surface (phi-theta grid).
        size : tuple of size 3
            Relative size of ellipsoid axes.
        rotation : tuple of size 2
            Polar and azimuthal angles of ellipsoid rotation in radians.
        """
        super(Ellipsoid, self).__init__()
        theta = np.linspace(0, np.pi, grid_shape[0], endpoint=False)  # polar angle
        phi = np.linspace(0, 2 * np.pi, grid_shape[1], endpoint=False)  # azimuthal angle
        self.Phi, self.Theta = np.meshgrid(phi, theta)
        self.Rgrid = None
        self.name = None

        self.generate(size, rotation=rotation)

        self.phi = self.Phi.flatten()
        self.theta = self.Theta.flatten()
        self.R = self.Rgrid.flatten()
        self.x, self.y, self.z = tr.spherical_to_cart(self.R, self.theta, self.phi)

    def generate(self, size, rotation=None):
        """
        Generate ellipsoid surface with given size and rotation.
        
        Parameters
        ----------
        size : tuple of size 3
            Relative size of ellipsoid axes.
        rotation : tuple of size 2
            Polar and azimuthal angles of ellipsoid rotation in radians.
        """

        if rotation is not None:
            th, ph = rotation
        else:
            th, ph = (0, 0)

        x = np.sin(self.Theta) * np.cos(self.Phi) * np.cos(th) + np.cos(self.Theta) * np.sin(th)
        y = np.sin(self.Theta) * np.sin(self.Phi)
        z = np.cos(self.Theta) * np.cos(th) - np.sin(self.Theta) * np.cos(self.Phi) * np.sin(th)
        self.Rgrid = np.sqrt(1 / ((x / size[0]) ** 2 + (y / size[1]) ** 2 + (z / size[2]) ** 2))

        if ph > 0:
            phi = self.Phi[1]
            i = np.argmin(abs(phi - ph))
            R = np.zeros_like(self.Rgrid)
            R[:, :i] = self.Rgrid[:, -i:]
            R[:, i:] = self.Rgrid[:, :-i]

            self.Rgrid = R

        self.name = 'Ellipsoid; half-axes=' + str(size[0]) + '-' + str(size[1]) + '-' + str(size[2]) \
                    + '; grid size=' + str(self.Rgrid.shape[0]) + 'x' + str(self.Rgrid.shape[1]) \
                    + '; rotation $\\theta$=' + str(round(th, 2)) + ', $\phi$=' + str(round(ph, 2))

    def profile_xy(self, r=None):
        """
        Plot the surface profile along the xy plane cut through the middle of the ellipsoid.
        
        Parameters
        ----------
        r : numpy.ndarray, optional
            The ellipsoid surface to project.
            If None, self.Rgid will be used.
            Default is None.

        Returns
        -------
        Profile : the generated projection.
        """
        if r is None:
            r = self.Rgrid
        r = r[int(r.shape[0] / 2)]
        theta = self.Phi[int(self.Theta.shape[0] / 2)]
        profile = Profile(r, theta)
        return profile

    def profile_xz(self, r=None):
        """
        Plot the surface profile along the xz plane cut through the middle of the ellipsoid.
        
        Parameters
        ----------
        r : numpy.ndarray, optional
            The ellipsoid surface to project.
            If None, self.Rgid will be used.
            Default is None.

        Returns
        -------
        Profile : the generated projection.
        """
        if r is None:
            r = self.Rgrid
        r = np.concatenate((r[:, int(r.shape[1] / 2)][::-1], r[:, 0]))
        theta = np.concatenate((2 * np.pi - self.Theta[:, int(self.Theta.shape[1] / 2)][::-1], self.Theta[:, 0]))
        profile = Profile(r, theta)
        return profile

    def profile_yz(self, r=None):
        """
        Plot the surface profile along the yz plane cut through the middle of the ellipsoid.
        
        Parameters
        ----------
        r : numpy.ndarray, optional
            The ellipsoid surface to project.
            If None, self.Rgid will be used.
            Default is None.

        Returns
        -------
        Profile : the generated projection.
        """
        if r is None:
            r = self.Rgrid
        r = np.concatenate((r[:, int(round(r.shape[1] * 3 / 4))][::-1], r[:, int(round(r.shape[1] / 4))]))
        theta = np.concatenate((2 * np.pi - self.Theta[:, int(round(self.Theta.shape[1] * 3 / 4))][::-1],
                                self.Theta[:, int(round(self.Theta.shape[1] / 4))]))
        profile = Profile(r, theta)
        return profile







