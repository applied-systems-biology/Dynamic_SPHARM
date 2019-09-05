from __future__ import division

import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from skimage import io
from mayavi import mlab
import warnings

from helper_lib import filelib
import SPHARM.lib.transformation as tr

from SPHARM.classes.spectrum import Spectrum


class Surface(object):
    """
    Class for a surface grid of an object
    """

    def __init__(self, filename=None, grid=None, data=None, **kwargs):
        """
        Initialize a surface from file or grid.
        
        Parameters
        ----------
        filename : str, optional
            Path to a surface file to read the surface data.
            If None, an empty surface will be initialized.
            Default is None.
        grid : numpy.ndarray, dimension (n, n) or (n, 2*n), n is even, optional
            A 2D equally sampled (default) or equally spaced complex grid 
            that conforms to the sampling theorem of Driscoll and Healy (1994). 
            The first latitudinal band corresponds to 90 N, the latitudinal band for 90 S is not included, 
            and the latitudinal sampling interval is 180/n degrees. 
            The first longitudinal band is 0 E, the longitude band for 360 E is not included, 
            and the longitudinal sampling interval is 360/n for an equally 
            and 180/n for an equally spaced grid, respectively.
        data : pandas DataFrame, optional
            DataFrame with surface coordinates.
            If None, an empty surface will be initialized.
            Default is None.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the self.read_from_file function.
        """

        self.X = None  # grid
        self.Y = None  # grid
        self.Z = None  # grid
        self.Phi = None  # grid
        self.Theta = None  # grid
        self.Rgrid = grid  # grid

        self.x = None  # list of points
        self.y = None  # list of points
        self.z = None  # list of points
        self.R = None  # list of points
        self.phi = None  # list of points
        self.theta = None  # list of points

        self.filename = filename
        self.spharm = None
        self.metadata = pd.Series()
        self.center = None
        self.migration_angles = None

        if grid is None:
            if filename is not None:
                self.read_from_file(filename, voxel_size=kwargs.get('voxel_size', 1))
            elif data is not None:
                self.from_dataframe(data)

        if self.Rgrid is not None:
            self.R = self.Rgrid.flatten()
            self.Theta = np.linspace(0, np.pi, self.Rgrid.shape[0], endpoint=False)
            self.Phi = np.linspace(0, 2*np.pi, self.Rgrid.shape[0], endpoint=False)
            self.Phi, self.Theta = np.meshgrid(self.Phi, self.Theta)
            self.phi = self.Phi.flatten()
            self.theta = self.Theta.flatten()
            self.x, self.y, self.z = tr.spherical_to_cart(self.R, self.theta, self.phi)

    def read_from_file(self, filename, voxel_size=1):
        """
        Read surface coordinates from file.
        
        Parameters
        ----------
        filename : str, optional
            Path to a surface file to read the surface data.
        voxel_size : scalar or sequence of scalars, optional
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
            Default is 1.
        """
        if os.path.exists(filename):
            f = open(filename)
            st = f.readlines()
            f.close()
            if len(st) > 0:
                stat = pd.read_csv(filename, sep='\t', index_col=0)

                if 'X' in stat.columns and 'Y' in stat.columns and 'Z' in stat.columns:
                    self.x = np.array(stat.X)
                    self.y = np.array(stat.Y)
                    self.z = np.array(stat.Z)

                else:
                    stat = pd.read_csv(filename, sep=',', header=None)
                    px = voxel_size
                    if 0 in stat.columns and 1 in stat.columns and 2 in stat.columns:
                        self.x = np.array(stat[0])*px
                        self.y = np.array(stat[1])*px
                        self.z = np.array(stat[2])*px
                self.to_spherical()
                self.metadata['Name'] = filename
                p = re.compile('[-+]?\d*\.*\d+')
                if 'Time' in stat.columns:
                    self.metadata['Time'] = stat['Time'].iloc[0]
                elif len(filename.split('Time')) > 1:
                    self.metadata['Time'] = float(p.findall(filename.split('Time')[-1])[0])
                else:
                    num = p.findall(filename)
                    if len(num) > 0:
                        self.metadata['Time'] = float(num[-1])
                    else:
                        self.metadata['Time'] = 0

                if 'TrackID' in stat.columns:
                    self.metadata['TrackID'] = stat['TrackID'].iloc[0]
                elif len(filename.split('Cell')) > 1:
                    self.metadata['TrackID'] = float(p.findall(filename.split('cells')[-1])[0])
                else:
                    self.metadata['TrackID'] = 0

    def from_dataframe(self, stat):
        """
        Get surface coordinates from a pandas DataFrame.

        Parameters
        ----------
        stat : pandas DataFrame, optional
            DataFrame with surface coordinates.
        """
        self.x = np.array(stat.X)
        self.y = np.array(stat.Y)
        self.z = np.array(stat.Z)
        self.metadata['Time'] = stat['Time'].iloc[0]

    def save(self, filename):
        """
        Save the surface coordinates to a csv file.
        
        Parameters
        ----------
        filename : str
            Output file name.
        """
        if self.x is not None:
            filelib.make_folders([os.path.dirname(filename)])
            stat = pd.DataFrame({'X': self.x, 'Y': self.y, 'Z': self.z})
            stat['Name'] = self.filename

            stat.to_csv(filename, sep='\t')

    def save_as_stack(self, filename, voxel_size):
        """
        Save the surface as a 3D stack.
        
        Parameters
        ----------
        filename : str
            Output file name.
        voxel_size : scalar or sequence of scalars
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
        """
        if self.x is not None:
            voxel_size = np.array([voxel_size]).flatten()
            if len(voxel_size) == 1:
                voxel_size = np.ones(3)*voxel_size
            filelib.make_folders([os.path.dirname(filename)])
            img = self.as_stack(voxel_size)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename, img.astype(np.uint8))
            metadata = pd.Series({'voxel_size_xy': voxel_size[2], 'voxel_size_z': voxel_size[0]})
            metadata['min_x'] = self.x.min() - 1
            metadata['min_y'] = self.y.min() - 1
            metadata['min_z'] = self.z.min() - 1
            metadata.to_csv(filename[:-4] + '.txt', sep='\t', header=False)

    def as_stack(self, voxel_size, minmax=None):
        """
        Convert the surface to a 3D image stack.
        
        Parameters
        ----------
        voxel_size : scalar or sequence of scalars
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
        minmax : ndarray, optional
            Boundaries to crop the image stack of the form [[z_min, z_max], [y_min, y_max], [x_min, x_max]].
            If None, set to the minimal and maximal given coordinates.
            Default is None.

        Returns
        -------
        ndimage : 3D binary image with surface point as foreground.

        """
        if minmax is not None:
            minmax = [minmax[0] / voxel_size[0],
                      minmax[1] / voxel_size[1],
                      minmax[2] / voxel_size[2]]
        img = tr.as_stack(np.array(self.x)/voxel_size[2],
                          np.array(self.y)/voxel_size[1],
                          np.array(self.z)/voxel_size[0], minmax=minmax)
        return img

    def centrate(self):
        """
        Substract the mean coordinate form each coordinate value.
        """
        self.center = np.array([np.mean(self.x), np.mean(self.y), np.mean(self.z)])
        self.x = self.x - np.mean(self.x)
        self.y = self.y - np.mean(self.y)
        self.z = self.z - np.mean(self.z)

    def to_spherical(self):
        """
        Convert coordinates from Cartesian to spherical.
        """
        self.R, self.theta, self.phi = tr.cart_to_spherical(self.x, self.y, self.z)

    def rotate(self, theta, phi):
        """
        Rotate the coordinates of the surface by given asimuthal and polar angles.

        Parameters
        ----------
        theta : float
            Polar angle.
        phi : float
            Azimuthal angle.
        """
        self.x, self.y, self.z = tr.rotate_spherical(self.x, self.y, self.z, theta, phi)

    def interpolate(self, grid_size, r=None):
        """
        Interpolate the surface points on a regular grid.
        
        Parameters
        ----------
        grid_size : int
            Dimension of the square grid to interpolate the surface points.
        r : array, optional
            The array of values to interpolate.
            If None, the self.R (radius) will be interpolated.
            Default is None.

        Returns
        -------
        ndarray of size (grid_size x grid_size) : interpolated grid.

        """

        R, theta, phi = (self.R, self.theta, self.phi)
        if r is None:
            r = R

        if len(r) > 4:

            # make a lattice
            I = np.linspace(0, np.pi, grid_size, endpoint=False)
            J = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
            J, I = np.meshgrid(J, I)

            # make a list of shape points (theta and phi angles) and values (radius)
            values = r

            points = np.array([theta, phi]).transpose()

            # add 0 and pi to theta
            points = np.concatenate((points, np.array([[0, 0], [0, 2*np.pi], [np.pi, 0], [np.pi, 2*np.pi]])), axis=0)
            rmin = np.mean(r[np.where(theta == theta.min())])
            rmax = np.mean(r[np.where(theta == theta.max())])
            values = np.concatenate((values, np.array([rmin, rmin, rmax, rmax])), axis=0)

            # add shape points shifted to the left and right in the longitude dimension, to fill the edges
            points = np.concatenate((points, points - [0, 2*np.pi], points + [0, 2*np.pi]), axis=0)
            values = np.concatenate((values, values, values), axis=0)

            # make list of lattice points
            xi = np.asarray([[I[i, j], J[i, j]] for i in range(len(I)) for j in range(len(I[0]))])

            # interpolate the shape points on the lattice
            grid = griddata(points, values, xi, method='linear')
            grid = grid.reshape((grid_size, grid_size))

        else:
            grid = None

        return grid

    def plot_points(self, scale_factor=0.1):
        """
        Plot a 3D view of the surface points with mayavi.
        
        Returns
        -------
        mayavi scene
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = mlab.points3d(self.x, self.y, self.z, self.z, scale_mode='none',
                                     scale_factor=scale_factor, mode='sphere', colormap='gray').scene
            mesh.background = (1, 1, 1)
            mesh.magnification = 10
        return mesh

    def plot_surface(self, points=False, extent=None):
        """
        Plot a 3D view of the surface grid with mayavi.
        
        Parameters
        ----------
        points : bool, optional
            If True, surface points will be displayed.
            Default is False.
        extent : [xmin, xmax, ymin, ymax, zmin, zmax], optional
            Minimal and maximal coordinates to display.
            Default is the x, y, z arrays extent. 

        Returns
        -------
        mayavi scene
        """
        if self.Rgrid is not None:
            grid_size = self.Rgrid.shape[0]
            I = np.linspace(0, np.pi, grid_size + 1, endpoint=True)
            J = np.linspace(0, 2 * np.pi, grid_size + 1, endpoint=True)
            J, I = np.meshgrid(J, I)
            theta = I
            phi = J
            grid = np.zeros((grid_size + 1, grid_size + 1))
            grid[:-1, :-1] = self.Rgrid
            grid[-1] = grid[0]
            grid[:, -1] = grid[:, 0]

            x, y, z = tr.spherical_to_cart(1, theta, phi)
            mlab.clf()
            if extent is not None:
                mesh = mlab.mesh(x * np.abs(grid), y * np.abs(grid), z * np.abs(grid), scalars=grid,
                                 colormap='jet', extent=extent)
            else:
                mesh = mlab.mesh(x * np.abs(grid), y * np.abs(grid), z * np.abs(grid), scalars=grid,
                                 colormap='jet')
            if points:
                mesh = mlab.points3d(self.x, self.y, self.z, self.z, scale_mode='none', scale_factor=0.05)

            mesh.scene.background = (1, 1, 1)
            mesh.scene.magnification = 10
            return mesh.scene

    def compute_spharm(self, grid_size=None, normalize=False, normalization_method='zero-component', ri=False):
        """
        Compute the spherical harmonics spectrum of the current surface.
        
        Parameters
        ----------
        grid_size : int, optional
            Dimension of the square grid to interpolate the surface points.
            Will be used to interpolate the surface coordinates if self.Rgrid is None 
             (in this case it is a mandatory parameter).
            Default is None.
        normalize : bool, optional
            If True, the values of the spectrum will be normalized according to the `normalization_method`.
            Default is False.            
        normalization_method : str, optional
            If 'mean-radius', the grid values will be divided by the mean grid value prior to the SPHARM transform.
            If 'zero-component', all spectral components will be divided by the value of the first component (m=0, n=0).
            Default is 'zero-component'.
        ri : bool, optional
            If True, rotation-invariant spectrum based on Cartesian coordinates is computed.
            Default is False.

        Returns
        -------
        Spectrum : spherical harmonics spectrum of the surface.
        """
        self.spharm = Spectrum()

        if ri:
            data = []
            for r in [self.x, self.y, self.z]:
                grid = self.interpolate(grid_size=grid_size, r=r)
                self.spharm.from_surface(surface=grid, normalize=normalize)
                data.append(self.spharm.harmonics_csv)
            for c in ['value', 'amplitude', 'power', 'real', 'imag']:
                self.spharm.harmonics_csv[c] = np.sqrt(data[0][c]**2 + data[1][c]**2 + data[2][c]**2)
            self.spharm.harmonics_csv['amplitude2'] = np.abs(self.spharm.harmonics_csv['value'])
            self.spharm.convert_to_shtools_array()

        else:
            if self.Rgrid is None:
                if grid_size is None:
                    raise TypeError('Grid size for interpolation must be provided')
                else:
                    self.Rgrid = self.interpolate(grid_size=grid_size)
            if self.Rgrid is None:
                print(len(self.x), grid_size)
            self.spharm.from_surface(surface=self.Rgrid, normalize=normalize, normalization_method=normalization_method)
        self.spharm.metadata = self.metadata
        return self.spharm

    def inverse_spharm(self, lmax=None):
        """
        Inverse transform the SPHARM spectrum to surface using the given number of components.
        
        Parameters
        ----------
        lmax : int, optional
            The maximum spherical harmonic degree to be used in the inverse transform.
            If None, all degrees will be used.
            Default is None.

        Returns
        -------
        ndarray : reconstructed surface grid.
        """
        self.Rgrid = self.spharm.spharm_to_surface(lmax=lmax)
        return self.Rgrid





















