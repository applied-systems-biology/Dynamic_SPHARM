from __future__ import division

import numpy as np
from skimage import io
import warnings

from helper_lib import filelib

from SPHARM.classes.time_spectrum import TimeSpectrum


class MovingSurface(object):
    """
    Class for storing the dynamics of an object surface.
    """
    def __init__(self, name=None):
        """
        Initiate the time series for the surface.
        
        Parameters
        ----------
        name : str, optional
            Name of the surface to add to the output file when saving results. 
            Default is None.
        """
        if name is None:
            self.name = ''
        else:
            self.name = name
        self.surfaces = []
        self.times = []
        self.timespectrum = TimeSpectrum(name=name)
        self.minmax = np.array([[1000., -1000.], [1000., -1000.], [1000., -1000.]])
        self.centers = []

    def add_surface(self, surface, timepoint=None):
        """
        Add new surface to the time series.
        
        Parameters
        ----------
        surface : Surface
            Object surface at one time point.
        timepoint : scalar, optional
            Time point to assign to the surface.
            If None, the number of previously added surfaces will be used to label the time point 
             (e.g. 0 if none were added).
            Default is None.
        """
        self.surfaces.append(surface)
        if timepoint is None:
            self.times.append(len(self.surfaces))
        else:
            self.times.append(timepoint)
        self.minmax[0, 0] = min(self.minmax[0, 0], np.min(surface.z))
        self.minmax[1, 0] = min(self.minmax[1, 0], np.min(surface.y))
        self.minmax[2, 0] = min(self.minmax[2, 0], np.min(surface.x))
        self.minmax[0, 1] = max(self.minmax[0, 1], np.max(surface.z))
        self.minmax[1, 1] = max(self.minmax[1, 1], np.max(surface.y))
        self.minmax[2, 1] = max(self.minmax[2, 1], np.max(surface.x))
        self.centers.append(surface.center)

    def compute_timespectrum(self, gridsize):
        """
        Compute a SPHARM spectrum for each surface and generate a TimeSpectrum object.
        
        Parameters
        ----------
        gridsize : int, optional
            Dimension of the square grid to interpolate the surface points.
            Will be used to interpolate the surface coordinates if self.Rgrid is None 
             (in this case it is a mandatory parameter).
            Default is None.
        """
        for i, surface in enumerate(self.surfaces):
            surface.compute_spharm(grid_size=gridsize)
            self.timespectrum.add_spectrum(surface.spharm, timepoint=self.times[i])

    def plot_surfaces(self, outputfolder, points=False):
        """
        Plot 3D views of all surfaces with mayavi and save to png files.
        
        Parameters
        ----------
        outputfolder : str
            Directory to save the plots.
        points : bool, optional
            If True, surface points will be displayed.
            Default is False.
        """
        filelib.make_folders([outputfolder])
        extent = np.array([self.minmax[2], self.minmax[1], self.minmax[0]]).flatten()
        for i, surface in enumerate(self.surfaces):
            mesh = surface.plot_surface(points=points, extent=extent)
            mesh.magnification = 3
            mesh.save(outputfolder + self.name + '_%03d.png' % self.times[i], size=(200, 200))

    def plot_max_projections(self, outputfolder, voxel_size):
        """
        Plot maxium projections of all surfaces and save to png files.
        
        Parameters
        ----------
        outputfolder : str
            Directory to save the plots.
        voxel_size : scalar or sequence of scalars
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
        """
        filelib.make_folders([outputfolder])
        voxel_size = np.array([voxel_size]).flatten()
        if len(voxel_size) == 1:
            voxel_size = np.ones(3)*voxel_size
        for i, surface in enumerate(self.surfaces):
            stack = surface.as_stack(voxel_size=voxel_size, minmax=self.minmax)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(outputfolder + 'xy_' + self.name + '_%03d.png' % self.times[i],
                          stack.max(0).astype(np.uint8))
                io.imsave(outputfolder + 'xz_' + self.name + '_%03d.png' % self.times[i],
                          stack.max(1).astype(np.uint8))
                io.imsave(outputfolder + 'yz_' + self.name + '_%03d.png' % self.times[i],
                          stack.max(2).astype(np.uint8))




