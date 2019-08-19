from __future__ import division

import os
import re
import numpy as np
import pandas as pd
from skimage import measure
from skimage.segmentation import find_boundaries
from skimage import io
from skimage.exposure import rescale_intensity
from mayavi import mlab
import warnings

from helper_lib import filelib


class ImageStack(object):
    """
    Class for handling 3D stacks
    """

    def __init__(self, filename, load=True):
        """
        Initialize an instance of the "ImageStack" class from a file name.
        
        Parameters
        ----------
        filename : str
            Path to the file with image data, or a string to extract metadata.
        load : bool, optional
            If True, the image data will be loaded from the file specified by `filename`.
            Default is True.
        """
        self.filename = filename
        self.data = None
        self.channel = None
        self.timepoint = None
        self.channel = 0
        self.parse_filename()
        if load:
            self.load_data()

    def parse_filename(self):
        """
        Extract information about the time point and the channel from the filename given by `self.filename`        
        """
        filename = self.filename

        p = re.compile('\d+')
        parts = filename.split('Time')
        if len(parts) > 1:
            num = p.findall(parts[-1])
            if len(num) > 0:
                self.timepoint = int(num[-1])

        p1 = re.compile('C.*Time')
        parts = p1.findall(filename)
        if len(parts) > 0:
            num = p.findall(parts[-1])
            if len(num) > 0:
                self.channel = int(num[0])

    def load_data(self):
        """
        Load the image data from the file specified by `self.filename`
        """
        self.data = io.imread(self.filename)

    def save(self, filename=None):
        """
        Save the image given in `self.data` into a given file.
        
        Parameters
        ----------
        filename : str, optional
            Path to a file to save the image.
            If None, the image will be saved into a file specified by `self.filename`.
            Default is None.
        """
        if filename is None:
            filename = self.filename

        filelib.make_folders([os.path.dirname(filename)])

        if self.data.max() > 255:
            self.data = self.data.astype(np.uint16)
        else:
            self.data = self.data.astype(np.uint8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filename, self.data)

    def save_max_projection(self, filename, axis=0):
        """
        Save maximum projection of the image on a given axis.
        
        Parameters
        ----------
        filename : str 
            Path to save the maximum projection.
        axis : int, optional
            Axis for the maximum projection.
            Default is 0.
        """
        maxproj = np.max(self.data, axis=axis)
        filelib.make_folders([os.path.dirname(filename)])
        if maxproj.max() > 255:
            maxproj = rescale_intensity(maxproj, out_range=(0, 255))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filename, maxproj.astype(np.uint8))

    def extract_surfaces(self, outputfolder, voxel_size=1, reconstruct=True, min_coord=None):
        """
        Extract surface coordinates of each connected region.
        
        Parameters
        ----------
        outputfolder : str
            Path to a directory to save the surfaces.
        voxel_size : scalar or sequence of scalars, optional
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
            Default is 1.
        reconstruct : bool, optional
            If True, surfaces will be reconstructed by the marching cube algorithm, 
              and coordiantes of the vertices will be extracted.
            If False, coordinates of the voxels connected to the background will be extracted.
            Default is True.
        min_coord : sequence of scalars, optional
            Starting coordinates of the surface.
            Three values: for z, y, and x are expected.
            In not None, these values will be added to all surface coordinates.
        """

        voxel_size = np.array([voxel_size]).flatten()
        if len(voxel_size) < 3:
            voxel_size = [voxel_size[0]]*3
        filelib.make_folders([outputfolder + os.path.dirname(self.filename)])

        llist = np.unique(self.data)[1:]

        if not reconstruct:
            border = find_boundaries(self.data)*self.data

        for i, l in enumerate(llist):
            if reconstruct:
                mask = np.where(self.data == l, 1, 0)
                verts = np.array(measure.marching_cubes_lewiner(mask, 0, spacing=tuple(voxel_size))[0])
                verts = verts.transpose()
            else:
                verts = np.array(np.where(border == l))
                for iv in range(3):
                    verts[iv] = verts[iv]*voxel_size[iv]
            if min_coord is not None:
                for i in range(3):
                    verts[i] += min_coord[i]
            stat = pd.DataFrame({'Z': verts[0], 'Y': verts[1], 'X': verts[2]})
            stat['Cell_ID'] = l
            stat['Image_name'] = self.filename
            if len(stat) > 0:
                stat.to_csv(outputfolder + self.filename[:-4] + '_Cell%05d.csv' % l, sep='\t')

    def save_3Dview(self, filename, voxel_size=1):
        """
        Save a 3D view of the image stack.
        
        Parameters
        ----------
        filename : str
            Path to save the 3D view.
        voxel_size : scalar or sequence of scalars, optional
            Voxel size of the image. 
            Specified either by individual value for each axis, or by one value for all axes.
            Default is 1.
        """
        filelib.make_folders([os.path.dirname(filename)])
        voxel_size = np.array([voxel_size]).flatten()
        if len(voxel_size) < 3:
            voxel_size = [voxel_size[0]]*3
        mlab.clf()
        s = mlab.pipeline.scalar_field(self.data)
        s.spacing = voxel_size/np.min(voxel_size)*5
        mesh = mlab.pipeline.volume(s, color=(1, 0, 0)).scene
        mesh.background = (1, 1, 1)
        mesh.magnification = 3
        mesh.save(filename, size=(200, 200))





