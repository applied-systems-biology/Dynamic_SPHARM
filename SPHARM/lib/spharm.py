from __future__ import division
import os
import pandas as pd
import numpy as np

from helper_lib import filelib
from SPHARM.classes.surface import Surface
from SPHARM.classes.moving_surface import MovingSurface
from SPHARM.lib import transformation as tr


def compute_spharm(**kwargs):
    """
    Compute spherical harmonics spectra for a given surface.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input surface.
    *item* : str
        File name of the input surface.
    *outputfolder* : str
        Directory to save the computed spectra.
    *grid_size* : int
        Dimension of the square grid to interpolate the surface points.
    *normalize* : bool
        If True, the values of the spectrum will be normalized according to the `normalization_method`.
    *normalization_method* : str, optional
        If 'mean-radius', the grid values will be divided by the mean grid value prior to the SPHARM transform.
        If 'zero-component', all spectral components will be divided by the value of the first component (m=0, n=0).
        Default is 'zero-component'. 
    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../spharm/')
    filename = kwargs.get('item')
    combined_tracks = kwargs.get('combined_tracks', False)
    rotate = kwargs.get('rotate', False)

    filelib.make_folders([os.path.dirname(outputfolder[:-1] + '_kwargs.csv')])
    pd.Series(kwargs).to_csv(outputfolder[:-1] + '_kwargs.csv', sep='\t', header=False)

    if not (os.path.exists(outputfolder+filename) or
            os.path.exists(outputfolder + filename[:-4] + '_Time_%03d.csv' % 1)):

        if combined_tracks:
            stat = pd.read_csv(inputfolder + filename, sep='\t', index_col=0)
            stat.at[:, 'Time'] = np.array(stat['Time']).astype(float).astype(int)
            stat = stat.sort_values('Time').reset_index()
            t_surface = MovingSurface()
            if not filename.endswith('.csv'):
                filename += '.csv'
            times = stat['Time'].unique()
            if len(times) > 2:
                for t in times:
                    curstat = stat[stat['Time'] == t]
                    if len(curstat) > 4:
                        surface = Surface(data=curstat)
                        surface.metadata['Name'] = filename[:-4] + '_Time_%03d.csv' % t
                        if 'TrackID' in curstat.columns:
                            surface.metadata['TrackID'] = curstat.iloc[0]['TrackID']
                        surface.centrate()
                        if len(t_surface.surfaces) > 0:
                            x, y, z = surface.center - t_surface.surfaces[-1].center  # direction of the previous interval
                            surface.migration_angles = tr.cart_to_spherical(x, y, z)[1:]
                        t_surface.add_surface(surface)
                    else:
                        print(filename, times, t, len(curstat))
                t_surface.surfaces[0].migration_angles = t_surface.surfaces[1].migration_angles

                for surface in t_surface.surfaces:
                    if rotate:
                        surface.rotate(surface.migration_angles[0], surface.migration_angles[1])
                    surface.to_spherical()
                    surface.compute_spharm(grid_size=kwargs.get('grid_size'), normalize=kwargs.get('normalize'),
                                           normalization_method=kwargs.get('normalization_method', 'zero-component'))
                    surface.spharm.save_to_csv(outputfolder + surface.metadata['Name'])

        else:
            surface = Surface(filename=inputfolder + filename)
            surface.centrate()
            surface.to_spherical()
            surface.compute_spharm(grid_size=kwargs.get('grid_size'), normalize=kwargs.get('normalize'),
                                   normalization_method=kwargs.get('normalization_method', 'zero-component'))
            if not filename.endswith('.csv'):
                filename += '.csv'
            surface.spharm.save_to_csv(outputfolder + filename)


def compute_frequency_spectra(**kwargs):
    """
    Compute frequency spectra for a given surface.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input surface.
    *item* : str
        File name of the input surface.
    *outputfolder* : str
        Directory to save the computed spectra.
    *grid_size* : int
        Dimension of the square grid to interpolate the surface points.
    *normalize* : bool
        If True, the values of the spectrum will be normalized according to the `normalization_method`.
    *normalization_method* : str, optional
        If 'mean-radius', the grid values will be divided by the mean grid value prior to the SPHARM transform.
        If 'zero-component', all spectral components will be divided by the value of the first component (m=0, n=0).
        Default is 'zero-component'. 
    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../spharm/')
    filename = kwargs.get('item')

    if not os.path.exists(outputfolder+filename):

        surface = Surface(filename=inputfolder + filename)
        surface.centrate()
        surface.to_spherical()
        surface.compute_spharm(grid_size=kwargs.get('grid_size'), normalize=kwargs.get('normalize'),
                               normalization_method=kwargs.get('normalization_method'))
        surface.spharm.save_to_csv(outputfolder + filename)


def convert_surfaces(**kwargs):
    """
    Convert surface file from txt to a csv format.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input surface.
    *item* : str
        File name of the input surface.
    *outputfolder* : str
        Directory to save the converted surface.

    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../surfaces/')
    filename = kwargs.get('item')

    surface = Surface(filename=inputfolder+filename, **kwargs)
    surface.save(outputfolder+filename+'.csv')


def convert_to_tiff(**kwargs):
    """
    Save the surface as a 3D image stack.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input surface.
    *item* : str
        File name of the input surface.
    *outputfolder* : str
        Directory to save the output 3D stack.
    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../stacks/')
    filename = kwargs.get('item')

    surface = Surface(filename=inputfolder+filename, **kwargs)
    surface.save_as_stack(outputfolder+filename+'.tif', voxel_size=kwargs.get('voxel_size'))



