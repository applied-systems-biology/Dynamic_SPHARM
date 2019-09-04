from __future__ import division
import os
import re
import pandas as pd

from helper_lib import filelib
from SPHARM.classes.image_stack import ImageStack


def make_metadata_files(**kwargs):
    """
    Generate metadata file with the given voxel size.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory to save the metadata file.
    *item* : str
        File name for the metadata file.
    *voxel_size* : list of scalars
        Voxel sizes for z and xy dimensions.
    """
    inputfolder = kwargs.get('inputfolder')
    filename = kwargs.get('item')
    voxel_size = kwargs.get('voxel_size')
    metadata = pd.Series({'voxel_size_z': voxel_size[0],
                          'voxel_size_xy': voxel_size[1]})
    metadata.to_csv(inputfolder + filename[:-4] + '.txt', sep='\t')


def extract_surfaces(**kwargs):
    """
    Extract surface coordinates of each connected region in a given image.
    
    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input image.
    *item* : str
        File name of the input image.
    *outputfolder* : str
        Directory to save the segmented image.
    *channelcodes* : list of str
        List of channel codes as they appear in the file names.
    *reconstruct* : bool, optional
        If True, surfaces will be reconstructed by the marching cube algorithm, 
          and coordiantes of the vertices will be extracted.
        If False, coordinates of the voxels connected to the background will be extracted.
        Default is True.
    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../surfaces/')
    filename = kwargs.get('item')
    reconstruct = kwargs.get('reconstruct', True)
    channelcodes = kwargs.get('channelcodes')

    channel = None
    if channelcodes is not None:
        for i, cc in enumerate(channelcodes):
            if len(filename.split(cc)) > 1:
                channel = i
    if channelcodes is None or channel is not None:
        stack = ImageStack(inputfolder + filename)
        metadata = pd.read_csv(inputfolder + filename[:-4] + '.txt',
                               sep='\t', index_col=0, header=-1).transpose().iloc[0].T.squeeze()
        min_coord = None
        if 'min_x' in metadata.index and 'min_y' in metadata.index and 'min_z' in metadata.index:
            min_coord = [metadata['min_z'], metadata['min_y'], metadata['min_x']]
        stack.filename = filename
        stack.extract_surfaces(outputfolder,
                               voxel_size=[metadata['voxel_size_z'],
                                           metadata['voxel_size_xy'],
                                           metadata['voxel_size_xy']],
                               reconstruct=reconstruct, min_coord=min_coord)


def combine_surfaces(inputfolder, outputfolder):
    """
    Combine surface files located in the same subfolder of a given input folder.

    Parameters
    ----------
    inputfolder : str
        Input directory with files to combine.
    outputfolder : str
        Output directory to save the combined files.
    """
    filelib.make_folders([outputfolder])
    folders = os.listdir(inputfolder)
    p = re.compile('\d*\.*\d+')
    for folder in folders:
        files = filelib.list_subfolders(inputfolder + folder + '/', extensions=['csv'])
        stat = pd.DataFrame()
        for fn in files:
            curstat = pd.read_csv(inputfolder + folder + '/' + fn, sep='\t')
            curstat['Time'] = p.findall(fn.split('/')[-1])[-2]
            stat = pd.concat([stat, curstat], ignore_index=True)
        stat.to_csv(outputfolder + folder + '.csv', sep='\t')


def split_to_surfaces(inputfile, outputfolder, combine_tracks=False,
                      adjust_frame_rate=False, metadata_file=None):
    """
    Split one surface file into separate files for surfaces of individual cells.
    
    Parameters
    ----------
    inputfile : str
        Input surface file.
    outputfolder : str
        Directory to save the output surface files.
    combine_tracks : bool, optional
        If True, connected time points will be combined into one file.
        Default is False.
    """
    filelib.make_folders([outputfolder])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    framerate = None
    if adjust_frame_rate:
        parts = inputfile.split('/')[-1].split('_')
        stem = parts[0] + '_' + parts[1]
        metadata = pd.read_csv(metadata_file, sep='\t')
        ind = metadata[metadata['Sample'] == stem].index[0]
        framerate = metadata.loc[ind, 'time']

    for track_id in stat['TrackID'].unique():
        combined_stat = pd.DataFrame()
        ntime = 1
        for t in stat['Time'].unique():
            curstat = stat[(stat['TrackID'] == track_id) & (stat['Time'] == t)].reset_index()
            if adjust_frame_rate and framerate == 20:
                if (t-1) % 3:
                    curstat = pd.DataFrame()
                else:
                    curstat['Time'] = ntime
                    ntime += 1
            if combine_tracks:
                combined_stat = pd.concat([combined_stat, curstat], ignore_index=True)
            else:
                if len(curstat) > 0:
                    curstat.to_csv(outputfolder + 'Track_' + str(int(track_id)) + '_Time_%03d.csv' % t, sep='\t')
        if combine_tracks and len(combined_stat) > 0:
            combined_stat.to_csv(outputfolder + 'Track_' + str(int(track_id)) + '.csv', sep='\t')


def split_to_surfaces_batch(inputfolder, outputfolder, combine_tracks=False,
                            adjust_frame_rate=False, metadata_file=None):
    """
    Split one surface files located in a given folder into separate files for surfaces of individual cells.
    
    Parameters
    ----------
    inputfolder : str
        Input directory
    outputfolder : str
        Output directory
    combine_tracks : bool, optional
        If True, connected time points will be combined into one file.
        Default is False.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])
    for fn in files:
        print(fn)
        ext = fn.split('.')[-1]
        if ext in ['csv']:
            split_to_surfaces(inputfolder + fn, outputfolder + fn[:-4] + '/', combine_tracks=combine_tracks,
                              adjust_frame_rate=adjust_frame_rate, metadata_file=metadata_file)




