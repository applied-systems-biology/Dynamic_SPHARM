from __future__ import division

import os
import numpy as np

from helper_lib import filelib
import helper_lib.parallel as prl


def run_parallel(**kwargs):
    """
    Run a given function in a parallel manner. 
    
    Parameters
    ----------
    kwargs : key, value pairings
        Arbitrary keyword arguments

    Keyword arguments
    -----------------
    *items* : list
        List of items. For each item, the `process` will be called.
        The value of the `item` parameter of `process` will be set to the value of the current item from the list.
        Remaining keyword arguments will be passed to the `process`
    *max_threads* : int, optional
        The maximal number of processes to run in parallel
        Default is 8
    *process* : callable
        The function that will be applied to each item of `kwargs.items`.
        The function should accept the argument `item`, which corresponds to one item from `kwargs.items`.
        An `item` is usually a name of the file that has to be processed or 
            a list of files that have to be combined / convolved /analyzed together.
        The function should not return any output, but the output should be saved in a specified directory.
    *inputfolder* : str
        Input directory with files to process.
    *outputfolder* : str
        Output directory to save the results.
    """

    files = filelib.list_subfolders(kwargs.get('inputfolder'), extensions=kwargs.get('extensions'))
    channelcodes = kwargs.get('channels', None)
    exclude = kwargs.get('exclude', None)
    if channelcodes is not None:
        files = list_of_files_to_combine(files, channelcodes)

    if exclude is not None:
        nfiles = []
        for fn in files:
            cellfile = True
            for excl in exclude:
                if fn[-len(excl):] == excl:
                    cellfile = False
            if cellfile:
                nfiles.append(fn)
        files = nfiles

    if kwargs.get('debug'):
        kwargs['item'] = files[0]
        kwargs.get('process')(**kwargs)
    else:
        kwargs['items'] = files
        prl.run_parallel(**kwargs)

        if kwargs.get('combine', True) and os.path.exists(kwargs.get('outputfolder', 'no_folder')):
            filelib.combine_statistics(kwargs.get('outputfolder'))


def list_of_files_to_combine(files, channelcodes):
    """
    Extract the channel information from file names and group file names of corresponding channels.
    
    Parameters
    ----------
    files : list
        List of file names
    channelcodes : list of str
        List of channel codes as they appear in the file names.
    """
    samples = []
    channels = []
    nfiles = []

    for fn in files:
        for i, cc in enumerate(channelcodes):
            if len(cc) > 0:
                parts = fn.split(cc)
                if len(parts) > 1:
                    nfiles.append(fn)
                    samples.append(parts[0] + parts[-1].split('Time')[-1])
                    channels.append(i)
                    break

    samples = np.array(samples)
    channels = np.array(channels)
    nfiles = np.array(nfiles)

    usamples = np.unique(samples)

    files = []
    for sample in usamples:
        curfiles = []
        for i in range(len(channelcodes)):
            fn = nfiles[(samples == sample) & (channels == i)]
            if len(fn) > 0:
                curfiles.append(fn[0])
            else:
                curfiles.append('')

        files.append(curfiles)

    return files
