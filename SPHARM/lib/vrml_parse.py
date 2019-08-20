from __future__ import division

import os
import numpy as np
import pandas as pd

from helper_lib import filelib
from SPHARM.classes.node import Node


def extract_node_names(inputfile, outputfile=None):
    """
    Extract the names of the nodes from vrml file.
    
    Parameters
    ----------
    inputfile : str
        Path to the vrml or wrl file.
    outputfile : str, optional
        Path to the output file to save the node names.
    """
    f = open(inputfile)
    st = f.readlines()

    st = ''.join(st).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    pairs = {'{': '}', '[': ']', '(': ')'}
    root = Node(('root', ''))
    stack = []
    k = 0
    for j, s in enumerate(st):
        if s in pairs:
            parts = st[k:j].split(' ')
            name = ''
            for i in range(1, len(parts)):
                if len(parts[-i]) > 0:
                    name = parts[-i]
                    break
            node = Node((name, pairs[s]))
            if len(stack) > 0:
                stack[-1].add_text(st[k + 1:j])
            stack.append(node)
            k = j

        elif len(stack) > 0 and s == stack[-1].bracket:
            node = stack.pop()
            node.add_text(st[k + 1:j])
            if len(stack) > 0:
                stack[-1].add_child(node)
            else:
                root.add_child(node)
            k = j
    if outputfile is None:
        outputfile = inputfile[:-4] + '_nodes.txt'
    filelib.make_folders([os.path.dirname(outputfile)])
    f = open(outputfile, 'w')
    root.print_children(outputfile=f)

    f.close()


def extract_node_names_batch(inputfolder, outputfolder):
    """
    Extract the names of the nodes from vrml file in a parallel mode.
    
    Parameters
    ----------
    inputfolder : str
        Path to a directory with vrml / wrl files to extract the node names.
    outputfolder : str
        Path to a directory to save the output.
    """
    files = os.listdir(inputfolder)
    for fn in files:
        ext = fn.split('.')[-1]
        if ext in ['wrl', 'vrml']:
            extract_node_names(inputfolder + fn, outputfile=outputfolder + fn)


def extract_key_nodes(inputfile, key):
    """
    Extract a list of nodes with a given name from a given vrml file.
    
    Parameters
    ----------
    inputfile : str
        Path to a vrml or wrl file
    key : str
        Target node name to extract.
    """
    f = open(inputfile)
    st = f.readlines()
    f.close()

    st = ''.join(st).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    pairs = {'{': '}', '[': ']', '(': ')'}
    root = Node(('root', ''))
    stack = []
    k = 0
    for j, s in enumerate(st):
        if s in pairs:
            parts = st[k:j].split(' ')
            name = ''
            for i in range(1, len(parts)):
                if len(parts[-i]) > 0:
                    name = parts[-i]
                    break
            node = Node((name, pairs[s]))
            if len(stack) > 0:
                stack[-1].add_text(st[k + 1:j])
            stack.append(node)
            k = j

        elif len(stack) > 0 and s == stack[-1].bracket:
            node = stack.pop()
            node.add_text(st[k + 1:j])
            if len(stack) > 0:
                stack[-1].add_child(node)
            else:
                root.add_child(node)
            k = j

    nodes = []
    root.extract_key_nodes(key=key, nodes=nodes)
    return nodes


def extract_coordinates(inputfile, outputfile):
    """
    Extract cell coordinates from a given vrml file.
    
    Parameters
    ----------
    inputfile : str
        Path to a vrml or wrl file with cell coordinates.
    outputfile : str
        Path to save the extracted coordinates in a table form.
    """


    stat = pd.DataFrame()
    curcoords = []
    timepoint = 0
    node_id = 0

    nodes = extract_key_nodes(inputfile, key='children')
    for node in nodes:
        if node.children[0].name == 'Shape':
            timepoint += 1
            for subnode1 in node.children:
                for subnode in subnode1.children:
                    if subnode.name == 'IndexedFaceSet':
                        curstat, curcoords = subnode.extract_coordinates(curcoords)
                        curstat['ID'] = node_id
                        curstat['Time'] = timepoint
                        stat = pd.concat([stat, curstat], ignore_index=True)
                        node_id += 1

    filelib.make_folders([os.path.dirname(outputfile)])
    stat.to_csv(outputfile, sep='\t')


def extract_coordinates_batch(inputfolder, outputfolder):
    """
    Extract cell coordinates from vrml files located in a given directory in a parallel mode.
    
    Parameters
    ----------
    inputfolder : str
        Path to the input directory.
    outputfolder : str
        Path to the output directory.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['wrl', 'vrml'])
    for fn in files:
        ext = fn.split('.')[-1]
        if ext in ['wrl', 'vrml']:
            extract_coordinates(inputfolder + fn, outputfolder + fn[:-4] + '.csv')


def combine_with_track_data(inputfile, trackfile, outputfile=None):
    """
    Add track IDs to the extracted coordinates.
    
    Parameters
    ----------
    inputfile : str
        Path to a file with extracted cell coordinates.
    trackfile : str
        Path to a file with track IDs.
    outputfile : str
        Path to the output file.
    """
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    if 'ID' not in stat.columns:
        stat['ID'] = stat['Cell_ID']
    summary = stat.groupby(['ID', 'Time']).mean().reset_index()
    trackstat = pd.read_excel(trackfile, sheet_name='Position', header=1)
    if 'Time' not in trackstat.columns:
        trackstat['Time'] = trackstat['Death [s]']

    for t in trackstat['Time'].unique():
        curstat = summary[summary['Time'] == t].reset_index()
        curtrackstat = trackstat[trackstat['Time'] == t].reset_index()
        for i in range(len(curtrackstat)):
            dist = np.sqrt((curstat['X'] - np.array(curtrackstat.iloc[i]['Position X']))**2 +
                           (curstat['Y'] - np.array(curtrackstat.iloc[i]['Position Y']))**2 +
                           (curstat['Z'] - np.array(curtrackstat.iloc[i]['Position Z']))**2)
            track_id = curtrackstat.iloc[i]['TrackID']
            if len(dist) > 0:
                ind = np.argmin(dist)
                stat.at[stat[(stat['ID'] == curstat.iloc[ind]['ID']) & (stat['Time'] == t)].index, 'TrackID'] = track_id
            else:
                print(trackfile, t, track_id)

    if outputfile is None:
        outputfile = inputfile[:-4] + '_tracked.csv'
    filelib.make_folders([os.path.dirname(outputfile)])
    stat.to_csv(outputfile, sep='\t')


def combine_with_track_data_batch(inputfolder, trackfolder, outputfolder):
    """
    Add track IDs to the extracted coordinates in a parallel mode.
    
    Parameters
    ----------
    inputfolder : str
        Path to a directory with coordinate files.
    trackfolder : str
        Path to a directory with track files.
    outputfolder : str
        Path to the output directory.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])
    trackfiles = filelib.list_subfolders(trackfolder, extensions=['xls', 'xlsx'])
    for fn in files:
        parts = fn.split('/')[-1].split('_')
        stem = parts[0] + '_' + parts[1]
        for trf in trackfiles:
            if trf.split('/')[0] == fn.split('/')[0] and len(trf.split(stem)) > 1:
                combine_with_track_data(inputfile=inputfolder + fn,
                                        trackfile=trackfolder + trf,
                                        outputfile=outputfolder + fn)






