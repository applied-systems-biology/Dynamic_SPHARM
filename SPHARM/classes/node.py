from __future__ import division

import re
import numpy as np
import pandas as pd


class Node(object):
    """
    Class for extracting the structure of vrml files
    """
    def __init__(self, dict):
        """
        Initialize an instance of the "Node" class from a file name.
        
        Parameters
        ----------
        dict : tuple of type (str, char)
            A tuple containing the name of the node and 
              the type of closing bracket that determines the end of the node content (e.g. '}')
        """
        self.name = dict[0]
        self.bracket = dict[1]
        self.children = []
        self.text = ''

    def add_child(self, node):
        """
        Add a child node to the list of childs.
        
        Parameters
        ----------
        node : Node
            The child node.
        """
        self.children.append(node)

    def add_text(self, text):
        """
        Append given text to the current text variable.
        
        Parameters
        ----------
        text : str
            The text to append.
        """
        self.text = self.text + text

    def print_children(self, offset=None, outputfile=None):
        """
        Print the name of the current node and all child nodes.
        
        Parameters
        ----------
        offset : str, optional
            Horizontal offset to print the child nodes (e.g. '    ').
            If None, the offset is set to an empty string ('', no offset).
            Default is None.
        outputfile : file, optional
            File to save the output.
            If None, the output will be printed out in the console.
            Default is None.
        """
        if offset is None:
            offset = ''
        if outputfile is None:
            print(offset + self.name)
        else:
            outputfile.write(offset + self.name + '\n')
        for i in range(len(self.children)):
            self.children[i].print_children(offset=offset + ' ', outputfile=outputfile)

    def extract_key_nodes(self, key, nodes=None):
        """
        Extract a list of (child) nodes with a given name.
        
        Parameters
        ----------
        key : str
            The key name to extract.
        nodes : list, optional
            The list of already extracted nodes.
            If None, the value is set to an empty list.
            Default is None.
        """
        if nodes is None:
           nodes = []
        if self.name == key:
            nodes.append(self)
        for i in range(len(self.children)):
            self.children[i].extract_key_nodes(key, nodes=nodes)

    def extract_coordinates(self, curcoords):
        """
        Extract cell coordinates. 
        
        Parameters
        ----------
        curcoords : list
            Previously extracted coordinates.
        """
        stat = pd.DataFrame()
        p = re.compile('[-+]?\d+\.*\d*e?[-+]?\d*')
        for child in self.children:
            if child.name == 'Coordinate':
                for ch in child.children:
                    if ch.name == 'point':
                        num = np.float_(p.findall(ch.text))
                        curcoords = np.reshape(num, [int(len(num)/3), 3])

            if child.name == 'coordIndex':
                ind = np.unique(np.int_(p.findall(child.text)))
                ind = ind[np.where(ind >= 0)]
                coord = curcoords[ind]
                stat['X'] = coord[:, 0]
                stat['Y'] = coord[:, 1]
                stat['Z'] = coord[:, 2]
        return stat, curcoords