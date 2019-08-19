from __future__ import division

import os
import pylab as plt
from helper_lib.filelib import make_folders


class Profile(object):
    """
    Class to handle the surface profile of an ellipsoid obtained after a cut by a xy, yz or xz plane.
    """
    def __init__(self, r, theta):
        """
        Generate a profile from give polar angle and radius.
        
        Parameters
        ----------
        r : sequence of scalars
            Radius
        theta : sequence of scalars
            Polar angle
        """
        self.theta = theta
        self.R = r

    def plot(self):
        """
        Plot and display the profile.
        """
        plt.clf()
        ax = plt.subplot(111, projection='polar')
        ax.plot(self.theta, self.R, color='r', linewidth=3)
        plt.show()

    def save(self, outputfile):
        """
        Plot the profile and save to a given file.
        
        Parameters
        ----------
        outputfile : str
            Output file name to save the plotted profile.
        """
        make_folders([os.path.dirname(outputfile)])
        plt.clf()
        ax = plt.subplot(111, projection='polar')
        ax.plot(self.theta, self.R, color='r', linewidth=3)
        plt.savefig(outputfile)

