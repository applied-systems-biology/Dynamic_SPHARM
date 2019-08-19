from __future__ import division

import os
import numpy as np
import pandas as pd
import pylab as plt
import pyshtools.expand as shtools
import seaborn as sns

from helper_lib import filelib


class Spectrum(object):
    """
    Class for storing and handling the SPHARM spectrum of a surface.
    """

    def __init__(self, surface=None, filename=None, name=None):
        """
        Initialize spectrum from given surface or file.
        
        Parameters
        ----------
        surface : numpy.ndarray, dimension (n, n) or (n, 2*n), n is even, optional
            A 2D equally sampled (default) or equally spaced complex grid 
            that conforms to the sampling theorem of Driscoll and Healy (1994). 
            The first latitudinal band corresponds to 90 N, the latitudinal band for 90 S is not included, 
            and the latitudinal sampling interval is 180/n degrees. 
            The first longitudinal band is 0 E, the longitude band for 360 E is not included, 
            and the longitudinal sampling interval is 360/n for an equally 
            and 180/n for an equally spaced grid, respectively.
            If None, an empty spectrum will be initialized.
            Default is None.
        filename : str, optional
            Path to a surface file to read the surface data.
            If None, an empty spectrum will be initialized.
            Default is None.
        name : str, optional
            Name of the spectrum to display during plotting. 
            If None, the file name will be displayed.
            Default is None.
        """
        self.harmonics_csv = None
        self.harmonics_shtools = None
        self.frequency_spectrum = None
        self.name = name
        if surface is not None:
            self.from_surface(surface)
        elif filename is not None:
            self.from_file(filename)
            if name is None:
                self.name = filename
        self.metadata = pd.Series()

    def from_surface(self, surface, normalize=False, normalization_method='zero-component'):
        """
        Initialize the spectrum from a given surface.
        
        Parameters
        ----------
         surface : numpy.ndarray, dimension (n, n) or (n, 2*n), n is even
            A 2D equally sampled (default) or equally spaced complex grid 
            that conforms to the sampling theorem of Driscoll and Healy (1994). 
            The first latitudinal band corresponds to 90 N, the latitudinal band for 90 S is not included, 
            and the latitudinal sampling interval is 180/n degrees. 
            The first longitudinal band is 0 E, the longitude band for 360 E is not included, 
            and the longitudinal sampling interval is 360/n for an equally 
            and 180/n for an equally spaced grid, respectively.
        normalize : bool, optional
            If True, the values of the spectrum will be normalized according to the `normalization_method`.
            Default is False.            
        normalization_method : str, optional
            If 'mean-radius', the grid values will be divided by the mean grid value prior to the SPHARM transform.
            If 'zero-component', all spectral components will be divided by the value of the first component (m=0, n=0).
            Default is 'zero-component'.
        """
        if surface.shape[1] % 2 or surface.shape[0] % 2:
            raise ValueError("The number of samples in latitude and longitude, n, must be even")
        if surface.shape[1] == surface.shape[0]:
            s = 1
        elif surface.shape[1] == 2*surface.shape[0]:
            s = 2
        else:
            raise ValueError("GRIDDH must be dimensioned as (N, 2*N) or (N, N)")
        if normalization_method not in ['zero-component', 'mean-radius']:
            raise ValueError("Invalid value for `method`: must be \'zero-component\' or \'mean-radius\'")
        if normalize is True and normalization_method == 'mean-radius':
            surface = surface/np.mean(np.abs(surface))
        self.harmonics_shtools = shtools.SHExpandDHC(surface, sampling=s)
        if normalize is True and normalization_method == 'zero-component':
            self.harmonics_shtools = self.harmonics_shtools / self.harmonics_shtools[0][0, 0]
        self.convert_to_csv()
        return self.harmonics_shtools

    def spharm_to_surface(self, lmax=None):
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
        grid = shtools.MakeGridDHC(self.harmonics_shtools, lmax_calc=lmax).real
        return grid

    def convert_to_csv(self):
        """
        Convert the spectrum from the pyshtools format to a table form.
        """
        harm = self.harmonics_shtools
        harmdata = pd.DataFrame()
        for degree in range(len(harm[0])):
            for order in range(degree + 1):
                harmdata = harmdata.append(pd.Series({'degree': int(degree),
                                                      'order': int(order),
                                                      'value': harm[0][degree, order]}), ignore_index=True)

            for order in range(1, degree + 1):
                harmdata = harmdata.append(pd.Series({'degree': int(degree),
                                                      'order': -int(order),
                                                      'value': harm[1][degree, order]}), ignore_index=True)

        harmdata['amplitude'] = np.abs(harmdata['value'])
        harmdata['power'] = harmdata['amplitude']**2
        harmdata['real'] = np.real(harmdata['value'])
        harmdata['imag'] = np.imag(harmdata['value'])
        harmdata['degree'] = np.int_(np.real(harmdata['degree']))
        harmdata['order'] = np.int_(np.real(harmdata['order']))
        harmdata['harmonic'] = ''
        for i in range(len(harmdata)):
            harmdata.at[i, 'harmonic'] = 'm=' + str(harmdata.iloc[i]['degree']) \
                                          + ' n=' + str(harmdata.iloc[i]['order'])

        self.harmonics_csv = harmdata
        return harmdata

    def convert_to_shtools_array(self):
        """
        Convert the spectrum from the table form to pyshtools format.
        """
        harmdata = self.harmonics_csv
        size = len(harmdata['degree'].unique())
        harm = np.zeros([2, size, size], dtype=complex)
        for degree in range(len(harm[0])):
            for order in range(degree + 1):
                line = harmdata[(harmdata['degree'] == degree) & (harmdata['order'] == order)].iloc[0]
                harm[0][degree, order] = line['real'] + 1j * line['imag']

            for order in range(1, degree + 1):
                line = harmdata[(harmdata['degree'] == degree) & (harmdata['order'] == -order)].iloc[0]
                harm[1][degree, order] = line['real'] + 1j * line['imag']

        self.harmonics_shtools = harm
        return harm

    def from_file(self, filename):
        """
        Read the spectrum in the table form from a given file.
        
        Parameters
        ----------
        filename : str
            Path to the spectrum file.
        """
        if os.path.exists(filename):
            self.harmonics_csv = pd.read_csv(filename, sep='\t', index_col=0)
            self.convert_to_shtools_array()
        else:
            raise ValueError('Input file does not exist!')

    def save_to_csv(self, filename):
        """
        Save the table form of the spectrum to a csv file.
        
        Parameters
        ----------
        filename : str
            Path to the output file.
        """
        if filename is not None:
            filelib.make_folders([os.path.dirname(filename)])
            for col in self.metadata.index:
                self.harmonics_csv[col] = self.metadata[col]
            self.harmonics_csv.to_csv(filename, sep='\t')

    def compute_frequency_spectrum(self, norm=False):
        """
        Compute the frequency spectrum by summarizing all orders of a given degree.
        
        Parameters
        ----------
        norm : bool, optional
            If True, each component of the frequency spectrum will be divided by the value of the zero frequency.
            Default is False.
        """
        stat = self.harmonics_csv.groupby(['degree']).sum().reset_index()
        if norm:
            maxline = stat[stat['degree'] == 0].iloc[0]
            for col in stat.columns:
                if col != 'degree':
                    stat.loc[:, col] = stat[col] / maxline[col]
        stat['amplitude'] = np.sqrt(stat['power'])
        stat['harmonic'] = stat['degree']
        self.frequency_spectrum = stat

    def save_frequency_spectrum_to_csv(self, filename, name=None):
        """
        Save the frequency spectrum to a csv file.
        
        Parameters
        ----------
        filename : str
            Path to the output file.
        name : str, optional
            Text to label the spectrum.
            If None, the value of `self.name` will be used.
        """
        if filename is not None:
            filelib.make_folders([os.path.dirname(filename)])
            if name is not None:
                self.frequency_spectrum['Name'] = name
            elif self.name is not None:
                self.frequency_spectrum['Name'] = self.name
            else:
                self.frequency_spectrum['Name'] = filename
            self.frequency_spectrum.to_csv(filename, sep='\t')

    def heatmap(self, value='amplitude', title=None, cutoff=None, logscale=False, **kwargs):
        """
        Plot the SPHARM spectrum as a heatmap.
        
        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to plot.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        title : str, optional
            Text to display in the plot title.
            Default is None.
        cutoff : int, optional
            The number of degrees to display.
            If None, all degrees will be displayed.
        logscale : bool, optional
            If True, the logarithm of the value will be plotted.
            Default is False.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.heatmap function.

        Returns
        -------
        seaborn.heatmap().figure
            The heatmap with the SPHARM degree displayed vertically and SPHARM order horizontally.

        """
        norm = kwargs.pop('norm', False)
        stat = self.harmonics_csv
        if norm:
            stat.loc[:, value] = np.array(stat[value]) / stat[value].iloc[0]
        if cutoff is not None:
            stat = stat[stat.degree < cutoff]
        if logscale:
            stat.loc[:, value] = np.log(stat[value])
        hm = stat.pivot('degree', 'order', value)
        plt.clf()
        plt.figure(figsize=(6, 5))
        pl = sns.heatmap(hm, **kwargs)
        if title is None:
            if self.name is not None:
                title = self.name + '; value = ' + value
            else:
                title = 'value = ' + value
        plt.title(title)
        return pl.figure

    def frequency_plot(self, value='amplitude', title=None, cutoff=None, **kwargs):
        """
        Plot the frequency spectrum as a bar plot.
        
        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to plot.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        title : str, optional
            Text to display in the plot title.
            Default is None.
        cutoff : int, optional
            The number of frequency components to display.
            If None, all frequencies will be displayed.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.barplot function.

        Returns
        -------
        seaborn.barplot().figure
            The bar plot with the values of spectral frequencies.
        """
        norm = kwargs.pop('norm', False)
        if self.frequency_spectrum is None:
            self.compute_frequency_spectrum(norm=norm)
        stat = self.frequency_spectrum
        if cutoff is not None:
            stat = stat[stat.degree < cutoff]
        plt.clf()
        pl = sns.barplot(data=stat, x='degree', y=value, **kwargs)
        plt.xticks(stat['degree'].unique(), rotation='vertical')
        if title is None:
            if self.name is not None:
                title = self.name + '; value = ' + value
            else:
                title = 'value = ' + value
        plt.title(title)
        return pl.figure

    def return_feature_vector(self, cutoff=None, static_features='amplitude', rotation_invariant=True):
        """
        Return the amplitudes of all harmonic components below a given degree.
        
        Parameters
        ----------
        cutoff : int, optional
            The number of frequency components to display.
            If None, all frequencies will be displayed.
        static_features : str, optional
            Name of the feature to represent the harmonic coefficients.
            Valid values: 'amplitude', 'real_imag'.
            Default is 'amplitude'
        rotation_invariant : bool, optional
            If True, rotation-invariant descriptors (frequencies) will be computed.
            If False, the whole spectrum will be used as a feature vector.
            Default is True.

        Returns
        -------
        numpy.array
            The returned feature vector
        """
        stat = self.harmonics_csv
        if cutoff is None:
            cutoff = np.max(stat.degree)
        if static_features == 'real_imag':
            static_features = ['real', 'imag']
        else:
            static_features = [static_features]
        if rotation_invariant:
            self.compute_frequency_spectrum(norm=False)
            stat = self.frequency_spectrum

        features = []
        for value in static_features:
            features = features + list(stat[value][stat.degree < cutoff + 1])
        return np.array(features)



















