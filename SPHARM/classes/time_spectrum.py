from __future__ import division

import os
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.fftpack import fft

from helper_lib import filelib


class TimeSpectrum(object):
    """
    Class for storing and handling the dynamics of SPHARM spectrum of a cell.
    """

    def __init__(self, name=None):
        """
        Initialize the time spectrum.
        
        Parameters
        ----------
        name : str, optional
            Name of the spectrum to display during plotting. 
            Default is None.
        """
        self.name = name
        self.spectra = []
        self.frequencies = pd.DataFrame()
        self.data = pd.DataFrame()
        self.frequency_data = pd.DataFrame()
        self.derivative = pd.DataFrame()
        self.mean_abs_derivative = pd.DataFrame()

    def add_spectrum(self, spectrum, timepoint=None):
        """
        Add new SPHARM spectrum to the time series.
        
        Parameters
        ----------
        spectrum : Spectrum
            SPHARM spectrum of one time point.
        timepoint : scalar, optional
            Time point to assign to the spectrum.
            If None, the number of previously added spectra will be used to label the time point 
             (e.g. 0 if none were added).
            Default is None.
        """
        if timepoint is None:
            timepoint = len(self.spectra)
        if len(self.data) == 0 or timepoint not in self.data['Time'].unique():
            self.spectra.append(spectrum)
            curstat = spectrum.harmonics_csv
            curstat.loc[:, 'Time'] = timepoint
            self.data = pd.concat([self.data, curstat], ignore_index=True)

            if spectrum.frequency_spectrum is None:
                spectrum.compute_frequency_spectrum(norm=False)
            curstat = spectrum.frequency_spectrum
            curstat.loc[:, 'Time'] = timepoint
            self.frequency_data = pd.concat([self.frequency_data, curstat], ignore_index=True)

    def save_to_csv(self, filename, name=None):
        """
        Save the time spectrum to a csv file.
        
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
                self.data['Name'] = name
            elif self.name is not None:
                self.data['Name'] = self.name
            else:
                self.data['Name'] = filename
            self.data.to_csv(filename, sep='\t')

    def fourier_analysis(self, value='amplitude', data=None):
        """
        Fourier analysis of each spectral component over time.
        
        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to analyze.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        data : pandas DataFrame
            Data to compute the frequencies on.
            If None, self.data is used.
            Default is None.
        """
        if data is None:
            data = self.data
        self.frequencies = pd.DataFrame()
        for harm in data['harmonic'].unique():
            curdata = data[data['harmonic'] == harm]
            curdata = curdata.sort_values('Time')
            curfrequency = pd.DataFrame({'frequency': np.arange(int(len(curdata)/2)),
                                         value: np.abs(fft(np.array(curdata[value]))[:int(len(curdata)/2)])})
            curfrequency['harmonic'] = harm
            curfrequency['degree'] = curdata['degree'].iloc[0]
            self.frequencies = pd.concat([self.frequencies, curfrequency], ignore_index=True)

    def save_frequencies_to_csv(self, filename, name=None):
        """
        Save the results of the Fourier analysis to a csv file.
        
        Parameters
        ----------
        filename: str
            Path to the output file.
        name : str, optional
            Text to label the spectrum.
            If None, the value of `self.name` will be used.
        """
        if filename is not None:
            filelib.make_folders([os.path.dirname(filename)])
            if name is not None:
                self.frequencies['Name'] = name
            elif self.name is not None:
                self.frequencies['Name'] = self.name
            else:
                self.frequencies['Name'] = filename
            self.frequencies.to_csv(filename, sep='\t')

    def compute_derivative(self, value='amplitude', data=None):
        """
        Compute the derivative of each spectral component over time.

        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to analyze.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        data : pandas DataFrame
            Data to compute the derivative on.
            If None, self.data is used.
            Default is None.
        """
        if data is None:
            data = self.data
        times = data['Time'].unique()
        data = data.sort_values(['Time', 'harmonic'])
        prevdata = data[data['Time'] == times[0]]
        self.derivative = pd.DataFrame()
        for i in range(1, len(times)):
            curdata = data[data['Time'] == times[i]]
            if len(np.array(curdata[value])) == len(np.array(prevdata[value])):
                curderivative = pd.DataFrame({value: (np.array(curdata[value]) -
                                                           np.array(prevdata[value]))/(times[i] - times[i-1]),
                                              'harmonic': curdata['harmonic'],
                                              'degree': curdata['degree']})
                curderivative.loc[:, 'Time'] = times[i]
                self.derivative = pd.concat([self.derivative, curderivative], ignore_index=True)
                prevdata = curdata
            else:
                self.derivative = pd.DataFrame()
                break
        if len(self.derivative) > 0:
            self.derivative.loc[:, 'absolute ' + value] = np.abs(self.derivative[value])
            self.mean_abs_derivative = self.derivative.groupby('harmonic').mean().reset_index()

    def derivative_heatmap(self, title=None, **kwargs):
        """
        Plot the derivatives of spectral components over time as a heatmap.
        
        Parameters
        ----------
        title : str, optional
            Text to display in the plot title.
            Default is None.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.heatmap function.

        Returns
        -------
        seaborn.heatmap().figure
            The heatmap with the time points displayed vertically and spectral components horizontally.
        """
        stat = self.derivative
        value = 'amplitude'
        if len(stat) > 0 and len(stat[(stat['Time'] == stat['Time'].unique()[0])
                & (stat['harmonic'] == stat['harmonic'].unique()[0])]) == 1:
            hm = stat.pivot('Time', 'harmonic', value)
            plt.clf()
            plt.figure(figsize=(30, 5))
            pl = sns.heatmap(hm, **kwargs)
            if title is None:
                if self.name is not None:
                    title = self.name + '; ' + value + ' vs time'
                else:
                    title = 'value = ' + value + ' vs time'
            labels = stat['harmonic'].unique()
            plt.xticks(np.arange(len(labels)) + 0.6, labels, rotation='vertical')
            margins = {'left': 0.15, 'right': 0.97, 'top': 0.93, 'bottom': 0.25}
            plt.subplots_adjust(**margins)
            plt.title(title)
            return pl.figure

    def plot_mean_abs_derivative(self, title=None, **kwargs):
        """
        Plot the absolute value of the derivative averaged over time.
        
        Parameters
        ----------
        title : str, optional
            Text to display in the plot title.
            Default is None.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.barplot function.

        Returns
        -------
        seaborn.barplot().figure
            The barplot with mean absolute derivative plotted versus spectral components.
        """
        plt.clf()
        pl = sns.barplot('harmonic', 'absolute amplitude', data=self.mean_abs_derivative, **kwargs)
        plt.ylabel('Mean absolute derivative')
        labels = self.mean_abs_derivative['harmonic'].unique()
        plt.xticks(np.arange(len(labels)) + 0.6, labels, rotation='vertical')
        plt.title(title)
        return pl.figure

    def time_heatmap(self, value='amplitude', title=None, logscale=False, stat=None, **kwargs):
        """
        Plot the values of spectral components over time as a heatmap.
        
        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to analyze.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        title : str, optional
            Text to display in the plot title.
            Default is None.
        logscale : bool, optional
            If True, the natural logarithm of the value will be displayed.
            Default is False.
        stat : pandas DataFrame
            Data used to plot the heatmap.
            If None, self.data is used.
            Default is None.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.heatmap function.

        Returns
        -------
        seaborn.heatmap().figure
            The heatmap with the time points displayed vertically and spectral components horizontally.
        """
        if stat is None:
            stat = self.data
        if logscale:
            stat[value] = np.log(stat[value])
        if len(stat[(stat['Time'] == stat['Time'].unique()[0])
                & (stat['harmonic'] == stat['harmonic'].unique()[0])]) == 1:
            hm = stat.pivot('Time', 'harmonic', value)
            plt.clf()
            plt.figure(figsize=(30, 5))
            pl = sns.heatmap(hm, **kwargs)
            if title is None:
                if self.name is not None:
                    title = self.name + '; ' + value + ' vs time'
                else:
                    title = 'value = ' + value + ' vs time'
            labels = stat['harmonic'].unique()
            plt.xticks(np.arange(len(labels)) + 0.6, labels, rotation='vertical')
            margins = {'left': 0.03, 'right': 0.998, 'top': 0.93, 'bottom': 0.25}
            plt.subplots_adjust(**margins)
            plt.title(title)
            return pl.figure

    def frequency_heatmap(self, value='amplitude', title=None, logscale=False, **kwargs):
        """
        Plot the Fourier frequencies of spectral components as a heatmap.
        
        Parameters
        ----------
        value : str, optional
            Part of the complex spectrum to analyze.
            Valid values: 'amplitude', 'power', 'real', 'imag'.
            Default is 'amplitude'.
        title : str, optional
            Text to display in the plot title.
            Default is None.
        logscale : bool
            If True, the natural logarithm of the value will be displayed.
            Default is False.
        kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the seaborn.heatmap function.

        Returns
        -------
        seaborn.heatmap().figure
            The heatmap with the time frequencies displayed vertically and spectral components horizontally.
        """
        stat = self.frequencies
        stat.at[stat[stat[value] == 0].index, value] = 10**(-16)
        harm_unique = stat['harmonic'].unique()
        stat = stat[stat['harmonic'] != harm_unique[0]]
        if logscale:
            stat.loc[:, value] = np.log(stat[value])
        if len(stat[(stat['frequency'] == stat['frequency'].unique()[0])
                & (stat['harmonic'] == stat['harmonic'].unique()[0])]) == 1:
            hm = stat.pivot('frequency', 'harmonic', value)
            plt.clf()
            plt.figure(figsize=(30, 5))
            pl = sns.heatmap(hm, **kwargs)
            if title is None:
                if self.name is not None:
                    title = self.name + '; ' + value + ' vs frequency'
                else:
                    title = 'value = ' + value + ' vs frequency'
            labels = stat['harmonic'].unique()
            plt.xticks(np.arange(len(labels)) + 0.6, labels, rotation='vertical')
            margins = {'left': 0.15, 'right': 0.97, 'top': 0.93, 'bottom': 0.25}
            plt.subplots_adjust(**margins)
            plt.title(title)
            return pl.figure

    def return_feature_vector(self, cutoff=None, dynamic_features='frequency',
                              static_features='amplitude', rotation_invariant=True):
        """
        Return the amplitudes, derivatives and/or time frequencies for all harmonic components below a given degree .
        
        Parameters
        ----------
        cutoff : int, optional
            The number of frequency components to display.
            If None, all frequencies will be displayed.
        dynamic_features : str, optional
            Name of the feature to use for computing shape dynamics.
            Valid values: 'time', 'derivative', 'frequency'.
            Default is 'frequency'.
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
        stat = self.data
        if cutoff is None:
            cutoff = np.max(stat.degree)
        if static_features == 'real_imag':
            static_features = ['real', 'imag']
        else:
            static_features = [static_features]

        if rotation_invariant:
            stat = self.frequency_data

        features = []
        for value in static_features:
            if dynamic_features == 'time':
                features = features + list(stat[value][stat.degree < cutoff + 1])
            elif dynamic_features == 'derivative':
                self.compute_derivative(value=value, data=stat)
                features = features + list(self.derivative[value][self.derivative.degree < cutoff + 1])
            elif dynamic_features == 'frequency':
                self.fourier_analysis(value=value, data=stat)
                features = features + list(self.frequencies[value][self.frequencies.degree < cutoff + 1])

        return np.array(features)






