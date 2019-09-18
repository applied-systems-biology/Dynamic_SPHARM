from __future__ import division
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as plt

from SPHARM.classes.image_stack import ImageStack
from SPHARM.classes.surface import Surface
from SPHARM.classes.spectrum import Spectrum
from SPHARM.classes.time_spectrum import TimeSpectrum

from helper_lib import filelib


def plot_maxprojections(**kwargs):
    """
    Save a maximum projection of a given image.

    Keyword arguments
    -----------------
    *inputfolder* : str
        Directory with the input image.
    *item* : str
        File name of the input image.
    *outputfolder* : str
        Directory to save the maximum projection image.
    """
    inputfolder = kwargs.get('inputfolder')
    outputfolder = kwargs.get('outputfolder', inputfolder + '../output/maxprojection/')
    filename = kwargs.get('item')

    stack = ImageStack(inputfolder + filename)
    stack.save_max_projection(outputfolder + filename, axis=kwargs.get('axis'))


def plot_3D_surfaces(inputfolder, outputfolder, points=True, gridsize=100):
    """
    Plot 3D views of surfaces located in a given directory.

    Parameters
    ----------
    inputfolder : str
        Input directory with surfaces.
    outputfolder : str
        Output directory to save the plots.
    points : bool, optional
        If True, surface points will be displayed.
        Default is True.
    gridsize : int, optional
        Dimension of the square grid to interpolate the surface points.
        Default is 100.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])

    for fn in files:
        s = Surface(filename=inputfolder + fn)
        s.centrate()
        s.to_spherical()
        s.Rgrid = s.interpolate(grid_size=gridsize)
        mesh = s.plot_surface(points=points)
        mesh.magnification = 3
        filelib.make_folders([os.path.dirname(outputfolder + fn[:-4])])
        mesh.save(outputfolder + fn[:-4] + '.png', size=(200, 200))

################################################################################


def plot_individual_heatmaps(inputfolder, outputfolder, **kwargs):
    """
    Plot heatmaps of individual SPHARM spectra in a given directory.

    Parameters
    ----------
    inputfolder : str
        Input directory with spectra to plot.
    outputfolder : str
        Output directory to save the heatmaps.
    kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the Spectrum.heatmap function.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])

    for fn in files:
        s = Spectrum(filename=inputfolder + fn)
        pl = s.heatmap(title=fn[:-4], **kwargs)
        filelib.make_folders([os.path.dirname(outputfolder + fn[:-4])])
        pl.savefig(outputfolder + fn[:-4] + '.png')
        pl.clf()


def plot_spectra(inputfolder, outputfolder, **kwargs):
    """
    Plot bar plots for individual frequency spectra in a given directory. 

    Parameters
    ----------
    inputfolder : str
        Input directory with spectra to plot.
    outputfolder : str
        Output directory to save the bar plots.
    kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the Spectrum.frequency_plot function.
    """
    files = filelib.list_subfolders(inputfolder, extensions=['csv'])

    for fn in files:
        s = Spectrum(filename=inputfolder + fn)
        pl = s.frequency_plot(title=fn[:-4], **kwargs)
        filelib.make_folders([os.path.dirname(outputfolder + fn[:-4])])
        pl.savefig(outputfolder + fn[:-4] + '.png')
        pl.clf()

################################################################################


def plot_individual_time_heatmaps(inputfile, outputfolder, group='Group', cutoff=None,
                                  logscale=False, id_col='TrackID'):
    """
    Plot the amplitude of spectral components over time for different groups as a heatmap.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted heat maps.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    logscale : bool, optional
        If True, the natural logarithm of the value will be displayed.
        Default is False.
    id_col : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    """
    filelib.make_folders([os.path.dirname(outputfolder)])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
    if cutoff is not None:
        stat = stat[stat['degree'] <= cutoff]

    for gr in stat[group].unique():
        substat = stat[stat[group] == gr]
        for id in substat[id_col].unique():
            subsubstat = substat[substat[id_col] == id]
            subsubstat = subsubstat.sort_values('Time').reset_index()
            time_spectrum = TimeSpectrum()
            for t in subsubstat['Time'].unique():
                sp = Spectrum()
                sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                time_spectrum.add_spectrum(sp, timepoint=t)
            pl = time_spectrum.time_heatmap(value='amplitude', logscale=logscale)
            if pl is not None:
                pl.savefig(outputfolder + '_' + gr + '_' + 'track_' + str(id) + '.png')


def plot_individual_frequency_heatmaps(inputfile, outputfolder, group='Group', cutoff=None,
                                       logscale=False, id_col='TrackID'):
    """
    Plot the Fourier frequencies of spectral components of different groups as a heatmap.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted heat maps.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    logscale : bool, optional
        If True, the natural logarithm of the value will be displayed.
        Default is False.
    id_col : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    """
    filelib.make_folders([os.path.dirname(outputfolder)])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
    if cutoff is not None:
        stat = stat[stat['degree'] <= cutoff]
    for gr in stat[group].unique():
        substat = stat[stat[group] == gr]
        for id in substat[id_col].unique():
            subsubstat = substat[substat[id_col] == id]
            time_spectrum = TimeSpectrum()
            for t in subsubstat['Time'].unique():
                sp = Spectrum()
                sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                time_spectrum.add_spectrum(sp, timepoint=t)

            time_spectrum.fourier_analysis(value='amplitude')
            pl = time_spectrum.frequency_heatmap(value='amplitude', logscale=logscale)
            if pl is not None:
                pl.savefig(outputfolder + '_' + gr + '_' + 'track_' + str(id) + '.png')


def plot_individual_derivative_heatmaps(inputfile, outputfolder, group='Group', cutoff=None, id_col='TrackID'):
    """
    Plot the derivatives of spectral components of different groups over time as a heatmap.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted heat maps.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    id_col : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    """
    filelib.make_folders([os.path.dirname(outputfolder)])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
    if cutoff is not None:
        stat = stat[stat['degree'] <= cutoff]
    for gr in stat[group].unique():
        substat = stat[stat[group] == gr]
        for id in substat[id_col].unique():
            subsubstat = substat[substat[id_col] == id]
            time_spectrum = TimeSpectrum()
            for t in subsubstat['Time'].unique():
                sp = Spectrum()
                sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                time_spectrum.add_spectrum(sp, timepoint=t)

            time_spectrum.compute_derivative()
            pl = time_spectrum.derivative_heatmap()
            if pl is not None:
                pl.savefig(outputfolder + '_' + gr + '_' + 'track_' + str(id) + '.png')

################################################################################


def plot_average_heatmaps(inputfile, outputfolder, **kwargs):
    """
    Plot heatmaps for group-averaged SPHARM spectra.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.    
    outputfolder : str
        Output directory to save the heatmaps.
    kwargs : key, value pairings
            Arbitrary keyword arguments to pass to the Spectrum.heatmap function.
    """
    filelib.make_folders([os.path.dirname(outputfolder), outputfolder + 'timepoints/'])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat.loc[:, 'Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
    if 'Group' not in stat.columns:
        for name in stat['Name'].unique():
            group = name.split('/')[0]
            stat = stat.set_value(stat[stat['Name'] == name].index, 'Group', group)

    data = stat.groupby(['degree', 'order', 'Group']).mean().reset_index()
    for gr in data['Group'].unique():
        curdata = data[data['Group'] == gr]
        s = Spectrum()
        s.harmonics_csv = curdata
        pl = s.heatmap(title=gr + ' average', **kwargs)
        pl.savefig(outputfolder + gr + '.png')
        pl.clf()

    # plot separate time points
    stat = stat.groupby(['Time', 'Group', 'degree', 'order']).mean().reset_index()

    for t in stat['Time'].unique():
        for gr in stat['Group'].unique():
            curdata = stat[(stat['Group'] == gr) & (stat['Time'] == t)]
            if len(curdata) > 0:
                s = Spectrum()
                s.harmonics_csv = curdata
                pl = s.heatmap(title=gr + ' average, time point ' + str(t), **kwargs)
                pl.savefig(outputfolder + 'timepoints/' + gr + '_time=' + str(t) + '.png')
                pl.clf()


def plot_average_spectra(inputfile, outputfolder, value='amplitude', group='Group', norm=False, cutoff=None):
    """
    Plot frequency spectra of spherical components.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted spectra.
    value : str, optional
        Part of the complex spectrum to plot.
        Valid values: 'amplitude', 'power', 'real', 'imag'.
        Default is 'amplitude'.
    norm : bool, optional
            If True, each component of the frequency spectrum will be divided by the value of the zero frequency.
            Default is False.        
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    """

    filelib.make_folders([os.path.dirname(outputfolder), outputfolder + 'timepoints/'])

    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    if norm:
        stat.loc[:, value] = np.array(stat[value]) / stat[value].iloc[0]
    if cutoff:
        stat = stat[stat.degree < cutoff]

    stat.loc[:, 'Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10

    sns.catplot(x='degree', y=value, data=stat, kind='bar', hue=group, estimator=np.nanmedian, ci=90)

    plt.savefig(outputfolder + '_All_time_points.png')
    plt.close()

    # plot separate time points

    times = stat.Time.unique()
    times.sort()

    for t in times:
        curstat = stat[stat.Time == t]
        sns.catplot(x='degree', y=value, data=curstat, kind='bar', hue=group, estimator=np.nanmedian, ci=90)

        plt.savefig(outputfolder + 'timepoints/time=' + str(t) + 's.png')
        plt.close()


def plot_average_frequency_heatmaps(inputfile, outputfolder, group='Group', cutoff=None,
                                    logscale=False, id_col='TrackID'):
    """
    Plot the Fourier frequencies of spectral components of different groups as a heatmap.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted heat maps.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    logscale : bool, optional
        If True, the natural logarithm of the value will be displayed.
        Default is False.
    id_col : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    """
    filelib.make_folders([os.path.dirname(outputfolder)])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat.loc[:, 'Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
    if cutoff is not None:
        stat = stat[stat['degree'] <= cutoff]
    frequency_stat = pd.DataFrame()
    for gr in stat[group].unique():
        substat = stat[stat[group] == gr]
        for id in substat[id_col].unique():
            subsubstat = substat[substat[id_col] == id]
            time_spectrum = TimeSpectrum()
            for t in subsubstat['Time'].unique():
                sp = Spectrum()
                sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                time_spectrum.add_spectrum(sp, timepoint=t)

            time_spectrum.fourier_analysis(value='amplitude')
            time_spectrum.frequencies['Group'] = gr
            frequency_stat = pd.concat([frequency_stat, time_spectrum.frequencies], ignore_index=True)

    frequency_stat = frequency_stat.groupby(['Group', 'frequency', 'harmonic']).mean().reset_index()
    for gr in stat[group].unique():
        time_spectrum = TimeSpectrum()
        time_spectrum.frequencies = frequency_stat[frequency_stat['Group'] == gr]

        pl = time_spectrum.frequency_heatmap(value='amplitude', logscale=logscale)
        if pl is not None:
            pl.savefig(outputfolder + gr + '.png')


def plot_mean_abs_derivative(inputfile, outputfolder, group='Group', cutoff=None, id_col='TrackID'):
    """
    Plot the derivative of each spectral component of different groups over time.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted derivative.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    id_col : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    """
    filelib.make_folders([os.path.dirname(outputfolder)])
    if not os.path.exists(inputfile[:-4] + '_mean_abs_derivative.csv'):
        stat = pd.read_csv(inputfile, sep='\t', index_col=0)
        if id_col == 'CellID':
            stat.loc[:, 'Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10
        nstat = pd.DataFrame()
        if cutoff is not None:
            stat = stat[stat['degree'] <= cutoff]
        for gr in stat[group].unique():
            substat = stat[stat[group] == gr]
            for id in substat[id_col].unique():
                subsubstat = substat[substat[id_col] == id]
                subsubstat = subsubstat.sort_values('Time')
                time_spectrum = TimeSpectrum()
                for t in subsubstat['Time'].unique():
                    sp = Spectrum()
                    sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                    time_spectrum.add_spectrum(sp, timepoint=t)

                time_spectrum.compute_derivative()
                meanderivstat = time_spectrum.mean_abs_derivative
                meanderivstat['Group'] = gr
                meanderivstat['TrackID'] = id
                nstat = pd.concat([nstat, meanderivstat], ignore_index=True)

        nstat.to_csv(inputfile[:-4] + '_mean_abs_derivative.csv', sep='\t')
    nstat = pd.read_csv(inputfile[:-4] + '_mean_abs_derivative.csv', sep='\t', index_col=0)
    nstat = nstat.sort_values(['harmonic', group])
    plt.clf()
    plt.figure(figsize=(20, 5))
    sns.barplot(x='harmonic', y='absolute amplitude', data=nstat, hue=group)
    plt.ylabel('Mean absolute derivative of amplitude')
    labels = nstat['harmonic'].unique()
    plt.xticks(np.arange(len(labels)) + 0.6, labels, rotation='vertical')
    margins = {'left': 0.07, 'right': 0.98, 'top': 0.93, 'bottom': 0.25}
    plt.subplots_adjust(**margins)
    plt.savefig(outputfolder + 'mean_abs_derivative.png')
    plt.close()


################################################################################

def plot_heatmap_difference(inputfile, outputfolder, value='amplitude', cutoff=None, group='Group'):
    """
    Plot the pairwise difference of group-averaged heat maps of spherical components.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted heat maps.
    value : str, optional
        Part of the complex spectrum to plot.
        Valid values: 'amplitude', 'power', 'real', 'imag'.
        Default is 'amplitude'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    """

    filelib.make_folders([os.path.dirname(outputfolder), outputfolder + 'timepoints/'])

    stat = pd.read_csv(inputfile, sep='\t', index_col=0)

    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10

    if cutoff:
        stat = stat[stat.degree < cutoff]

    times = stat.Time.unique()
    times.sort()
    curstat = stat.groupby([group, 'degree', 'order']).mean().reset_index()
    groups = curstat[group].unique()
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            grstat1 = stat[stat[group] == groups[i]].groupby(['degree', 'order']).mean().reset_index()
            grstat2 = stat[stat[group] == groups[j]].groupby(['degree', 'order']).mean().reset_index()
            hm1 = grstat1.pivot('degree', 'order', value)
            hm2 = grstat2.pivot('degree', 'order', value)
            hm = hm1 - hm2
            plt.clf()
            plt.figure(figsize=(6, 5))
            sns.heatmap(hm, vmin=-0.1, vmax=0.1)
            plt.title(groups[i] + ' - ' + groups[j])
            plt.savefig(outputfolder + '_' + groups[i] + ' - ' + groups[j] + '_all_time_points.png')
            plt.close()

            plt.clf()
            hm = (hm2 - hm1) * 2 / (hm2 + hm1)
            sns.heatmap(hm)
            plt.title(groups[i] + ' - ' + groups[j] + ' (normalized)')
            plt.savefig(outputfolder + '_' + groups[i] + ' - ' + groups[j] + '_normalized_all_time_points.png')
            plt.close()

            # plot separate time points

            for t in times:
                grstat1 = stat[(stat[group] == groups[i])
                               & (stat.Time == t)].groupby(['degree', 'order']).mean().reset_index()
                grstat2 = stat[(stat[group] == groups[j])
                               & (stat.Time == t)].groupby(['degree', 'order']).mean().reset_index()
                if len(grstat1) > 0 and len(grstat2) > 0:
                    hm1 = grstat1.pivot('degree', 'order', value)
                    hm2 = grstat2.pivot('degree', 'order', value)
                    hm = hm1 - hm2
                    sns.heatmap(hm)
                    plt.title(groups[i] + ' - ' + groups[j])
                    plt.savefig(outputfolder + 'timepoints/' + groups[i] + ' - ' + groups[j]
                                + '_time=' + str(t) + '.png')
                    plt.close()

                    hm = (hm2-hm1)*2/(hm2+hm1)
                    sns.heatmap(hm)
                    plt.title(groups[i] + ' - ' + groups[j] + ' (normalized)')
                    plt.savefig(outputfolder + 'timepoints/' + groups[i] + ' - ' +
                                groups[j] + '_normalized_time=' + str(t) + '.png')
                    plt.close()


def cohens_d(x, y):
    """
    Compute the effect size by Cohen's d.

    Parameters
    ----------
    x : numpy array or list
        data set 1
    y : numpy array or list
        data set 2

    Returns
    -------
    scalar : the computed effect size.

    """
    sx = np.var(x) * len(x)
    sy = np.var(y) * len(y)

    s = np.sqrt((sx + sy) / (len(x) + len(y) - 2))
    d = (np.mean(x) - np.mean(y)) / s
    return d


def plot_effect_size(inputfile, outputfolder, value='amplitude', group='Group', cutoff=None):
    """
    Plot effect size between spectral frequency components of different groups.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted effect size.
    value : str, optional
        Part of the complex spectrum to plot.
        Valid values: 'amplitude', 'power', 'real', 'imag'.
        Default is 'amplitude'.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    """
    filelib.make_folders([os.path.dirname(outputfolder), outputfolder + 'timepoints/'])

    stat = pd.read_csv(inputfile, sep='\t', index_col=0)

    if cutoff:
        stat = stat[stat.degree < cutoff]
    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10

    # plot combined Time
    groups = stat[group].unique()
    ngr = len(groups) - 1
    if ngr > 0:
        times = stat.Time.unique()
        times.sort()

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                effstat = pd.DataFrame()
                grstat1 = stat[stat[group] == groups[i]]
                grstat2 = stat[stat[group] == groups[j]]
                for d in grstat1['degree'].unique():
                    if len(grstat1[grstat1['degree'] == d]) > 2 and len(grstat2[grstat2['degree'] == d]) > 2:
                        effect = cohens_d(np.array(grstat1[grstat1['degree'] == d][value]),
                                          np.array(grstat2[grstat2['degree'] == d][value]))
                        effstat = effstat.append(pd.Series({'effect size of ' + value: effect, 'degree': d}),
                                                 ignore_index=True)
                sns.barplot(x='degree', y='effect size of ' + value, data=effstat)
                plt.title(groups[i] + ' vs ' + groups[j])
                plt.ylim(-3.5, 3.5)
                plt.savefig(outputfolder + groups[i] + ' vs ' + groups[j] + '_all_time_points.png')
                plt.close()

                # plot separate time points

                for t in times:
                    effstat = pd.DataFrame()
                    grstat1 = stat[(stat[group] == groups[i]) & (stat.Time == t)]
                    grstat2 = stat[(stat[group] == groups[j]) & (stat.Time == t)]
                    for d in grstat1['degree'].unique():
                        if len(grstat1[grstat1['degree'] == d]) > 2 and len(grstat2[grstat2['degree'] == d]) > 2:
                            effect = cohens_d(np.array(grstat1[grstat1['degree'] == d][value]),
                                              np.array(grstat2[grstat2['degree'] == d][value]))
                            effstat = effstat.append(pd.Series({'effect size of ' + value: effect, 'degree': d}),
                                                     ignore_index=True)

                    if len(effstat) > 0:
                        sns.barplot(x='degree', y='effect size of ' + value, data=effstat)
                        plt.title(groups[i] + ' vs ' + groups[j])
                        plt.ylim(-3.5, 3.5)
                        plt.savefig(outputfolder + 'timepoints/' + groups[i] + ' vs '
                                    + groups[j] + '_time=' + str(t) + '.png')
                        plt.close()


def plot_pairplots(inputfile, outputfolder, group='Group', cutoff=None):
    """
    Plot pairwise distributions of different spherical components in different groups.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted distributions.
    cutoff : int, optional
        The number of degrees to display.
        If None, all degrees will be displayed.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    """
    filelib.make_folders([os.path.dirname(outputfolder), outputfolder + 'timepoints/'])

    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    if cutoff:
        stat = stat[stat.degree < cutoff]
    else:
        stat = stat[stat['degree'] < 7]

    stat['Time'] = np.int_(np.round_(stat['Time'] / 10.)) * 10

    nstat = pd.DataFrame()

    times = stat.Time.unique()
    times.sort()
    for t in times:
        curstat = stat[stat.Time == t]

        pivoted = curstat.pivot(index='Name', columns='degree', values='amplitude')
        pivoted['Name'] = pivoted.index

        for i in pivoted['Name'].unique():
            pivoted.at[pivoted[pivoted['Name'] == i].index, group] = stat[stat['Name'] == i][group].iloc[0]

        pivoted = pivoted.drop('Name', 1)
        if len(pivoted) > 0:
            f = sns.pairplot(pivoted, hue=group)
            f.savefig(outputfolder + 'timepoints/time=' + str(t) + 's.png')
            plt.close()

        nstat = pd.concat([nstat, pivoted], ignore_index=True)

    if len(nstat) > 0:
        f = sns.pairplot(nstat, hue=group)
        f.savefig(outputfolder + 'all_time_points.png')
        plt.close()


def plot_inverse_shapes(inputfile, outputfolder, group='Group'):
    """
    Plot average cells shapes obtained by inverse SPHARM.

    Parameters
    ----------
    inputfile : str
        Path to the file with spectral data.
    outputfolder : str
        Directory to save the plotted distributions.
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    """

    filelib.make_folders([os.path.dirname(outputfolder)])
    stat = pd.read_csv(inputfile, sep='\t', index_col=0)
    stat['value'] = stat['real'] + stat['imag']*1j
    if 'Group' not in stat.columns:
        for name in stat['Name'].unique():
            group = name.split('/')[0]
            stat = stat.set_value(stat[stat['Name'] == name].index, 'Group', group)

    data = stat.groupby(['degree', 'order', 'Group']).mean().reset_index()
    groups = data[group].unique()
    for gr in groups:
        curdata = data[data[group] == gr]
        sp = Spectrum()
        sp.harmonics_csv = curdata
        sp.convert_to_shtools_array()
        surf = Surface()
        surf.spharm = sp
        maxdegree = np.max(sp.harmonics_csv['degree'])
        for lmax in np.arange(5, maxdegree + 1, 5):
            surf.inverse_spharm(lmax=lmax)
            surf.plot_surface(points=False).save(outputfolder + '_' + gr + '_inverse_lmax=' + str(lmax) + '.png',
                                                 size=(200, 200))

        surf.inverse_spharm(lmax=None)
        surf.plot_surface(points=False).save(outputfolder + '_' + gr + '_inverse_full.png',
                                             size=(200, 200))




