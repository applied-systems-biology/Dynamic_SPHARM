from __future__ import division
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut

from SPHARM.classes.spectrum import Spectrum
from SPHARM.classes.time_spectrum import TimeSpectrum
from SPHARM.classes.stratified_group_shuffle_split import GroupShuffleSplitStratified


def extract_features(input_stat, cell_id='Name', group='Group', static=True, dynamic_features=None,
                     timelength=10, static_features='amplitude', one_time_point=True, rotation_invariant=True):
    """
    Extract spectral features for classification.

    Parameters
    ----------
    input_stat : pandas.DataFrame
        Input data to extract features
    cell_id : str
        Column in the input data sheet to group connected time points.
        Default is 'TrackID'
    group : str, optional
        Column in the input data sheet to use for grouping.
        Default is 'Group'.
    static : bool, optional
        If True, static features will be extracted.
        If False, dynamic features will be extracted.
        Default is True.
    dynamic_features : str, optional
            Name of the feature to use for computing shape dynamics.
            Valid values: 'time', 'derivative', 'frequency'.
            Default is 'frequency'.
    static_features : str, optional
        Name of the feature to represent the harmonic coefficients.
        Valid values: 'amplitude', 'real_imag'.
        Default is 'amplitude'
    timelength : int, optional
        Number of time points to include into dynamic features.
        Default is 10.
    one_time_point : bool, optional
        If True, only the first time point of each cell will be used.
        If False, all time points will be used as independent samples.
        Default is True.
    rotation_invariant : bool, optional
        If True, rotation-invariant descriptors (frequencies) will be computed.
        If False, the whole spectrum will be used as a feature vector.
        Default is True.

    Returns
    -------
    features : N x K numpy.array
        The returned feature vector of N samples and K features
    classes : array of length N
        Labels of the true classes.
    """

    features = []
    classes = []
    names = []
    group_names = []
    samples = []

    groups = input_stat[group].unique()
    for i in range(len(groups)):
        stat = input_stat[input_stat[group] == groups[i]]
        if len(stat) > 0:
            for name in stat[cell_id].unique():
                subsubstat = stat[stat[cell_id] == name].reset_index()
                subsubstat = subsubstat.sort_values('Time')
                times = subsubstat['Time'].unique()
                if static:
                    spectrum = Spectrum()
                    if one_time_point:
                        spectrum.harmonics_csv = subsubstat[subsubstat['Time'] == times[0]]
                        features.append(spectrum.return_feature_vector(static_features=static_features,
                                                                       rotation_invariant=rotation_invariant))
                        classes.append(i)
                        names.append(spectrum.harmonics_csv.iloc[0]['Name'])
                        group_names.append(groups[i])
                        if 'Sample' in subsubstat.columns:
                            samples.append(subsubstat['Sample'].iloc[0])
                    else:
                        for t in times:
                            spectrum.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                            features.append(spectrum.return_feature_vector(static_features=static_features,
                                                                           rotation_invariant=rotation_invariant))
                            classes.append(i)
                            names.append(spectrum.harmonics_csv.iloc[0]['Name'])
                            group_names.append(groups[i])
                            if 'Sample' in subsubstat.columns:
                                samples.append(subsubstat['Sample'].iloc[0])
                else:
                    if len(times) >= timelength:
                        spectrum = TimeSpectrum()
                        for t in times[:timelength]:
                            sp = Spectrum()
                            sp.harmonics_csv = subsubstat[subsubstat['Time'] == t]
                            spectrum.add_spectrum(sp, timepoint=t)
                        features.append(spectrum.return_feature_vector(dynamic_features=dynamic_features,
                                                                       static_features=static_features,
                                                                       rotation_invariant=rotation_invariant))
                        classes.append(i)
                        group_names.append(groups[i])
                        names.append(spectrum.data.iloc[0]['Name'])
                        if 'Sample' in subsubstat.columns:
                            samples.append(subsubstat['Sample'].iloc[0])
    classes = np.array(classes)
    features = np.array(features)
    names = np.array(names)
    group_names = np.array(group_names)
    samples = np.array(samples)

    return features, classes, names, group_names, samples


def predict_classes_loo(features, classes, C=1, groups=None):
    """
    Computes accuracy and predicts classes by cross-validation
    Parameters
    ----------
    features : array-like
        The data to fit.
    classes : array-like
        The target classes to try to predict.
    C : float, optional
        Penalty parameter C of the error term.
        Default is 1.
    groups : array-like, optional
        Group labels for the samples used while splitting the dataset into train/test set.
        Default is None.

    Returns
    -------
    accuracy: pandas DataFrame
        Array of scores of the estimator for each run of the cross validation.
    predicted: pandas DataFrame
        Predicted classes
    """
    clf = svm.SVC(kernel='linear', C=C, cache_size=1000, decision_function_shape='ovo', random_state=0)
    cv = LeaveOneGroupOut()
    accuracy = pd.DataFrame({'Accuracy': cross_val_score(clf, X=features, y=classes, groups=groups, cv=cv)})
    predicted = pd.DataFrame({'Actual class': classes,
                              'Predicted class': cross_val_predict(clf, X=features,
                                                                   y=classes, groups=groups, cv=cv)})

    return accuracy, predicted


def predict_classes_st_kfold(features, classes, nsplits=7, random_state=0, C=1):
    """
    Computes accuracy and predicts classes by cross-validation
    Parameters
    ----------
    features : array-like
        The data to fit.
    classes : array-like
        The target classes to try to predict.
    nsplits : int, optional
        Number of folds.
        Default is 7.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    C : float, optional
        Penalty parameter C of the error term.
        Default is 1.

    Returns
    -------
    accuracy: pandas DataFrame
        Array of scores of the estimator for each run of the cross validation.
    predicted: pandas DataFrame
        Predicted classes
    """

    clf = svm.SVC(kernel='linear', C=C, cache_size=1000, decision_function_shape='ovo', random_state=0)

    cv = StratifiedKFold(n_splits=nsplits, random_state=random_state)
    accuracy = pd.DataFrame({'Accuracy': cross_val_score(clf, X=features, y=classes,cv=cv)})
    predicted = pd.DataFrame({'Actual class': classes,
                              'Predicted class': cross_val_predict(clf, X=features, y=classes, cv=cv)})

    return accuracy, predicted


def predict_shuffle_split(features, classes, C=1, nsplits=100, test_size=2./5, random_state=0):
    """
    Computes accuracy and predicts classes by cross-validation
    Parameters
    ----------
    features : array-like
        The data to fit.
    classes : array-like
        The target classes to try to predict.
    C : float, optional
        Penalty parameter C of the error term.
        Default is 1.
    nsplits : int, optional
        Number of folds.
        Default is 100.
    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
        Default is 2./5
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.

    Returns
    -------
    accuracy: pandas DataFrame
        Array of scores of the estimator for each run of the cross validation.
    """

    clf = svm.SVC(kernel='linear', C=C, cache_size=1000, decision_function_shape='ovo', random_state=0)
    cv = StratifiedShuffleSplit(n_splits=nsplits, test_size=test_size, random_state=random_state)
    accuracy = pd.DataFrame({'Accuracy': cross_val_score(clf, features, classes, cv=cv)})
    return accuracy


def predict_group_shuffle_split(features, classes, C=1, nsplits=100, test_size=1, random_state=0, groups=None):
    """
    Computes accuracy and predicts classes by cross-validation
    Parameters
    ----------
    features : array-like
        The data to fit.
    classes : array-like
        The target classes to try to predict.
    C : float, optional
        Penalty parameter C of the error term.
        Default is 1.
    nsplits : int, optional
        Number of folds.
        Default is 100.
    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
        Default is 1
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    groups : array-like, optional
        Group labels for the samples used while splitting the dataset into train/test set.
        Default is None.

    Returns
    -------
    accuracy: pandas DataFrame
        Array of scores of the estimator for each run of the cross validation.
    """

    clf = svm.SVC(kernel='linear', C=C, cache_size=1000, decision_function_shape='ovo', random_state=0)
    cv = GroupShuffleSplitStratified(n_splits=nsplits, test_size=test_size, random_state=random_state)
    accuracy = pd.DataFrame({'Accuracy': cross_val_score(clf, X=features, y=classes, groups=groups, cv=cv)})
    return accuracy















