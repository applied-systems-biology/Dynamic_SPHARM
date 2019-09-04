from __future__ import division

import numpy as np


def cart_to_spherical(x, y, z):
    """
    Convert coordinates from Cartesian to spherical.
    
    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    z : float
        z coordiante.

    Returns
    -------
    r : float
        Radius.
    theta : float
        Polar angle.
    phi : float
        Azimuthal angle.

    """

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.where(r == 0, 0.0001, r)
    theta = np.arccos(z*1./r)
    phi = np.arctan2(y, x) + np.pi
    return r, theta, phi


def spherical_to_cart(r, theta, phi):
    """
    Convert coordinates from spherical to Cartesian.
    
    Parameters
    ----------
    r : float
        Radius.
    theta : float
        Polar angle.
    phi : float
        Azimuthal angle.

    Returns
    -------
    x : float
        x coordinate.
    y : float
        y coordinate.
    z : float
        z coordiante.
    """
    x = r*np.sin(theta)*np.cos(phi - np.pi)
    y = r*np.sin(theta)*np.sin(phi - np.pi)
    z = r*np.cos(theta)

    return x, y, z


def cart_to_polar(x, y):
    """
    Convert coordiantes from Cartesian to polar.
    
    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.

    Returns
    -------
    r : float
        Radius.
    phi : float
        Azimuthal angle.
    """
    x = x - x.mean()
    y = y - y.mean()

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) + np.pi
    return r, phi


def rotate_spherical(x, y, z, theta, phi):
    """
    Rotate the given coordinate by given asimuthal and polar angles.

    Parameters
    ----------
    x : scalar or array
        x coordinate(s).
    y : scalar or array
        y coordinate(s).
    z : scalar or array
        z coordinate(s).
    theta : float
        Polar angle for rotation.
    phi : float
        Azimuthal angle for rotation.

    Returns
    -------
    x : scalar or array
        rotated x coordinate(s).
    y : scalar or array
        rotated y coordinate(s).
    z : scalar or array
        rotated z coordinate(s).
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    x1 = np.cos(phi) * x - np.sin(phi) * y
    y1 = np.sin(phi) * x + np.cos(phi) * y
    z1 = z

    x2 = np.cos(theta) * x1 + np.sin(theta) * z1
    y2 = y1
    z2 = - np.sin(theta) * x1 + np.cos(theta) * z1

    return x2, y2, z2


def as_stack(x, y, z, minmax=None):
    """
    Generate a binary image with given foreground coordinates.
    
    Parameters
    ----------
    x : list or ndarray
        x coordinates.
    y : list or ndarray
        y coordinates.
    z : list or ndarray
        z coordiantes.
    minmax : ndarray, optional
        Boundaries to crop the image stack of the form [[z_min, z_max], [y_min, y_max], [x_min, x_max]].
        If None, set to the minimal and maximal given coordinates.
        Default is None.
        
    Returns
    -------
    ndimage : 3D binary image with surface point as foreground.
    """
    if minmax is None:
        minmax = np.int_([[z.min(), z.max()],
                          [y.min(), y.max()],
                          [x.min(), x.max()]])
    else:
        minmax = np.int_(np.round_(minmax))

    x = np.int_(x) - minmax[2, 0] + 1
    y = np.int_(y) - minmax[1, 0] + 1
    z = np.int_(z) - minmax[0, 0] + 1

    img = np.zeros([minmax[0, 1] - minmax[0, 0] + 3, minmax[1, 1] - minmax[1, 0] + 3, minmax[2, 1] - minmax[2, 0] + 3])
    img[z, y, x] = 255

    return img



