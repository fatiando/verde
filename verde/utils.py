"""
General utilities.
"""
import os
import numpy as np


def get_home():
    """
    Get the path of the verde home directory.

    Defaults to ``$HOME/.verde``.

    If the folder doesn't already exist, it will be created.

    Returns
    -------
    path : str
        The path of the home directory.

    """
    home = os.path.abspath(os.environ.get('HOME'))
    verde_home = os.path.join(home, '.verde')
    os.makedirs(verde_home, exist_ok=True)
    return verde_home


def get_data_dir():
    """
    Get the path of the verde data directory.

    Defaults to ``get_home()/data``.

    If the folder doesn't already exist, it will be created.

    Returns
    -------
    path : str
        The path of the data directory.

    """
    data_dir = os.path.join(get_home(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def variance_to_weights(variance, tol=1e-15):
    """
    Converts data variances to weights for gridding.

    Weights are defined as the inverse of the variance, scaled to the range
    [0, 1], i.e. ``variance.min()/variance``.

    Any variance that is smaller than *tol* will automatically receive a weight
    of 1 to avoid zero division or blown up weights.

    Parameters
    ----------
    variance : array
        An array with the variance of each point. Can have NaNs but they will
        be converted to zeros and therefore receive a weight of 1.
    tol : float
        The tolerance, or cutoff threshold, for small variances.

    Returns
    -------
    weights : array
        Data weights in the range [0, 1] with the same shape as *variance*.

    Examples
    --------

    >>> print(variance_to_weights([0, 2, 0.2, 1e-16]))
    [1.  0.1 1.  1. ]
    >>> print(variance_to_weights([0, 0, 0, 0]))
    [1 1 1 1]

    """
    variance = np.nan_to_num(np.atleast_1d(variance), copy=False)
    weights = np.ones_like(variance)
    nonzero = variance > tol
    if np.any(nonzero):
        nonzero_var = variance[nonzero]
        weights[nonzero] = nonzero_var.min()/nonzero_var
    return weights
