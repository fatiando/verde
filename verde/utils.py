"""
General utilities.
"""
import os
import numpy as np

from .base import check_data


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
    home = os.path.abspath(os.environ.get("HOME"))
    verde_home = os.path.join(home, ".verde")
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
    data_dir = os.path.join(get_home(), "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def variance_to_weights(variance, tol=1e-15, dtype="float64"):
    """
    Converts data variances to weights for gridding.

    Weights are defined as the inverse of the variance, scaled to the range
    [0, 1], i.e. ``variance.min()/variance``.

    Any variance that is smaller than *tol* will automatically receive a weight
    of 1 to avoid zero division or blown up weights.

    Parameters
    ----------
    variance : array or tuple of arrays
        An array with the variance of each point. If there are multiple arrays
        in a tuple, will calculated weights for each of them separately. Can
        have NaNs but they will be converted to zeros and therefore receive a
        weight of 1.
    tol : float
        The tolerance, or cutoff threshold, for small variances.
    dtype : str or numpy dtype
        The type of the output weights array.

    Returns
    -------
    weights : array or tuple of arrays
        Data weights in the range [0, 1] with the same shape as *variance*. If
        more than one variance array was provided, then this will be a tuple
        with the weights corresponding to each variance array.

    Examples
    --------

    >>> print(variance_to_weights([0, 2, 0.2, 1e-16]))
    [1.  0.1 1.  1. ]
    >>> print(variance_to_weights([0, 0, 0, 0]))
    [1. 1. 1. 1.]
    >>> for w  in variance_to_weights(([0, 1, 10], [2, 4.0, 8])):
    ...     print(w)
    [1.  1.  0.1]
    [1.   0.5  0.25]

    """
    variance = check_data(variance)
    weights = []
    for var in variance:
        var = np.nan_to_num(np.atleast_1d(var), copy=False)
        w = np.ones_like(var, dtype=dtype)
        nonzero = var > tol
        if np.any(nonzero):
            nonzero_var = var[nonzero]
            w[nonzero] = nonzero_var.min() / nonzero_var
        weights.append(w)
    if len(weights) == 1:
        return weights[0]
    return tuple(weights)
