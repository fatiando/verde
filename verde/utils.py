"""
General utilities.
"""
import os

import numpy as np
import scipy.linalg as spla


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


def normalize_jacobian(jacobian):
    """
    """
    transform = 1/np.abs(jacobian).max(axis=0)
    # Element-wise multiplication with the diagonal of the scale matrix is the
    # same as A.dot(S)
    jacobian *= transform
    return jacobian, transform


def linear_fit(jacobian, data, weights=None, damping=None):
    """
    """
    if weights is None:
        weights = np.ones_like(data)
    hessian = jacobian.T.dot(weights.reshape((weights.size, 1))*jacobian)
    if damping is not None:
        hessian += damping*np.identity(jacobian.shape[1])
    gradient = jacobian.T.dot(weights*data)
    try:
        params = spla.solve(hessian, gradient, assume_a='pos')
    except spla.LinAlgError:
        raise spla.LinAlgError(' '.join([
            "Least-squares matrix is singular.",
            "Try increasing regularization or decreasing the number",
            "of model parameters."]))
    return params
