"""
General utilities.
"""
import functools
import itertools

import numpy as np
import pandas as pd


def parse_engine(engine):
    """
    Choose the best engine available and check if it's valid.

    Parameters
    ----------
    engine : str
        The name of the engine. If ``"auto"`` will favor numba if it's available.

    Returns
    -------
    engine : str
        The name of the engine that should be used.

    """
    engines = {"auto", "numba", "numpy"}
    if engine not in engines:
        raise ValueError("Invalid engine '{}'. Must be in {}.".format(engine, engines))
    if engine == "auto":
        try:
            import numba  # pylint: disable=unused-variable

            return "numba"
        except ImportError:
            return "numpy"
    return engine


def dummy_jit(**kwargs):  # pylint: disable=unused-argument
    """
    Replace numba.jit if not installed with a function that raises RunTimeError.

    Use as a decorator.

    Parameters
    ----------
    function
        A function that you would decorate with :func:`numba.jit`.

    Returns
    -------
    function
        A function that raises :class:`RunTimeError` warning that numba isn't installed.

    """

    def dummy_decorator(function):
        "The actual decorator"

        @functools.wraps(function)
        def dummy_function(*args, **kwargs):  # pylint: disable=unused-argument
            "Just raise an exception."
            raise RuntimeError("Could not find numba.")

        return dummy_function

    return dummy_decorator


def n_1d_arrays(arrays, n):
    """
    Get the first n elements from a tuple/list, make sure they are arrays, and ravel.

    Parameters
    ----------
    arrays : tuple of arrays
        The arrays. Can be lists or anything that can be converted to a numpy array
        (including numpy arrays).
    n : int
        How many arrays to return.

    Returns
    -------
    1darrays : tuple of arrays
        The converted 1D numpy arrays.

    Examples
    --------

    >>> import numpy as np
    >>> arrays = [np.arange(4).reshape(2, 2)]*3
    >>> n_1d_arrays(arrays, n=2)
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """
    return tuple(np.atleast_1d(i).ravel() for i in arrays[:n])


def check_data(data):
    """
    Check the *data* argument and make sure it's a tuple.
    If the data is a single array, return it as a tuple with a single element.

    This is the default format accepted and used by all gridders and processing
    functions.

    Examples
    --------

    >>> check_data([1, 2, 3])
    ([1, 2, 3],)
    >>> check_data(([1, 2], [3, 4]))
    ([1, 2], [3, 4])
    """
    if not isinstance(data, tuple):
        data = (data,)
    return data


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


def maxabs(*args):
    """
    Calculate the maximum absolute value of the given array(s).

    Use this to set the limits of your colorbars and center them on zero.

    Parameters
    ----------
    args
        One or more arrays. If more than one are given, a single maximum will be
        calculated across all arrays.

    Returns
    -------
    maxabs : float
        The maximum absolute value across all arrays.

    Examples
    --------

    >>> maxabs((1, -10, 25, 2, 3))
    25
    >>> maxabs((1, -10.5, 25, 2), (0.1, 100, -500), (-200, -300, -0.1, -499))
    500.0

    """
    arrays = [np.atleast_1d(i) for i in args]
    absolute = [np.abs([i.min(), i.max()]).max() for i in arrays]
    return np.max(absolute)


def grid_to_table(grid):
    """
    Converts an xarray Dataset grid to a Pandas DataFrame

    Parameters
    ----------
    grid
        An xarray grid Dataset composed of X and Y coordinates and a variable.

    Returns
    -------
    grid_to_table : DataFrame
        Pandas DataFrame with coordinates and variables and increasing index

    Examples
    --------


    """
    coordinate_names = [
        *grid.coords.keys()
    ]  # extracts the coordinate names from the dataarray
    variable_name = [
        *grid.data_vars.keys()
    ]  # extracts the variable name from the dataarray
    coord_one = grid.coords[
        coordinate_names[0]
    ].values  # gets the first set of coordinates as a list
    coord_two = grid.coords[
        coordinate_names[1]
    ].values  # gets the second set of coordinates as a list
    locations = np.asarray(
        list(itertools.product(coord_one, coord_two))
    )  # creates a numpy array from the coordinates
    variable = grid[variable_name[0]].values.flatten(
        order="F"
    )  # extracts the variable values for each location
    data_dict = {
        coordinate_names[0]: locations[:, 0],
        coordinate_names[1]: locations[:, 1],
        variable_name[0]: variable,
    }  # dict of coordinates and variables to pass to pandas
    data = pd.DataFrame(data_dict)
    return data
