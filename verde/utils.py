"""
General utilities.
"""
import functools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

try:
    from pykdtree.kdtree import KDTree as pyKDTree
except ImportError:
    pyKDTree = None

try:
    import numba
except ImportError:
    numba = None

from .base.utils import check_data, n_1d_arrays


class DummyClient:  # pylint: disable=no-self-use,too-few-public-methods
    """
    Dummy client to mimic a dask.distributed.Client for immediate local
    execution.

    >>> client = DummyClient()
    >>> client.submit(sum, (1, 2, 3))
    6
    >>> print(client.scatter("bla"))
    bla

    """

    def submit(self, function, *args, **kwargs):
        "Execute function with the given arguments and return its output"
        return function(*args, **kwargs)

    def scatter(self, value):
        "Does nothing but return the input"
        return value


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
        if numba is None:
            return "numpy"
        return "numba"
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


def maxabs(*args, nan=True):
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

    If the array contains NaNs, we'll use the ``nan`` version of of the numpy functions
    by default. You can turn this off through the *nan* argument.

    >>> import numpy as np
    >>> maxabs((1, -10, 25, 2, 3, np.nan))
    25.0
    >>> maxabs((1, -10, 25, 2, 3, np.nan), nan=False)
    nan

    """
    arrays = [np.atleast_1d(i) for i in args]
    if nan:
        npmin, npmax = np.nanmin, np.nanmax
    else:
        npmin, npmax = np.min, np.max
    absolute = [npmax(np.abs([npmin(i), npmax(i)])) for i in arrays]
    return npmax(absolute)


def grid_to_table(grid):
    """
    Convert a grid to a table with the values and coordinates of each point.

    Takes a 2D grid as input, extracts the coordinates and runs them through
    :func:`numpy.meshgrid` to create a 2D table. Works for 2D grids and any number of
    variables. Use cases includes passing gridded data to functions that expect data in
    XYZ format, such as :class:`verde.BlockReduce`

    Parameters
    ----------
    grid : :class:`xarray.Dataset`
        A 2D grid with one or more data variables.

    Returns
    -------
    table : :class:`pandas.DataFrame`
        Table with coordinates and variable values for each point in the grid.

    Examples
    --------

    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample grid with a single data variable
    >>> temperature = xr.DataArray(
    ...     np.arange(20).reshape((4, 5)),
    ...     coords=(np.arange(4), np.arange(5, 10)),
    ...     dims=['northing', 'easting']
    ... )
    >>> grid = xr.Dataset({"temperature": temperature})
    >>> table  = grid_to_table(grid)
    >>> list(sorted(table.columns))
    ['easting', 'northing', 'temperature']
    >>> print(table.northing.values)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
    >>> print(table.easting.values)
    [5 6 7 8 9 5 6 7 8 9 5 6 7 8 9 5 6 7 8 9]
    >>> print(table.temperature.values)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    >>> # Grids with multiple data variables will have more columns.
    >>> wind_speed = xr.DataArray(
    ...     np.arange(20, 40).reshape((4, 5)),
    ...     coords=(np.arange(4), np.arange(5, 10)),
    ...     dims=['northing', 'easting']
    ... )
    >>> grid['wind_speed'] = wind_speed
    >>> table = grid_to_table(grid)
    >>> list(sorted(table.columns))
    ['easting', 'northing', 'temperature', 'wind_speed']
    >>> print(table.northing.values)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
    >>> print(table.easting.values)
    [5 6 7 8 9 5 6 7 8 9 5 6 7 8 9 5 6 7 8 9]
    >>> print(table.temperature.values)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    >>> print(table.wind_speed.values)
    [20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]

    """
    coordinate_names = [*grid.coords.keys()]
    coord_north = grid.coords[coordinate_names[0]].values
    coord_east = grid.coords[coordinate_names[1]].values
    coordinates = [i.ravel() for i in np.meshgrid(coord_east, coord_north)]
    coord_dict = {
        coordinate_names[0]: coordinates[1],
        coordinate_names[1]: coordinates[0],
    }
    variable_name = [*grid.data_vars.keys()]
    variable_data = grid.to_array().values
    variable_arrays = variable_data.reshape(
        len(variable_name), int(len(variable_data.ravel()) / len(variable_name))
    )
    var_dict = dict(zip(variable_name, variable_arrays))
    coord_dict.update(var_dict)
    data = pd.DataFrame(coord_dict)
    return data


def kdtree(coordinates, use_pykdtree=True, **kwargs):
    """
    Create a KD-Tree object with the given coordinate arrays.

    Automatically transposes and flattens the coordinate arrays into a single matrix for
    use in the KD-Tree classes.

    All other keyword arguments are passed to the KD-Tree class.

    If installed, package ``pykdtree`` will be used instead of
    :class:`scipy.spatial.cKDTree` for better performance. Not all features are
    available in ``pykdtree`` so if you require the scipy version set
    ``use_pykdtee=False``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). All coordinate arrays are
        used.
    use_pykdtree : bool
        If True, will prefer ``pykdtree`` (if installed) over
        :class:`scipy.spatial.cKDTree`. Otherwise, always use the scipy version.

    Returns
    -------
    tree : :class:`scipy.spatial.cKDTree` or ``pykdtree.kdtree.KDTree``
        The tree instance initialized with the given coordinates and arguments.

    """
    points = np.transpose(n_1d_arrays(coordinates, len(coordinates)))
    if pyKDTree is not None and use_pykdtree:
        tree = pyKDTree(points, **kwargs)
    else:
        tree = cKDTree(points, **kwargs)
    return tree
