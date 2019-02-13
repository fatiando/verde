"""
General utilities.
"""
import functools

import numpy as np
import pandas as pd
import xarray as xr


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


def load_surfer(fname, dtype='float64'):
    """
    Read data from a Surfer ASCII grid file.

    Surfer is a contouring, griding and surface mapping software
    from GoldenSoftware. The names and logos for Surfer and Golden
    Software are registered trademarks of Golden Software, Inc.

    http://www.goldensoftware.com/products/surfer

    This feature was ported from Fatiando, but instead of
    output being a dictionary, in verde the output is a `xarray.DataArray`
    that contains the easting and northing coordinates, and data values.

    Parameters:

    * fname : str
        Name of the Surfer grid file
    * dtype : numpy dtype object or string
        The type of variable used for the data. Default is numpy.float64. Use
        numpy.float32 if the data are large and precision is not an issue.

    Returns:

    * data : :class: `xarray.DataArray`
        A 2D grid with the data variables.
    """
    # Surfer ASCII grid structure
    # DSAA            Surfer ASCII GRD ID
    # nCols nRows     number of columns and rows
    # xMin xMax       X min max
    # yMin yMax       Y min max
    # zMin zMax       Z min max
    # z11 z21 z31 ... List of Z values
    with open(fname) as input_file:
        # DSAA is a Surfer ASCII GRD ID (discard it for now)
        input_file.readline()
        # Read the number of columns (ny) and rows (nx)
        ydims, xdims = [int(s) for s in input_file.readline().split()]
        # Our x points North, so the first thing we read is y, not x.
        ymin, ymax = [float(s) for s in input_file.readline().split()]
        xmin, xmax = [float(s) for s in input_file.readline().split()]
        dmin, dmax = [float(s) for s in input_file.readline().split()]
        field = np.fromiter((float(s)
                             for line in input_file
                             for s in line.split()),
                            dtype=dtype)
        nans = field >= 1.70141e+38
        if np.any(nans):
            field = np.ma.masked_where(nans, field)
        err_msg = "{} of data ({}) doesn't match one from file ({})."
        assert np.allclose(dmin, field.min()), err_msg.format('Min', dmin,
                                                              field.min())
        assert np.allclose(dmax, field.max()), err_msg.format('Max', dmax,
                                                              field.max())
        data = xr.DataArray(field.reshape(ydims, xdims), coords=(np.linspace(
        ymin, ymax, ydims), np.linspace(xmin, xmax, xdims)), dims=['northing',
        'easting'])
    return data
