"""
General utilities.
"""
import functools

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
    Convert a grid to a table with the values and coordinates of each point.

    Takes a 2D grid as input, extracts the coordinates and runs them through
    :func:`numpy.meshgrid` to create a 2D table. Works for 2D grids and
    any number of variables. Use cases includes passing gridded data to
    functions that expect data in XYZ format such as :class:`verde.BlockReduce`

    Parameters
    ----------
    grid : :class:`xarray.Dataset`
        A 2D grid with a single data variable.

    Returns
    -------
    table : :class:`pandas.DataFrame`
        :class:`pandas.DataFrame` with coordinates and variables.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> temperature = xr.DataArray(np.arange(20).reshape(4,5),\
    coords = (np.arange(4), np.arange(5)), dims=['northing', 'easting'])
    >>> wind_speed = xr.DataArray(np.arange(20,40).reshape(4,5),\
    coords = (np.arange(4), np.arange(5)), dims=['northing', 'easting'])
    >>> print(temperature)
    <xarray.DataArray (northing: 4, easting: 5)>
    array([[20, 21, 22, 23, 24],
           [25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34],
           [35, 36, 37, 38, 39]])
    Coordinates:
      * northing  (northing) int32 0 1 2 3
      * easting   (easting) int32 0 1 2 3 4

    >>> example_dataset = temperature.to_dataset(name = 'temperature')
    >>> example_dataset['wind_speed'] = wind_speed
    >>> print(example_dataset)
    <xarray.Dataset>
    Dimensions:      (easting: 5, northing: 4)
    Coordinates:
      * northing     (northing) int32 0 1 2 3
      * easting      (easting) int32 0 1 2 3 4
    Data variables:
        temperature  (northing, easting) int32 0 1 2 3 4 ... 14 15 16 17 18 19
        wind_speed   (northing, easting) int32 20 21 22 ... 35 36 37 38 39

    >>> print(grid_to_table(example_dataset))
    	   northing	easting	temperature	wind_speed
        0	0         0	       0	      20
        1	0         1        1	      21
        2	0	      2	       2	      22
        3	0	      3        3          23
        4	0	      4    	   4	      24
        5	1	      0	       5	      25
        6	1	      1	       6	      26
        7	1	      2	       7	      27
        8	1	      3	       8	      28
        9	1	      4	       9	      29
        10	2	      0	       10	      30
        11	2	      1	       11	      31
        12	2	      2	       12	      32
        13	2	      3	       13  	      33
        14	2	      4	       14	      34
        15	3         0	       15	      35
        16	3	      1	       16	      36
        17	3	      2	       17         37
        18	3         3	       18	      38
        19	3	      4	       19	      39


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
