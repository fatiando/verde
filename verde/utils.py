"""
General utilities.
"""
import functools

import dask
import numpy as np
import pandas as pd
import xarray as xr
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
from .coordinates import check_coordinates


def dispatch(function, delayed=False, client=None):
    """
    Decide how to wrap a function for Dask depending on the options given.

    Parameters
    ----------
    function : callable
        The function that will be called.
    delayed : bool
        If True, will wrap the function in :func:`dask.delayed`.
    client : None or dask.distributed Client
        If *delayed* is False and *client* is not None, will return a partial
        execution of the ``client.submit`` with the function as first argument.

    Returns
    -------
    function : callable
        The function wrapped in Dask.

    """
    if delayed:
        return dask.delayed(function)
    if client is not None:
        return functools.partial(client.submit, function)
    return function


def parse_engine(engine):
    """
    Choose the best engine available and check if it's valid.

    Parameters
    ----------
    engine : str
        The name of the engine. If ``"auto"`` will favor numba if it's
        available.

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
    Replace numba.jit if not installed with a function that raises RunTimeError

    Use as a decorator.

    Parameters
    ----------
    function
        A function that you would decorate with :func:`numba.jit`.

    Returns
    -------
    function
        A function that raises :class:`RunTimeError` warning that numba isn't
        installed.

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
        One or more arrays. If more than one are given, a single maximum will
        be calculated across all arrays.

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

    If the array contains NaNs, we'll use the ``nan`` version of of the numpy
    functions by default. You can turn this off through the *nan* argument.

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


def make_xarray_grid(
    coordinates,
    data,
    data_name,
    dims=("northing", "easting"),
    extra_coords_name=None,
):
    """
    Create an :class:`xarray.Dataset` grid from numpy arrays

    This functions creates an :class:`xarray.Dataset` out of 2d gridded data
    including easting and northing coordinates, any extra coordinates (like
    upward elevation, time, etc) and data arrays.

    Use this to transform the outputs of :func:`verde.grid_coordinates` and
    the ``predict`` method of a gridder into an :class:`xarray.Dataset`. 
    
    .. note::
    
        This is a utility function to help create 2D grids (i.e., grids with two ``dims`` 
        coordinates). For arbitrary N-dimensional arrays, use :mod:`xarray` directly.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with coordinates of each point in the grid. Each array must
        contain values for a dimension in the order: easting, northing,
        vertical, etc. All arrays must be 2d and need to have the same *shape*.
        These coordinates can be generated through
        :func:`verde.grid_coordinates`.
    data : array or tuple of arrays
        Array or tuple of arrays with data values on each point in the grid.
        Each array must contain values for a dimension in the same order as
        the coordinates. All arrays need to have the same *shape*.
    data_name : str or list
        String or list containing strings with the names of the data arrays.
    dims : list or None
        The names of the northing and easting data dimensions, respectively,
        in the output grid. Must be defined in the following order: northing
        dimension, easting dimension.
        **NOTE: This is an exception to the "easting" then
        "northing" pattern but is required for compatibility with xarray.**
        The easting and northing coordinates in the :class:`xarray.Dataset`
        will have the same names as the passed dimensions.
    extra_coords_name : str or list
        Name or list of names for any additional coordinates besides the
        easting and northing ones. Ignored if coordinates has
        only two elements. The extra coordinates are non-index coordinates of
        the grid array.

    Returns
    -------
    grid : :class:`xarray.Dataset`
        A 2D grid with one or more data variables.

    Examples
    --------

    >>> import numpy as np
    >>> import verde as vd
    >>> # Create the coordinates of the regular grid
    >>> coordinates = vd.grid_coordinates((-10, -6, 8, 10), spacing=2)
    >>> # And some dummy data for each point of the grid
    >>> data = np.ones_like(coordinates[0])
    >>> # Create the grid
    >>> grid = make_xarray_grid(coordinates, data, data_name="dummy")
    >>> print(grid)
    <xarray.Dataset>
    Dimensions:   (easting: 3, northing: 2)
    Coordinates:
      * easting   (easting) float64 -10.0 -8.0 -6.0
      * northing  (northing) float64 8.0 10.0
    Data variables:
        dummy     (northing, easting) float64 1.0 1.0 1.0 1.0 1.0 1.0

    >>> # Create a grid with an extra coordinate
    >>> coordinates = vd.grid_coordinates(
    ...     (-10, -6, 8, 10), spacing=2, extra_coords=5
    ... )
    >>> # And some dummy data for each point of the grid
    >>> data = np.ones_like(coordinates[0])
    >>> # Create the grid
    >>> grid = make_xarray_grid(
    ...     coordinates, data, data_name="dummy", extra_coords_name="upward"
    ... )
    >>> print(grid)
    <xarray.Dataset>
    Dimensions:   (easting: 3, northing: 2)
    Coordinates:
      * easting   (easting) float64 -10.0 -8.0 -6.0
      * northing  (northing) float64 8.0 10.0
        upward    (northing, easting) float64 5.0 5.0 5.0 5.0 5.0 5.0
    Data variables:
        dummy     (northing, easting) float64 1.0 1.0 1.0 1.0 1.0 1.0

    """
    coordinates = check_coordinates(coordinates)
    data = check_data(data)
    if isinstance(data_name, str):
        data_name = (data_name,)
    # dims is like shape with order (rows, cols) for the array
    # so the first element is northing and second is easting
    coords = {dims[1]: coordinates[0][0, :], dims[0]: coordinates[1][:, 0]}
    # Add any extra_coords to the dictionary
    if coordinates[2:]:
        extra_coords_name = _check_extra_coords_name(coordinates, extra_coords_name)
        for name, extra_coord in zip(extra_coords_name, coordinates[2:]):
            coords[name] = (dims, extra_coord)
    # Generate object
    if len(data) != len(data_name):
        raise ValueError(
            "Invalid data_names '{}'. ".format(data_name)
            + "Number of data names must match the number of "
            + "data arrays ('{}').".format(len(data))
        )
    data_vars = {name: (dims, value) for name, value in zip(data_name, data)}
    return xr.Dataset(data_vars, coords)


def _check_extra_coords_name(coordinates, extra_coords_name):
    """
    Sanity checks for extra_coords_name against coordinates

    Assuming that there are extra coordinates on the ``coordinates`` tuple.
    """
    # Convert to tuple if it's a str
    if isinstance(extra_coords_name, str):
        extra_coords_name = (extra_coords_name,)
    # Check if it's not None
    if extra_coords_name is None:
        raise ValueError(
            "Invalid extra_coords_name equal to None. "
            + "When passing one or more extra coordinate, "
            + "extra_coords_name cannot be None."
        )
    # Check if there are the same number of extra_coords than extra_coords_name
    if len(coordinates[2:]) != len(extra_coords_name):
        raise ValueError(
            "Invalid extra_coords_name '{}'. ".format(extra_coords_name)
            + "Number of extra coordinates names must match the number of "
            + "additional coordinates ('{}').".format(len(coordinates[2:]))
        )
    return extra_coords_name


def grid_to_table(grid):
    """
    Convert a grid to a table with the values and coordinates of each point.

    Takes a 2D grid as input, extracts the coordinates and runs them through
    :func:`numpy.meshgrid` to create a 2D table. Works for 2D grids and any
    number of variables. Use cases includes passing gridded data to functions
    that expect data in XYZ format, such as :class:`verde.BlockReduce`

    Parameters
    ----------
    grid : :class:`xarray.Dataset` or :class:`xarray.DataArray`
        A 2D grid with one or more data variables.

    Returns
    -------
    table : :class:`pandas.DataFrame`
        Table with coordinates and variable values for each point in the grid.
        Column names are taken from the grid. If *grid* is a
        :class:`xarray.DataArray` that doesn't have a ``name`` attribute
        defined, the column with data values will be called ``"scalars"``.

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
    >>> print(temperature.values)
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]]
    >>> # For DataArrays, the data column will be "scalars" by default
    >>> table = grid_to_table(temperature)
    >>> list(sorted(table.columns))
    ['easting', 'northing', 'scalars']
    >>> print(table.scalars.values)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    >>> print(table.northing.values)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
    >>> print(table.easting.values)
    [5 6 7 8 9 5 6 7 8 9 5 6 7 8 9 5 6 7 8 9]
    >>> # If the DataArray defines a "name", we will use that instead
    >>> temperature.name = "temperature_K"
    >>> table = grid_to_table(temperature)
    >>> list(sorted(table.columns))
    ['easting', 'northing', 'temperature_K']
    >>> # Conversion of Datasets will preserve the data variable names
    >>> grid = xr.Dataset({"temperature": temperature})
    >>> table  = grid_to_table(grid)
    >>> list(sorted(table.columns))
    ['easting', 'northing', 'temperature']
    >>> print(table.temperature.values)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    >>> print(table.northing.values)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
    >>> print(table.easting.values)
    [5 6 7 8 9 5 6 7 8 9 5 6 7 8 9 5 6 7 8 9]
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
    if hasattr(grid, "data_vars"):
        # It's a Dataset
        data_names = list(grid.data_vars.keys())
        data_arrays = [grid[name].values.ravel() for name in data_names]
        coordinate_names = list(grid[data_names[0]].dims)
    else:
        # It's a DataArray
        data_names = [grid.name if grid.name is not None else "scalars"]
        data_arrays = [grid.values.ravel()]
        coordinate_names = list(grid.dims)
    north = grid.coords[coordinate_names[0]].values
    east = grid.coords[coordinate_names[1]].values
    # Need to flip the coordinates because the names are in northing and
    # easting order
    coordinates = [i.ravel() for i in np.meshgrid(east, north)][::-1]
    data_dict = dict(zip(coordinate_names, coordinates))
    data_dict.update(dict(zip(data_names, data_arrays)))
    return pd.DataFrame(data_dict)


def kdtree(coordinates, use_pykdtree=True, **kwargs):
    """
    Create a KD-Tree object with the given coordinate arrays.

    Automatically transposes and flattens the coordinate arrays into a single
    matrix for use in the KD-Tree classes.

    All other keyword arguments are passed to the KD-Tree class.

    If installed, package ``pykdtree`` will be used instead of
    :class:`scipy.spatial.cKDTree` for better performance. Not all features are
    available in ``pykdtree`` so if you require the scipy version set
    ``use_pykdtee=False``.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). All coordinate
        arrays are used.
    use_pykdtree : bool
        If True, will prefer ``pykdtree`` (if installed) over
        :class:`scipy.spatial.cKDTree`. Otherwise, always use the scipy
        version.

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


def partition_by_sum(array, parts):
    """
    Partition an array into parts of approximately equal sum.

    Does not change the order of the array elements.

    Produces the partition indices on the array. Use :func:`numpy.split` to
    divide the array along these indices.

    .. warning::

        Depending on the input and number of parts, there might not exist
        partition points. In these cases, the function will raise
        ``ValueError``. This is more likely to happen as the number of parts
        approaches the number of elements in the array.

    Parameters
    ----------
    array : array or array-like
        The 1D array that will be partitioned. The array will be raveled before
        computations.
    parts : int
        Number of parts to split the array. Can be at most the number of
        elements in the array.

    Returns
    -------
    indices : array
        The indices in which the array should be split.

    Notes
    -----

    Solution from https://stackoverflow.com/a/54024280

    Examples
    --------

    >>> import numpy as np
    >>> array = np.arange(10)
    >>> split_points = partition_by_sum(array, parts=2)
    >>> print(split_points)
    [7]
    >>> for part in np.split(array, split_points):
    ...     print(part, part.sum())
    [0 1 2 3 4 5 6] 21
    [7 8 9] 24
    >>> split_points = partition_by_sum(array, parts=3)
    >>> print(split_points)
    [6 8]
    >>> for part in np.split(array, split_points):
    ...     print(part, part.sum())
    [0 1 2 3 4 5] 15
    [6 7] 13
    [8 9] 17
    >>> split_points = partition_by_sum(array, parts=5)
    >>> print(split_points)
    [4 6 7 9]
    >>> for part in np.split(array, split_points):
    ...     print(part, part.sum())
    [0 1 2 3] 6
    [4 5] 9
    [6] 6
    [7 8] 15
    [9] 9
    >>> # Use an array with a random looking element order
    >>> array = [5, 6, 4, 6, 8, 1, 2, 6, 3, 3]
    >>> split_points = partition_by_sum(array, parts=2)
    >>> print(split_points)
    [4]
    >>> for part in np.split(array, split_points):
    ...     print(part, part.sum())
    [5 6 4 6] 21
    [8 1 2 6 3 3] 23
    >>> # Splits can have very different sums but this is best that can be done
    >>> # without changing the order of the array.
    >>> split_points = partition_by_sum(array, parts=5)
    >>> print(split_points)
    [1 3 4 7]
    >>> for part in np.split(array, split_points):
    ...     print(part, part.sum())
    [5] 5
    [6 4] 10
    [6] 6
    [8 1 2] 11
    [6 3 3] 12

    """
    array = np.atleast_1d(array).ravel()
    if parts > array.size:
        raise ValueError(
            "Cannot partition an array of size {} into {} parts of equal sum.".format(
                array.size, parts
            )
        )
    cumulative_sum = array.cumsum()
    # Ideally, we want each part to have the same number of points (total /
    # parts).
    ideal_sum = cumulative_sum[-1] // parts
    # If the parts are ideal, the cumulative sum of each part will be this
    ideal_cumsum = np.arange(1, parts) * ideal_sum
    # Find the places in the real cumulative sum where the ideal values would
    # be. These are the split points. Between each split point, the sum of
    # elements will be approximately the ideal sum. Need to insert to the right
    # side so that we find cumsum[i - 1] <= ideal < cumsum[i]. This way, if a
    # part has ideal sum, the last element (i - 1) will be included. Otherwise,
    # we would never have ideal sums.
    indices = np.searchsorted(cumulative_sum, ideal_cumsum, side="right")
    # Check for repeated split points, which indicates that there is no way to
    # split the array.
    if np.unique(indices).size != indices.size:
        raise ValueError(
            "Could not find partition points to split the array into {} parts "
            "of equal sum.".format(parts)
        )
    return indices
