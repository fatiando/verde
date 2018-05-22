"""
Functions for generating and manipulating coordinates.
"""
import numpy as np
from sklearn.utils import check_random_state


def check_region(region):
    """
    Check that the given region dimensions are valid.

    For example, the west limit should not be greater than the east and there
    must be exactly 4 values given.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.

    Raises
    ------
    ValueError
        If the region doesn't have exactly 4 entries, W > E, or S > N.

    """
    if len(region) != 4:
        raise ValueError("Invalid region '{}'. Only 4 values allowed."
                         .format(region))
    w, e, s, n = region
    if w > e:
        raise ValueError("Invalid region '{}' (W, E, S, N). Must have W =< E."
                         .format(region))
    if s > n:
        raise ValueError("Invalid region '{}' (W, E, S, N). Must have S =< N."
                         .format(region))


def get_region(easting, northing):
    """
    Get the bounding region of the given coordinates.

    Parameters
    ----------
    easting : array
        The values of the West-East coordinates of each data point.
    northing : array
        The values of the South-North coordinates of each data point.

    Returns
    -------
    region : tuple = (W, E, S, N)
        The boundaries of a given region in Cartesian or geographic
        coordinates.

    Examples
    --------

    >>> east, north = grid_coordinates((0, 1, -10, -6), shape=(10, 10))
    >>> print(get_region(east, north))
    (0.0, 1.0, -10.0, -6.0)

    """
    region = (np.min(easting), np.max(easting),
              np.min(northing), np.max(northing))
    return region


def scatter_points(region, size, random_state=None):
    """
    Generate the coordinates for a random scatter of points.

    The points are drawn from a uniform distribution.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    size : int
        The number of points to generate.
    random_state : numpy.random.RandomState or an int seed
        A random number generator used to define the state of the random
        permutations. Use a fixed seed to make sure computations are
        reproducible. Use ``None`` to choose a seed automatically (resulting in
        different numbers with each run).

    Returns
    -------
    easting, northing : 1d arrays
        The West-East and South-North coordinates of each point.

    Examples
    --------

    >>> # We'll use a seed value will ensure that the same will be generated
    >>> # every time.
    >>> easting, northing = scatter_points((0, 10, -2, -1), 4, random_state=0)
    >>> print(', '.join(['{:.4f}'.format(i) for i in easting]))
    5.4881, 7.1519, 6.0276, 5.4488
    >>> print(', '.join(['{:.4f}'.format(i) for i in northing]))
    -1.5763, -1.3541, -1.5624, -1.1082

    See also
    --------
    grid_coordinates : Generate coordinates for each point on a regular grid
    profile_coordinates : Coordinates for a profile between two points

    """
    check_region(region)
    random = check_random_state(random_state)
    w, e, s, n = region
    easting = random.uniform(w, e, size)
    northing = random.uniform(s, n, size)
    return easting, northing


def grid_coordinates(region, shape=None, spacing=None, adjust='spacing',
                     pixel_register=False):
    """
    Generate the coordinates for each point on a regular grid.

    The grid can be specified by either the number of points in each dimension
    (the *shape*) or by the grid node spacing.

    If the given region is not divisible by the desired spacing, either the
    region or the spacing will have to be adjusted. By default, the spacing
    will be rounded to the nearest multiple. Optionally, the East and North
    boundaries of the region can be adjusted to fit the exact spacing given.
    See the examples below.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    shape : tuple = (n_north, n_east) or None
        The number of points in the South-North and West-East directions,
        respectively.
    spacing : float, tuple = (s_north, s_east), or None
        The grid spacing in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if
        *shape* is given instead of *spacing*. Defaults to adjusting the
        spacing.
    pixel_register : bool
        If True, the coordinates will refer to the center of each grid pixel
        instead of the grid lines. In practice, this means that there will be
        one less element per dimension of the grid when compared to grid line
        registered. Default is False.

    Returns
    -------
    easting, northing : 2d arrays
        The West-East and South-North coordinates of each point in the grid.
        The arrays have the specified *shape*.

    Examples
    --------

    >>> east, north = grid_coordinates(region=(0, 5, 0, 10), shape=(5, 3))
    >>> print(east.shape, north.shape)
    (5, 3) (5, 3)
    >>> # Lower printing precision to shorten this example
    >>> import numpy as np; np.set_printoptions(precision=1, suppress=True)
    >>> print(east)
    [[0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]]
    >>> print(north)
    [[ 0.   0.   0. ]
     [ 2.5  2.5  2.5]
     [ 5.   5.   5. ]
     [ 7.5  7.5  7.5]
     [10.  10.  10. ]]
    >>> # The grid can also be specified using the spacing between points
    >>> # instead of the shape.
    >>> east, north = grid_coordinates(region=(0, 5, 0, 10), spacing=2.5)
    >>> print(east.shape, north.shape)
    (5, 3) (5, 3)
    >>> print(east)
    [[0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]
     [0.  2.5 5. ]]
    >>> print(north)
    [[ 0.   0.   0. ]
     [ 2.5  2.5  2.5]
     [ 5.   5.   5. ]
     [ 7.5  7.5  7.5]
     [10.  10.  10. ]]
    >>> # The spacing can be different for northing and easting, respectively
    >>> east, north = grid_coordinates(region=(-5, 1, 0, 10), spacing=(2.5, 1))
    >>> print(east.shape, north.shape)
    (5, 7) (5, 7)
    >>> print(east)
    [[-5. -4. -3. -2. -1.  0.  1.]
     [-5. -4. -3. -2. -1.  0.  1.]
     [-5. -4. -3. -2. -1.  0.  1.]
     [-5. -4. -3. -2. -1.  0.  1.]
     [-5. -4. -3. -2. -1.  0.  1.]]
    >>> print(north)
    [[ 0.   0.   0.   0.   0.   0.   0. ]
     [ 2.5  2.5  2.5  2.5  2.5  2.5  2.5]
     [ 5.   5.   5.   5.   5.   5.   5. ]
     [ 7.5  7.5  7.5  7.5  7.5  7.5  7.5]
     [10.  10.  10.  10.  10.  10.  10. ]]
    >>> # If the region can't be divided into the desired spacing, the spacing
    >>> # will be adjusted to conform to the region
    >>> east, north = grid_coordinates(region=(-5, 0, 0, 5), spacing=2.6)
    >>> print(east.shape, north.shape)
    (3, 3) (3, 3)
    >>> print(east)
    [[-5.  -2.5  0. ]
     [-5.  -2.5  0. ]
     [-5.  -2.5  0. ]]
    >>> print(north)
    [[0.  0.  0. ]
     [2.5 2.5 2.5]
     [5.  5.  5. ]]
    >>> east, north = grid_coordinates(region=(-5, 0, 0, 5), spacing=2.4)
    >>> print(east.shape, north.shape)
    (3, 3) (3, 3)
    >>> print(east)
    [[-5.  -2.5  0. ]
     [-5.  -2.5  0. ]
     [-5.  -2.5  0. ]]
    >>> print(north)
    [[0.  0.  0. ]
     [2.5 2.5 2.5]
     [5.  5.  5. ]]
    >>> # You can also choose to adjust the East and North boundaries of the
    >>> # region instead.
    >>> east, north = grid_coordinates(region=(-5, 0, 0, 5), spacing=2.6,
    ...                                adjust='region')
    >>> print(east.shape, north.shape)
    (3, 3) (3, 3)
    >>> print(east)
    [[-5.  -2.4  0.2]
     [-5.  -2.4  0.2]
     [-5.  -2.4  0.2]]
    >>> print(north)
    [[0.  0.  0. ]
     [2.6 2.6 2.6]
     [5.2 5.2 5.2]]
    >>> east, north = grid_coordinates(region=(-5, 0, 0, 5), spacing=2.4,
    ...                                adjust='region')
    >>> print(east.shape, north.shape)
    (3, 3) (3, 3)
    >>> print(east)
    [[-5.  -2.6 -0.2]
     [-5.  -2.6 -0.2]
     [-5.  -2.6 -0.2]]
    >>> print(north)
    [[0.  0.  0. ]
     [2.4 2.4 2.4]
     [4.8 4.8 4.8]]
    >>> # We can optionally generate coordinates for the center of each grid
    >>> # pixel instead of the corner (default)
    >>> east, north = grid_coordinates(region=(0, 5, 0, 10), spacing=2.5,
    ...                                pixel_register=True)
    >>> # Lower printing precision to shorten this example
    >>> import numpy as np; np.set_printoptions(precision=2, suppress=True)
    >>> print(east.shape, north.shape)
    (4, 2) (4, 2)
    >>> print(east)
    [[1.25 3.75]
     [1.25 3.75]
     [1.25 3.75]
     [1.25 3.75]]
    >>> print(north)
    [[1.25 1.25]
     [3.75 3.75]
     [6.25 6.25]
     [8.75 8.75]]

    See also
    --------
    scatter_points : Generate the coordinates for a random scatter of points
    profile_coordinates : Coordinates for a profile between two points

    """
    check_region(region)
    if shape is not None and spacing is not None:
        raise ValueError(
            "Both grid shape and spacing provided. Only one is allowed.")
    if shape is None and spacing is None:
        raise ValueError("Either a grid shape or a spacing must be provided.")
    if spacing is not None:
        shape, region = spacing_to_shape(region, spacing, adjust)
    nnorth, neast = shape
    w, e, s, n = region
    east_lines = np.linspace(w, e, neast)
    north_lines = np.linspace(s, n, nnorth)
    if pixel_register:
        east_lines = east_lines[:-1] + (east_lines[1] - east_lines[0])/2
        north_lines = north_lines[:-1] + (north_lines[1] - north_lines[0])/2
    easting, northing = np.meshgrid(east_lines, north_lines)
    return easting, northing


def spacing_to_shape(region, spacing, adjust):
    """
    Convert the grid spacing to a grid shape.

    Adjusts the spacing or the region if the desired spacing is not a multiple
    of the grid dimensions.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    spacing : float, tuple = (s_north, s_east), or None
        The grid spacing in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if
        *shape* is given instead of *spacing*. Defaults to adjusting the
        spacing.

    Returns
    -------
    shape, region : tuples
        The calculated shape and region that best fits the desired spacing.
        Spacing or region may be adjusted.

    """
    if adjust not in ['spacing', 'region']:
        raise ValueError(
            "Invalid value for *adjust* '{}'. Should be 'spacing' or 'region'"
            .format(adjust))

    spacing = np.atleast_1d(spacing)
    if len(spacing) == 1:
        deast = dnorth = spacing[0]
    elif len(spacing) == 2:
        dnorth, deast = spacing
    else:
        raise ValueError("Only two values allowed for grid spacing: {}"
                         .format(str(spacing)))

    w, e, s, n = region
    # Add 1 to get the number of nodes, not segments
    nnorth = int(round((n - s)/dnorth)) + 1
    neast = int(round((e - w)/deast)) + 1
    if adjust == 'region':
        # The shape is the same but we adjust the region so that the spacing
        # isn't altered when we do the linspace.
        n = s + (nnorth - 1)*dnorth
        e = w + (neast - 1)*deast
    return (nnorth, neast), (w, e, s, n)


def profile_coordinates(point1, point2, size, coordinate_system='cartesian'):
    """
    Coordinates for a profile along a line between two points.

    If on a geographic coordinate system, will calculate along a great circle.
    Otherwise, will use a straight line.

    Parameters
    ----------
    point1 : tuple or list
        ``(easting, northing)`` West-East and South-North coordinates of the
        first point, respectively.
    point2 : tuple or list
        ``(easting, northing)`` West-East and South-North coordinates of the
        second point, respectively.
    size : int
        Number of points to sample along the line.
    coordinate_system : str
        The coordinate system used to define the points and the line. Either
        ``'cartesian'`` or ``'geographic'``.

    Returns
    -------
    easting, northing, distances : 1d arrays
        The easting and northing coordinates of points along the straight line
        and the distances from the first point.

    Examples
    --------

    >>> east, north, dist = profile_coordinates((1, 10), (1, 20), size=11)
    >>> print('easting:', ', '.join('{:.1f}'.format(i) for i in east))
    easting: 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    >>> print('northing:', ', '.join('{:.1f}'.format(i) for i in north))
    northing: 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
    >>> print('distance:', ', '.join('{:.1f}'.format(i) for i in dist))
    distance: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0

    See also
    --------
    scatter_points : Generate the coordinates for a random scatter of points
    grid_coordinates : Generate coordinates for each point on a regular grid

    """
    valid_coordinate_systems = ['cartesian', 'geographic']
    if coordinate_system not in valid_coordinate_systems:
        raise ValueError(
            "Invalid coordinate system '{}'. Must be one of {}."
            .format(coordinate_system, str(valid_coordinate_systems)))
    if size <= 0:
        raise ValueError("Invalid profile size '{}'. Must be > 0."
                         .format(size))
    if coordinate_system == 'geographic':
        raise NotImplementedError()
    elif coordinate_system == 'cartesian':
        east1, north1 = point1
        east2, north2 = point2
        separation = np.sqrt((east1 - east2)**2 + (north1 - north2)**2)
        distances = np.linspace(0, separation, size)
        angle = np.arctan2(north2 - north1, east2 - east1)
        easting = east1 + distances*np.cos(angle)
        northing = north1 + distances*np.sin(angle)
    return easting, northing, distances


def inside(easting, northing, region, out=None, tmp=None):
    """
    Determine which points fall inside a given region.

    Points at the boundary are counted as being outsize.

    Parameters
    ----------
    easting : array
        The values of the West-East coordinates of each data point.
    northing : array
        The values of the South-North coordinates of each data point.
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    out : None or array of booleans
        Numpy array to be used as output. The contents will be overwritten and
        the same array will be returned.
    tmp : None or tuple
        Numpy arrays used to store the outputs of temporary logical operations.
        Passing in pre-allocated arrays avoids the overhead of allocation when
        calling this function repeatedly. If not None, then should be a tuple
        of 4 numpy arrays of boolean type and a shape equal to or broadcast
        from the input coordinates.

    Returns
    -------
    are_inside : array of booleans
        An array of booleans with the same shape as the input coordinate
        arrays. Will be ``True`` if the respective coordinates fall inside the
        area, ``False`` otherwise.

    Examples
    --------

    >>> import numpy as np
    >>> east = np.array([1, 2, 3, 4, 5, 6])
    >>> north = np.array([10, 11, 12, 13, 14, 15])
    >>> region = [2.5, 5.5, 12, 15]
    >>> print(inside(east, north, region))
    [False False  True  True  True False]
    >>> # This also works for 2D-arrays
    >>> east = np.array([[1, 1, 1],
    ...                  [2, 2, 2],
    ...                  [3, 3, 3]])
    >>> north = np.array([[5, 7, 9],
    ...                   [5, 7, 9],
    ...                   [5, 7, 9]])
    >>> region = [0.5, 2.5, 6, 9]
    >>> print(inside(east, north, region))
    [[False  True  True]
     [False  True  True]
     [False False False]]

    """
    check_region(region)
    w, e, s, n = region
    # Allocate temporary arrays to minimize memory allocation overhead
    if out is None:
        out = np.empty_like(easting, dtype=np.bool)
    if tmp is None:
        tmp = tuple(np.empty_like(easting, dtype=np.bool) for i in range(4))
    # Using the logical functions is a lot faster than & > < for some reason
    # Plus, this way avoids repeated allocation of intermediate arrays
    in_we = np.logical_and(np.greater_equal(easting, w, out=tmp[0]),
                           np.less_equal(easting, e, out=tmp[1]),
                           out=tmp[2])
    in_ns = np.logical_and(np.greater_equal(northing, s, out=tmp[0]),
                           np.less_equal(northing, n, out=tmp[1]),
                           out=tmp[3])
    are_inside = np.logical_and(in_we, in_ns, out=out)
    return are_inside
