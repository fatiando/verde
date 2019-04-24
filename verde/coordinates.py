"""
Functions for generating and manipulating coordinates.
"""
import numpy as np
from sklearn.utils import check_random_state

from .base.utils import n_1d_arrays
from .utils import kdtree


def check_region(region, latlon=False):
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
        raise ValueError("Invalid region '{}'. Only 4 values allowed.".format(region))
    w, e, s, n = region
    if latlon:
        if w > 360 or w < -180 or e > 360 or e < -180:
            raise ValueError(
                "Invalid region '{}'. ".format(region)
                + "Longitudes must be > -180 degrees and < 360 degrees."
            )
        if s > 90 or s < -90 or n > 90 or n < -90:
            raise ValueError(
                "Invalid region '{}'. ".format(region)
                + "Latitudes must be > -90 degrees and < 90 degrees."
            )
        if abs(e - w) > 360:
            raise ValueError(
                "Invalid region '{}' (W, E, S, N). ".format(region)
                + "East and West boundaries must not be separated by an angle greater"
                + "than 360 degrees."
            )
    else:
        if w > e:
            raise ValueError(
                "Invalid region '{}' (W, E, S, N).Must have W =< E.".format(region)
                + "If working with geographic coordinates, don't forget to add the "
                + "latlon=True argument."
            )
        if s > n:
            raise ValueError(
                "Invalid region '{}' (W, E, S, N). Must have S =< N.".format(region)
            )


def get_region(coordinates):
    """
    Get the bounding region of the given coordinates.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.

    Returns
    -------
    region : tuple = (W, E, S, N)
        The boundaries of a given region in Cartesian or geographic
        coordinates.

    Examples
    --------

    >>> coords = grid_coordinates((0, 1, -10, -6), shape=(10, 10))
    >>> print(get_region(coords))
    (0.0, 1.0, -10.0, -6.0)

    """
    easting, northing = coordinates[:2]
    region = (np.min(easting), np.max(easting), np.min(northing), np.max(northing))
    return region


def pad_region(region, pad):
    """
    Extend the borders of a region by the given amount.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    pad : float or tuple = (pad_north, pad_east)
        The amount of padding to add to the region. If it's a single number,
        add this to all boundaries of region equally. If it's a tuple of
        numbers, then will add different padding to the North-South and
        East-West dimensions.

    Returns
    -------
    padded_region : list = [W, E, S, N]
        The padded region.

    Examples
    --------

    >>> pad_region((0, 1, -5, -3), 1)
    (-1, 2, -6, -2)
    >>> pad_region((0, 1, -5, -3), (3, 2))
    (-2, 3, -8, 0)

    """
    if np.isscalar(pad):
        pad = (pad, pad)
    w, e, s, n = region
    padded = (w - pad[1], e + pad[1], s - pad[0], n + pad[0])
    return padded


def project_region(region, projection):
    """
    Calculate the bounding box of a region in projected coordinates.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    projection : callable or None
        If not None, then should be a callable object (like a function)
        ``projection(easting, northing) -> (proj_easting, proj_northing)`` that
        takes in easting and northing coordinate arrays and returns projected
        northing and easting coordinate arrays.

    Returns
    -------
    proj_region : list = [W, E, S, N]
        The bounding box of the projected region.

    Examples
    --------

    >>> def projection(x, y):
    ...     return (2*x, -1*y)
    >>> project_region((3, 5, -9, -4), projection)
    (6.0, 10.0, 4.0, 9.0)

    """
    east, north = grid_coordinates(region, shape=(101, 101))
    east, north = projection(east.ravel(), north.ravel())
    return (east.min(), east.max(), north.min(), north.max())


def scatter_points(region, size, random_state=None, extra_coords=None):
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
    extra_coords : None, scalar, or list
        If not None, then value(s) of extra coordinate arrays to be generated. These
        extra arrays will have the same *size* as the others but will contain a
        constant value. Will generate an extra array per value given in *extra_coords*.
        Use this to generate arrays of constant heights or times, for example, that
        might be needed to evaluate a gridder.

    Returns
    -------
    coordinates : tuple of arrays
        Arrays with coordinates of each point in the grid. Each array contains values
        for a dimension in the order: easting, northing, vertical, and any extra
        dimensions given in *extra_coords*. All arrays will have the specified *size*.

    Examples
    --------

    >>> # We'll use a seed value will ensure that the same will be generated
    >>> # every time.
    >>> easting, northing = scatter_points((0, 10, -2, -1), 4, random_state=0)
    >>> print(', '.join(['{:.4f}'.format(i) for i in easting]))
    5.4881, 7.1519, 6.0276, 5.4488
    >>> print(', '.join(['{:.4f}'.format(i) for i in northing]))
    -1.5763, -1.3541, -1.5624, -1.1082
    >>> easting, northing, height = scatter_points((0, 10, -2, -1), 4, random_state=0,
    ...                                            extra_coords=12)
    >>> print(height)
    [12. 12. 12. 12.]
    >>> easting, northing, height, time = scatter_points(
    ...     (0, 10, -2, -1), 4, random_state=0, extra_coords=[12, 1986])
    >>> print(height)
    [12. 12. 12. 12.]
    >>> print(time)
    [1986. 1986. 1986. 1986.]

    See also
    --------
    grid_coordinates : Generate coordinates for each point on a regular grid
    profile_coordinates : Coordinates for a profile between two points

    """
    check_region(region)
    random = check_random_state(random_state)
    coordinates = []
    for lower, upper in np.array(region).reshape((len(region) // 2, 2)):
        coordinates.append(random.uniform(lower, upper, size))
    if extra_coords is not None:
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates)


def grid_coordinates(
    region,
    shape=None,
    spacing=None,
    adjust="spacing",
    pixel_register=False,
    extra_coords=None,
):
    """
    Generate the coordinates for each point on a regular grid.

    The grid can be specified by either the number of points in each dimension (the
    *shape*) or by the grid node spacing.

    If the given region is not divisible by the desired spacing, either the region or
    the spacing will have to be adjusted. By default, the spacing will be rounded to the
    nearest multiple. Optionally, the East and North boundaries of the region can be
    adjusted to fit the exact spacing given. See the examples below.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic coordinates.
    shape : tuple = (n_north, n_east) or None
        The number of points in the South-North and West-East directions, respectively.
    spacing : float, tuple = (s_north, s_east), or None
        The grid spacing in the South-North and West-East directions, respectively. A
        single value means that the spacing is equal in both directions.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if *shape* is
        given instead of *spacing*. Defaults to adjusting the spacing.
    pixel_register : bool
        If True, the coordinates will refer to the center of each grid pixel instead of
        the grid lines. In practice, this means that there will be one less element per
        dimension of the grid when compared to grid line registered. Default is False.
    extra_coords : None, scalar, or list
        If not None, then value(s) of extra coordinate arrays to be generated. These
        extra arrays will have the same *shape* as the others but will contain a
        constant value. Will generate an extra array per value given in *extra_coords*.
        Use this to generate arrays of constant heights or times, for example, that
        might be needed to evaluate a gridder.

    Returns
    -------
    coordinates : tuple of arrays
        Arrays with coordinates of each point in the grid. Each array contains values
        for a dimension in the order: easting, northing, vertical, and any extra
        dimensions given in *extra_coords*. All arrays will have the specified *shape*.

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
    >>> # Generate arrays for other coordinates that have a constant value.
    >>> east, north, height = grid_coordinates(region=(0, 5, 0, 10), spacing=2.5,
    ...                                        extra_coords=57)
    >>> print(east.shape, north.shape, height.shape)
    (5, 3) (5, 3) (5, 3)
    >>> print(height)
    [[57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]]
    >>> east, north, height, time = grid_coordinates(region=(0, 5, 0, 10), spacing=2.5,
    ...                                              extra_coords=[57, 0.1])
    >>> print(east.shape, north.shape, height.shape, time.shape)
    (5, 3) (5, 3) (5, 3) (5, 3)
    >>> print(height)
    [[57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]]
    >>> print(time)
    [[0.1 0.1 0.1]
     [0.1 0.1 0.1]
     [0.1 0.1 0.1]
     [0.1 0.1 0.1]
     [0.1 0.1 0.1]]
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
        raise ValueError("Both grid shape and spacing provided. Only one is allowed.")
    if shape is None and spacing is None:
        raise ValueError("Either a grid shape or a spacing must be provided.")
    if spacing is not None:
        shape, region = spacing_to_shape(region, spacing, adjust)
    east_lines = np.linspace(region[0], region[1], shape[1])
    north_lines = np.linspace(region[2], region[3], shape[0])
    if pixel_register:
        east_lines = east_lines[:-1] + (east_lines[1] - east_lines[0]) / 2
        north_lines = north_lines[:-1] + (north_lines[1] - north_lines[0]) / 2
    coordinates = list(np.meshgrid(east_lines, north_lines))
    if extra_coords is not None:
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates)


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
    if adjust not in ["spacing", "region"]:
        raise ValueError(
            "Invalid value for *adjust* '{}'. Should be 'spacing' or 'region'".format(
                adjust
            )
        )

    spacing = np.atleast_1d(spacing)
    if len(spacing) == 1:
        deast = dnorth = spacing[0]
    elif len(spacing) == 2:
        dnorth, deast = spacing
    else:
        raise ValueError(
            "Only two values allowed for grid spacing: {}".format(str(spacing))
        )

    w, e, s, n = region
    # Add 1 to get the number of nodes, not segments
    nnorth = int(round((n - s) / dnorth)) + 1
    neast = int(round((e - w) / deast)) + 1
    if adjust == "region":
        # The shape is the same but we adjust the region so that the spacing
        # isn't altered when we do the linspace.
        n = s + (nnorth - 1) * dnorth
        e = w + (neast - 1) * deast
    return (nnorth, neast), (w, e, s, n)


def profile_coordinates(point1, point2, size, extra_coords=None):
    """
    Coordinates for a profile along a straight line between two points.

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
    extra_coords : None, scalar, or list
        If not None, then value(s) of extra coordinate arrays to be generated. These
        extra arrays will have the same *size* as the others but will contain a
        constant value. Will generate an extra array per value given in *extra_coords*.
        Use this to generate arrays of constant heights or times, for example, that
        might be needed to evaluate a gridder.

    Returns
    -------
    coordinates, distances : tuple and 1d array
        The coordinates of points along the straight line and the distances from the
        first point.

    Examples
    --------

    >>> (east, north), dist = profile_coordinates((1, 10), (1, 20), size=11)
    >>> print('easting:', ', '.join('{:.1f}'.format(i) for i in east))
    easting: 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    >>> print('northing:', ', '.join('{:.1f}'.format(i) for i in north))
    northing: 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
    >>> print('distance:', ', '.join('{:.1f}'.format(i) for i in dist))
    distance: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    >>> (east, north, height), dist = profile_coordinates(
    ...     (1, 10), (1, 20), size=11, extra_coords=35)
    >>> print(height)
    [35. 35. 35. 35. 35. 35. 35. 35. 35. 35. 35.]
    >>> (east, north, height, time), dist = profile_coordinates(
    ...     (1, 10), (1, 20), size=11, extra_coords=[35, 0.1])
    >>> print(height)
    [35. 35. 35. 35. 35. 35. 35. 35. 35. 35. 35.]
    >>> print(time)
    [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]

    See also
    --------
    scatter_points : Generate the coordinates for a random scatter of points
    grid_coordinates : Generate coordinates for each point on a regular grid

    """
    if size <= 0:
        raise ValueError("Invalid profile size '{}'. Must be > 0.".format(size))
    diffs = [i - j for i, j in zip(point2, point1)]
    separation = np.hypot(*diffs)
    distances = np.linspace(0, separation, size)
    angle = np.arctan2(*reversed(diffs))
    coordinates = [
        point1[0] + distances * np.cos(angle),
        point1[1] + distances * np.sin(angle),
    ]
    if extra_coords is not None:
        for value in np.atleast_1d(extra_coords):
            coordinates.append(np.ones_like(coordinates[0]) * value)
    return tuple(coordinates), distances


def inside(coordinates, region, latlon=False):
    """
    Determine which points fall inside a given region.

    Points at the boundary are counted as being outsize.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    latlon : bool (optional)
        If True both `region` and `coordinates` will be assumed to be geographic
        coordinates in degrees.

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
    >>> print(inside((east, north), region))
    [False False  True  True  True False]
    >>> # This also works for 2D-arrays
    >>> east = np.array([[1, 1, 1],
    ...                  [2, 2, 2],
    ...                  [3, 3, 3]])
    >>> north = np.array([[5, 7, 9],
    ...                   [5, 7, 9],
    ...                   [5, 7, 9]])
    >>> region = [0.5, 2.5, 6, 9]
    >>> print(inside((east, north), region))
    [[False  True  True]
     [False  True  True]
     [False False False]]
    >>> # Geographic coordinates are also supported
    >>> east, north = grid_coordinates([0, 350, -20, 20], spacing=10)
    >>> region = [-10, 10, -10, 10]
    >>> are_inside = inside([east, north], region, latlon=True)
    >>> print(east[are_inside])
    [  0.  10. 350.   0.  10. 350.   0.  10. 350.]
    >>> print(north[are_inside])
    [-10. -10. -10.   0.   0.   0.  10.  10.  10.]

    """
    w, e, s, n = region
    easting, northing = coordinates[:2]
    if latlon:
        w, e, easting = _latlon_continuity(w, e, easting)
        region = [w, e, s, n]
    check_region(region, latlon=latlon)
    # Allocate temporary arrays to minimize memory allocation overhead
    out = np.empty_like(easting, dtype=np.bool)
    tmp = tuple(np.empty_like(easting, dtype=np.bool) for i in range(4))
    # Using the logical functions is a lot faster than & > < for some reason
    # Plus, this way avoids repeated allocation of intermediate arrays
    in_we = np.logical_and(
        np.greater_equal(easting, w, out=tmp[0]),
        np.less_equal(easting, e, out=tmp[1]),
        out=tmp[2],
    )
    in_ns = np.logical_and(
        np.greater_equal(northing, s, out=tmp[0]),
        np.less_equal(northing, n, out=tmp[1]),
        out=tmp[3],
    )
    are_inside = np.logical_and(in_we, in_ns, out=out)
    return are_inside


def block_split(coordinates, spacing, adjust="spacing", region=None):
    """
    Split a region into blocks and label points according to where they fall.

    The labels are integers corresponding to the index of the block. The same
    index is used for the coordinates of each block.

    .. note::

        If installed, package ``pykdtree`` will be used instead of
        :class:`scipy.spatial.cKDTree` for better performance.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    spacing : float, tuple = (s_north, s_east), or None
        The block size in the South-North and West-East directions,
        respectively. A single value means that the size is equal in both
        directions.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if
        *shape* is given instead of *spacing*. Defaults to adjusting the
        spacing.
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates. If not region is given, will use the bounding region of
        the given points.

    Returns
    -------
    block_coordinates : tuple of arrays
        (easting, northing) arrays with the coordinates of the center of each
        block.
    labels : array
        integer label for each data point. The label is the index of the block
        to which that point belongs.

    See also
    --------
    BlockReduce : Apply a reduction operation to the data in blocks (windows).

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> coords = grid_coordinates((-5, 0, 5, 10), spacing=1)
    >>> block_coords, labels = block_split(coords, spacing=2.5)
    >>> for coord in block_coords:
    ...     print(', '.join(['{:.2f}'.format(i) for i in coord]))
    -3.75, -1.25, -3.75, -1.25
    6.25, 6.25, 8.75, 8.75
    >>> print(labels.reshape(coords[0].shape))
    [[0 0 0 1 1 1]
     [0 0 0 1 1 1]
     [0 0 0 1 1 1]
     [2 2 2 3 3 3]
     [2 2 2 3 3 3]
     [2 2 2 3 3 3]]

    """
    if region is None:
        region = get_region(coordinates)
    block_coords = tuple(
        i.ravel()
        for i in grid_coordinates(
            region, spacing=spacing, adjust=adjust, pixel_register=True
        )
    )
    tree = kdtree(block_coords)
    labels = tree.query(np.transpose(n_1d_arrays(coordinates, 2)))[1]
    return block_coords, labels


def _latlon_continuity(west, east, longitude_coords):
    """
    Modify longitudinal geographic coordinates to ensure continuity around the globe.
    """
    # Check if region is defined all around the globe
    all_globe = bool((east - west) % 360 == 0 and east != west)
    # Move coordinates to [0, 360]
    west = west % 360
    east = east % 360
    longitude_coords = longitude_coords % 360
    # Move west=0 and east=360 if region longitudes goes all around the globe
    if all_globe:
        west, east = 0, 360
    # Check if the [-180, 180] interval is better suited
    if west > east:
        east = ((east + 180) % 360) - 180
        west = ((west + 180) % 360) - 180
        longitude_coords = ((longitude_coords + 180) % 360) - 180
    return west, east, longitude_coords
