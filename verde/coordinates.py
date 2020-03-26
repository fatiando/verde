"""
Functions for generating and manipulating coordinates.
"""
import warnings

import numpy as np
from sklearn.utils import check_random_state

from .base.utils import n_1d_arrays, check_coordinates
from .utils import kdtree


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
        raise ValueError("Invalid region '{}'. Only 4 values allowed.".format(region))
    w, e, s, n = region
    if w > e:
        raise ValueError(
            "Invalid region '{}' (W, E, S, N). Must have W =< E. ".format(region)
            + "If working with geographic coordinates, don't forget to match geographic"
            + " region with coordinates using 'verde.longitude_continuity'."
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
        If not None, then value(s) of extra coordinate arrays to be generated.
        These extra arrays will have the same *size* as the others but will
        contain a constant value. Will generate an extra array per value given
        in *extra_coords*. Use this to generate arrays of constant heights or
        times, for example, that might be needed to evaluate a gridder.

    Returns
    -------
    coordinates : tuple of arrays
        Arrays with coordinates of each point in the grid. Each array contains
        values for a dimension in the order: easting, northing, vertical, and
        any extra dimensions given in *extra_coords*. All arrays will have the
        specified *size*.

    Examples
    --------

    >>> # We'll use a seed value will ensure that the same will be generated
    >>> # every time.
    >>> easting, northing = scatter_points((0, 10, -2, -1), 4, random_state=0)
    >>> print(', '.join(['{:.4f}'.format(i) for i in easting]))
    5.4881, 7.1519, 6.0276, 5.4488
    >>> print(', '.join(['{:.4f}'.format(i) for i in northing]))
    -1.5763, -1.3541, -1.5624, -1.1082
    >>> easting, northing, height = scatter_points(
    ...     (0, 10, -2, -1), 4, random_state=0, extra_coords=12
    ... )
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
        registered (only if given *spacing* and not *shape*). Default is False.
    extra_coords : None, scalar, or list
        If not None, then value(s) of extra coordinate arrays to be generated.
        These extra arrays will have the same *shape* as the others but will
        contain a constant value. Will generate an extra array per value given
        in *extra_coords*. Use this to generate arrays of constant heights or
        times, for example, that might be needed to evaluate a gridder.

    Returns
    -------
    coordinates : tuple of arrays
        Arrays with coordinates of each point in the grid. Each array contains
        values for a dimension in the order: easting, northing, vertical, and
        any extra dimensions given in *extra_coords*. All arrays will have the
        specified *shape*.

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

    The grid can also be specified using the spacing between points instead of
    the shape:

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

    The spacing can be different for northing and easting, respectively:

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

    If the region can't be divided into the desired spacing, the spacing will
    be adjusted to conform to the region:

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

    You can choose to adjust the East and North boundaries of the region
    instead:

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

    We can optionally generate coordinates for the center of each grid pixel
    instead of the corner (default):

    >>> east, north = grid_coordinates(region=(0, 5, 0, 10), spacing=2.5,
    ...                                pixel_register=True)
    >>> # Raise the printing precision for this example
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> # Notice that the shape is 1 less than when pixel_register=False
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
    >>> east, north = grid_coordinates(region=(0, 5, 0, 10), shape=(4, 2),
    ...                                pixel_register=True)
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

    Generate arrays for other coordinates that have a constant value:

    >>> east, north, height = grid_coordinates(
    ...     region=(0, 5, 0, 10), spacing=2.5, extra_coords=57
    ... )
    >>> print(east.shape, north.shape, height.shape)
    (5, 3) (5, 3) (5, 3)
    >>> print(height)
    [[57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]
     [57. 57. 57.]]
    >>> east, north, height, time = grid_coordinates(
    ...     region=(0, 5, 0, 10), spacing=2.5, extra_coords=[57, 0.1]
    ... )
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
    elif pixel_register:
        # Starts by generating grid-line registered coordinates and shifting
        # them to the center of the pixel. Need 1 more point if given a shape
        # so that we can do that because we discard the last point when
        # shifting the coordinates.
        shape = tuple(i + 1 for i in shape)
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


def shape_to_spacing(region, shape, pixel_register=False):
    """
    Calculate the spacing of a grid given region and shape.

    Parameters
    ----------
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    shape : tuple = (n_north, n_east) or None
        The number of points in the South-North and West-East directions,
        respectively.
    pixel_register : bool
        If True, the coordinates will refer to the center of each grid pixel
        instead of the grid lines. In practice, this means that there will be
        one less element per dimension of the grid when compared to grid line
        registered (only if given *spacing* and not *shape*). Default is False.

    Returns
    -------
    spacing : tuple = (s_north, s_east)
        The grid spacing in the South-North and West-East directions,
        respectively.

    Examples
    --------

    >>> spacing = shape_to_spacing([0, 10, -5, 1], (7, 11))
    >>> print("{:.1f}, {:.1f}".format(*spacing))
    1.0, 1.0
    >>> spacing = shape_to_spacing([0, 10, -5, 1], (14, 11))
    >>> print("{:.1f}, {:.1f}".format(*spacing))
    0.5, 1.0
    >>> spacing = shape_to_spacing([0, 10, -5, 1], (7, 21))
    >>> print("{:.1f}, {:.1f}".format(*spacing))
    1.0, 0.5
    >>> spacing = shape_to_spacing(
    ...     [-0.5, 10.5, -5.5, 1.5], (7, 11), pixel_register=True,
    ... )
    >>> print("{:.1f}, {:.1f}".format(*spacing))
    1.0, 1.0
    >>> spacing = shape_to_spacing(
    ...     [-0.25, 10.25, -5.5, 1.5], (7, 21), pixel_register=True,
    ... )
    >>> print("{:.1f}, {:.1f}".format(*spacing))
    1.0, 0.5

    """
    spacing = []
    for i, n_points in enumerate(reversed(shape)):
        if not pixel_register:
            n_points -= 1
        spacing.append((region[2 * i + 1] - region[2 * i]) / n_points)
    return tuple(reversed(spacing))


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
        If not None, then value(s) of extra coordinate arrays to be generated.
        These extra arrays will have the same *size* as the others but will
        contain a constant value. Will generate an extra array per value given
        in *extra_coords*. Use this to generate arrays of constant heights or
        times, for example, that might be needed to evaluate a gridder.

    Returns
    -------
    coordinates, distances : tuple and 1d array
        The coordinates of points along the straight line and the distances
        from the first point.

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


def inside(coordinates, region):
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

    Geographic coordinates are also supported using
    :func:`verde.longitude_continuity`:

    >>> from verde import longitude_continuity
    >>> east, north = grid_coordinates([0, 350, -20, 20], spacing=10)
    >>> region = [-10, 10, -10, 10]
    >>> are_inside = inside(*longitude_continuity([east, north], region))
    >>> print(east[are_inside])
    [  0.  10. 350.   0.  10. 350.   0.  10. 350.]
    >>> print(north[are_inside])
    [-10. -10. -10.   0.   0.   0.  10.  10.  10.]

    """
    check_region(region)
    w, e, s, n = region
    easting, northing = coordinates[:2]
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


def block_split(coordinates, spacing=None, adjust="spacing", region=None, shape=None):
    """
    Split a region into blocks and label points according to where they fall.

    The labels are integers corresponding to the index of the block. Also
    returns the coordinates of the center of each block (following the same
    index as the labels).

    The size of the blocks can be specified by the *spacing* parameter.
    Alternatively, the number of blocks in the South-North and West-East
    directions can be specified using the *shape* parameter.

    .. note::

        If installed, package ``pykdtree`` will be used instead of
        :class:`scipy.spatial.cKDTree` for better performance.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively.
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
    rolling_window : Select points on a rolling (moving) window.
    expanding_window : Select points on windows of changing size.

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
    >>> # Use the shape instead of the block size
    >>> block_coords, labels = block_split(coords, shape=(4, 2))
    >>> for coord in block_coords:
    ...     print(', '.join(['{:.3f}'.format(i) for i in coord]))
    -3.750, -1.250, -3.750, -1.250, -3.750, -1.250, -3.750, -1.250
    5.625, 5.625, 6.875, 6.875, 8.125, 8.125, 9.375, 9.375
    >>> print(labels.reshape(coords[0].shape))
    [[0 0 0 1 1 1]
     [0 0 0 1 1 1]
     [2 2 2 3 3 3]
     [4 4 4 5 5 5]
     [6 6 6 7 7 7]
     [6 6 6 7 7 7]]

    """
    # Select the coordinates after checking to make sure indexing will still
    # work on the ignored coordinates.
    coordinates = check_coordinates(coordinates)[:2]
    if region is None:
        region = get_region(coordinates)
    block_coords = grid_coordinates(
        region, spacing=spacing, shape=shape, adjust=adjust, pixel_register=True
    )
    tree = kdtree(block_coords)
    labels = tree.query(np.transpose(n_1d_arrays(coordinates, 2)))[1]
    return n_1d_arrays(block_coords, len(block_coords)), labels


def rolling_window(
    coordinates, size, spacing=None, shape=None, region=None, adjust="spacing"
):
    """
    Select points on a rolling (moving) window.

    A window of the given size is moved across the region at a given step
    (specified by *spacing* or *shape*). Returns the indices of points falling
    inside each window step. You can use the indices to select points falling
    inside a given window.

    The size of the step when moving the windows can be specified by the
    *spacing* parameter. Alternatively, the number of windows in the
    South-North and West-East directions can be specified using the *shape*
    parameter. **One of the two must be given.**

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    size : float
        The size of the windows. Units should match the units of *coordinates*.
    spacing : float, tuple = (s_north, s_east), or None
        The window size in the South-North and West-East directions,
        respectively. A single value means that the size is equal in both
        directions.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively.
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates. If not region is given, will use the bounding region of
        the given points.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if
        *shape* is given instead of *spacing*. Defaults to adjusting the
        spacing.

    Returns
    -------
    window_coordinates : tuple of arrays
        Coordinate arrays for the center of each window.
    indices : array
        Each element of the array corresponds the indices of points falling
        inside a window. The array will have the same shape as the
        *window_coordinates*. Use the array elements to index the coordinates
        for each window. The indices will depend on the number of dimensions in
        the input coordinates. For example, if the coordinates are 2D arrays,
        each window will contain indices for 2 dimensions (row, column).

    See also
    --------
    block_split : Split a region into blocks and label points accordingly.
    expanding_window : Select points on windows of changing size.

    Examples
    --------

    Generate a set of sample coordinates on a grid and determine the indices
    of points for each rolling window:

    >>> from verde import grid_coordinates
    >>> coords = grid_coordinates((-5, -1, 6, 10), spacing=1)
    >>> print(coords[0])
    [[-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]]
    >>> print(coords[1])
    [[ 6.  6.  6.  6.  6.]
     [ 7.  7.  7.  7.  7.]
     [ 8.  8.  8.  8.  8.]
     [ 9.  9.  9.  9.  9.]
     [10. 10. 10. 10. 10.]]
    >>> # Get the rolling window indices
    >>> window_coords, indices = rolling_window(coords, size=2, spacing=2)
    >>> # Window coordinates will be 2D arrays. Their shape is the number of
    >>> # windows in each dimension
    >>> print(window_coords[0].shape, window_coords[1].shape)
    (2, 2) (2, 2)
    >>> # The there are the easting and northing coordinates for the center of
    >>> # each rolling window
    >>> for coord in window_coords:
    ...     print(coord)
    [[-4. -2.]
     [-4. -2.]]
    [[7. 7.]
     [9. 9.]]
    >>> # The indices of points falling on each window will have the same shape
    >>> # as the window center coordinates
    >>> print(indices.shape)
    (2, 2)
    >>> # The points in the first window. Indices are 2D positions because the
    >>> # coordinate arrays are 2D.
    >>> print(len(indices[0, 0]))
    2
    >>> for dimension in indices[0, 0]:
    ...     print(dimension)
    [0 0 0 1 1 1 2 2 2]
    [0 1 2 0 1 2 0 1 2]
    >>> for dimension in indices[0, 1]:
    ...     print(dimension)
    [0 0 0 1 1 1 2 2 2]
    [2 3 4 2 3 4 2 3 4]
    >>> for dimension in indices[1, 0]:
    ...     print(dimension)
    [2 2 2 3 3 3 4 4 4]
    [0 1 2 0 1 2 0 1 2]
    >>> for dimension in indices[1, 1]:
    ...     print(dimension)
    [2 2 2 3 3 3 4 4 4]
    [2 3 4 2 3 4 2 3 4]
    >>> # To get the coordinates for each window, use indexing
    >>> print(coords[0][indices[0, 0]])
    [-5. -4. -3. -5. -4. -3. -5. -4. -3.]
    >>> print(coords[1][indices[0, 0]])
    [6. 6. 6. 7. 7. 7. 8. 8. 8.]

    If the coordinates are 1D, the indices will also be 1D:

    >>> coords1d = [coord.ravel() for coord in coords]
    >>> window_coords, indices = rolling_window(coords1d, size=2, spacing=2)
    >>> print(len(indices[0, 0]))
    1
    >>> print(indices[0, 0][0])
    [ 0  1  2  5  6  7 10 11 12]
    >>> print(indices[0, 1][0])
    [ 2  3  4  7  8  9 12 13 14]
    >>> print(indices[1, 0][0])
    [10 11 12 15 16 17 20 21 22]
    >>> print(indices[1, 1][0])
    [12 13 14 17 18 19 22 23 24]
    >>> # The returned indices can be used in the same way as before
    >>> print(coords1d[0][indices[0, 0]])
    [-5. -4. -3. -5. -4. -3. -5. -4. -3.]
    >>> print(coords1d[1][indices[0, 0]])
    [6. 6. 6. 7. 7. 7. 8. 8. 8.]

    By default, the windows will span the entire data region. You can also
    control the specific region you'd like the windows to cover:

    >>> # Coordinates on a larger region but with the same spacing as before
    >>> coords = grid_coordinates((-10, 5, 0, 20), spacing=1)
    >>> # Get the rolling window indices but limited to the region from before
    >>> window_coords, indices = rolling_window(
    ...     coords, size=2, spacing=2, region=(-5, -1, 6, 10),
    ... )
    >>> # The windows should still be in the same place as before
    >>> for coord in window_coords:
    ...     print(coord)
    [[-4. -2.]
     [-4. -2.]]
    [[7. 7.]
     [9. 9.]]
    >>> # And indexing the coordinates should also provide the same result
    >>> print(coords[0][indices[0, 0]])
    [-5. -4. -3. -5. -4. -3. -5. -4. -3.]
    >>> print(coords[1][indices[0, 0]])
    [6. 6. 6. 7. 7. 7. 8. 8. 8.]

    Only the first 2 coordinates are considered (assumed to be the horizontal
    ones). All others will be ignored by the function.

    >>> coords = grid_coordinates((-5, -1, 6, 10), spacing=1, extra_coords=20)
    >>> print(coords[2])
    [[20. 20. 20. 20. 20.]
     [20. 20. 20. 20. 20.]
     [20. 20. 20. 20. 20.]
     [20. 20. 20. 20. 20.]
     [20. 20. 20. 20. 20.]]
    >>> window_coords, indices = rolling_window(coords, size=2, spacing=2)
    >>> # The windows would be the same in this case since coords[2] is ignored
    >>> for coord in window_coords:
    ...     print(coord)
    [[-4. -2.]
     [-4. -2.]]
    [[7. 7.]
     [9. 9.]]
    >>> print(indices.shape)
    (2, 2)
    >>> for dimension in indices[0, 0]:
    ...     print(dimension)
    [0 0 0 1 1 1 2 2 2]
    [0 1 2 0 1 2 0 1 2]
    >>> for dimension in indices[0, 1]:
    ...     print(dimension)
    [0 0 0 1 1 1 2 2 2]
    [2 3 4 2 3 4 2 3 4]
    >>> for dimension in indices[1, 0]:
    ...     print(dimension)
    [2 2 2 3 3 3 4 4 4]
    [0 1 2 0 1 2 0 1 2]
    >>> for dimension in indices[1, 1]:
    ...     print(dimension)
    [2 2 2 3 3 3 4 4 4]
    [2 3 4 2 3 4 2 3 4]
    >>> # The indices can still be used with the third coordinate
    >>> print(coords[0][indices[0, 0]])
    [-5. -4. -3. -5. -4. -3. -5. -4. -3.]
    >>> print(coords[1][indices[0, 0]])
    [6. 6. 6. 7. 7. 7. 8. 8. 8.]
    >>> print(coords[2][indices[0, 0]])
    [20. 20. 20. 20. 20. 20. 20. 20. 20.]

    """
    # Select the coordinates after checking to make sure indexing will still
    # work on the ignored coordinates.
    coordinates = check_coordinates(coordinates)[:2]
    if region is None:
        region = get_region(coordinates)
    # Calculate the region spanning the centers of the rolling windows
    window_region = [
        dimension + (-1) ** (i % 2) * size / 2 for i, dimension in enumerate(region)
    ]
    _check_rolling_window_overlap(window_region, size, shape, spacing)
    centers = grid_coordinates(
        window_region, spacing=spacing, shape=shape, adjust=adjust
    )
    # pykdtree doesn't support query_ball_point yet and we need that
    tree = kdtree(coordinates, use_pykdtree=False)
    # Coordinates must be transposed because the kd-tree wants them as columns
    # of a matrix
    # Use p=inf (infinity norm) to get square windows instead of circular ones
    indices1d = tree.query_ball_point(
        np.transpose(n_1d_arrays(centers, 2)), r=size / 2, p=np.inf
    )
    # Make the indices array the same shape as the center coordinates array.
    # That preserves the information of the number of windows in each
    # dimension. Need to first create an empty array of object type because
    # otherwise numpy tries to use the index tuples as dimensions (even if
    # given ndim=1 explicitly). Can't make it 1D and then reshape because the
    # reshape is ignored for some reason. The workaround is to create the array
    # with the correct shape and assign the values to a raveled view of the
    # array.
    indices = np.empty(centers[0].shape, dtype="object")
    # Need to convert the indices to int arrays because unravel_index doesn't
    # like empty lists but can handle empty integer arrays in case a window has
    # no points inside it.
    indices.ravel()[:] = [
        np.unravel_index(np.array(i, dtype="int"), shape=coordinates[0].shape)
        for i in indices1d
    ]
    return centers, indices


def _check_rolling_window_overlap(region, size, shape, spacing):
    """
    Warn the user if there is no overlap between neighboring windows.
    """
    if shape is not None:
        ndims = len(shape)
        dimensions = [region[i * ndims + 1] - region[i * ndims] for i in range(ndims)]
        # The - 1 is because we need to divide by the number of intervals, not
        # the number of nodes.
        spacing = tuple(dim / (n - 1) for dim, n in zip(dimensions, shape))
    spacing = np.atleast_1d(spacing)
    if np.any(spacing > size):
        warnings.warn(
            f"Rolling windows do not overlap (size '{size}' and spacing '{spacing}'). "
            "Some data points may not be included in any window. "
            "Increase size or decrease spacing to avoid this."
        )


def expanding_window(coordinates, center, sizes):
    """
    Select points on windows of changing size around a center point.

    Returns the indices of points falling inside each window.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    center : tuple
        The coordinates of the center of the window. Should be in the
        following order: (easting, northing, vertical, ...).
    sizes : array
        The sizes of the windows. Does not have to be in any particular order.
        The order of indices returned will match the order of window sizes
        given. Units should match the units of *coordinates* and *center*.

    Returns
    -------
    indices : list
        Each element of the list corresponds to  the indices of points falling
        inside a window. Use them to index the coordinates for each window. The
        indices will depend on the number of dimensions in the input
        coordinates. For example, if the coordinates are 2D arrays, each window
        will contain indices for 2 dimensions (row, column).

    See also
    --------
    block_split : Split a region into blocks and label points accordingly.
    rolling_window : Select points on a rolling (moving) window.

    Examples
    --------

    Generate a set of sample coordinates on a grid and determine the indices
    of points for each expanding window:

    >>> from verde import grid_coordinates
    >>> coords = grid_coordinates((-5, -1, 6, 10), spacing=1)
    >>> print(coords[0])
    [[-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]
     [-5. -4. -3. -2. -1.]]
    >>> print(coords[1])
    [[ 6.  6.  6.  6.  6.]
     [ 7.  7.  7.  7.  7.]
     [ 8.  8.  8.  8.  8.]
     [ 9.  9.  9.  9.  9.]
     [10. 10. 10. 10. 10.]]
    >>> # Get the expanding window indices
    >>> indices = expanding_window(coords, center=(-3, 8), sizes=[1, 2, 4])
    >>> # There is one index per window
    >>> print(len(indices))
    3
    >>> # The points in the first window. Indices are 2D positions because the
    >>> # coordinate arrays are 2D.
    >>> print(len(indices[0]))
    2
    >>> for dimension in indices[0]:
    ...     print(dimension)
    [2]
    [2]
    >>> for dimension in indices[1]:
    ...     print(dimension)
    [1 1 1 2 2 2 3 3 3]
    [1 2 3 1 2 3 1 2 3]
    >>> for dimension in indices[2]:
    ...     print(dimension)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
    [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
    >>> # To get the coordinates for each window, use indexing
    >>> print(coords[0][indices[0]])
    [-3.]
    >>> print(coords[1][indices[0]])
    [8.]
    >>> print(coords[0][indices[1]])
    [-4. -3. -2. -4. -3. -2. -4. -3. -2.]
    >>> print(coords[1][indices[1]])
    [7. 7. 7. 8. 8. 8. 9. 9. 9.]

    If the coordinates are 1D, the indices will also be 1D:

    >>> coords1d = [coord.ravel() for coord in coords]
    >>> indices = expanding_window(coords1d, center=(-3, 8), sizes=[1, 2, 4])
    >>> print(len(indices))
    3
    >>> # Since coordinates are 1D, there is only one index
    >>> print(len(indices[0]))
    1
    >>> print(indices[0][0])
    [12]
    >>> print(indices[1][0])
    [ 6  7  8 11 12 13 16 17 18]
    >>> # The returned indices can be used in the same way as before
    >>> print(coords1d[0][indices[0]])
    [-3.]
    >>> print(coords1d[1][indices[0]])
    [8.]

    Only the first 2 coordinates are considered (assumed to be the horizontal
    ones). All others will be ignored by the function.

    >>> coords = grid_coordinates((-5, -1, 6, 10), spacing=1, extra_coords=15)
    >>> print(coords[2])
    [[15. 15. 15. 15. 15.]
     [15. 15. 15. 15. 15.]
     [15. 15. 15. 15. 15.]
     [15. 15. 15. 15. 15.]
     [15. 15. 15. 15. 15.]]
    >>> indices = expanding_window(coords, center=(-3, 8), sizes=[1, 2, 4])
    >>> # The returned indices should be the same as before, ignoring coords[2]
    >>> print(len(indices[0]))
    2
    >>> for dimension in indices[0]:
    ...     print(dimension)
    [2]
    [2]
    >>> for dimension in indices[1]:
    ...     print(dimension)
    [1 1 1 2 2 2 3 3 3]
    [1 2 3 1 2 3 1 2 3]
    >>> for dimension in indices[2]:
    ...     print(dimension)
    [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
    [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
    >>> # The indices can be used to index all 3 coordinates
    >>> print(coords[0][indices[0]])
    [-3.]
    >>> print(coords[1][indices[0]])
    [8.]
    >>> print(coords[2][indices[0]])
    [15.]

    """
    # Select the coordinates after checking to make sure indexing will still
    # work on the ignored coordinates.
    coordinates = check_coordinates(coordinates)[:2]
    shape = coordinates[0].shape
    center = np.atleast_2d(center)
    # pykdtree doesn't support query_ball_point yet and we need that
    tree = kdtree(coordinates, use_pykdtree=False)
    indices = []
    for size in sizes:
        # Use p=inf (infinity norm) to get square windows instead of circular
        index1d = tree.query_ball_point(center, r=size / 2, p=np.inf)[0]
        # Convert indices to an array to avoid errors when the index is empty
        # (no points in the window). unravel_index doesn't like empty lists.
        indices.append(np.unravel_index(np.array(index1d, dtype="int"), shape=shape))
    return indices


def longitude_continuity(coordinates, region):
    """
    Modify coordinates and region boundaries to ensure longitude continuity.

    Longitudinal boundaries of the region are moved to the ``[0, 360)`` or
    ``[-180, 180)`` degrees interval depending which one is better suited for
    that specific region.

    Parameters
    ----------
    coordinates : list or array
        Set of geographic coordinates that will be moved to the same degrees
        interval as the one of the modified region.
    region : list or array
        List or array containing the boundary coordinates `w`, `e`, `s`, `n` of
        the region in degrees.

    Returns
    -------
    modified_coordinates : array
        Modified set of extra geographic coordinates.
    modified_region : array
        List containing the modified boundary coordinates `w, `e`, `s`, `n` of
        the region.

    Examples
    --------

    >>> # Modify region with west > east
    >>> w, e, s, n = 350, 10, -10, 10
    >>> print(longitude_continuity(coordinates=None, region=[w, e, s, n]))
    [-10  10 -10  10]
    >>> # Modify region and extra coordinates
    >>> from verde import grid_coordinates
    >>> region = [-70, -60, -40, -30]
    >>> coordinates = grid_coordinates([270, 320, -50, -20], spacing=5)
    >>> [longitude, latitude], region = longitude_continuity(
    ...     coordinates, region
    ... )
    >>> print(region)
    [290 300 -40 -30]
    >>> print(longitude.min(), longitude.max())
    270.0 320.0
    >>> # Another example
    >>> region = [-20, 20, -20, 20]
    >>> coordinates = grid_coordinates([0, 350, -90, 90], spacing=10)
    >>> [longitude, latitude], region = longitude_continuity(
    ...     coordinates, region
    ... )
    >>> print(region)
    [-20  20 -20  20]
    >>> print(longitude.min(), longitude.max())
    -180.0 170.0
    """
    # Get longitudinal boundaries and check region
    w, e, s, n = region[:4]
    # Run sanity checks for region
    _check_geographic_region([w, e, s, n])
    # Check if region is defined all around the globe
    all_globe = np.allclose(abs(e - w), 360)
    # Move coordinates to [0, 360)
    interval_360 = True
    w = w % 360
    e = e % 360
    # Move west=0 and east=360 if region longitudes goes all around the globe
    if all_globe:
        w, e = 0, 360
    # Check if the [-180, 180) interval is better suited
    if w > e:
        interval_360 = False
        e = ((e + 180) % 360) - 180
        w = ((w + 180) % 360) - 180
    region = np.array(region)
    region[:2] = w, e
    # Modify extra coordinates if passed
    if coordinates:
        # Run sanity checks for coordinates
        _check_geographic_coordinates(coordinates)
        longitude = coordinates[0]
        if interval_360:
            longitude = longitude % 360
        else:
            longitude = ((longitude + 180) % 360) - 180
        coordinates = np.array(coordinates)
        coordinates[0] = longitude
        return coordinates, region
    return region


def _check_geographic_coordinates(coordinates):
    "Check if geographic coordinates are within accepted degrees intervals"
    longitude, latitude = coordinates[:2]
    if np.any(longitude > 360) or np.any(longitude < -180):
        raise ValueError(
            "Invalid longitude coordinates. They should be < 360 and > -180 degrees."
        )
    if np.any(latitude > 90) or np.any(latitude < -90):
        raise ValueError(
            "Invalid latitude coordinates. They should be < 90 and > -90 degrees."
        )


def _check_geographic_region(region):
    """
    Check if region is in geographic coordinates are within accepted intervals.
    """
    w, e, s, n = region[:4]
    # Check if coordinates are within accepted degrees intervals
    if np.any(np.array([w, e]) > 360) or np.any(np.array([w, e]) < -180):
        raise ValueError(
            "Invalid region '{}' (W, E, S, N). ".format(region)
            + "Longitudinal coordinates should be < 360 and > -180 degrees."
        )
    if np.any(np.array([s, n]) > 90) or np.any(np.array([s, n]) < -90):
        raise ValueError(
            "Invalid region '{}' (W, E, S, N). ".format(region)
            + "Latitudinal coordinates should be < 90 and > -90 degrees."
        )
    # Check if longitude boundaries do not involve more than one spin around
    # the globe
    if abs(e - w) > 360:
        raise ValueError(
            "Invalid region '{}' (W, E, S, N). ".format(region)
            + "East and West boundaries must not be separated by an angle greater "
            + "than 360 degrees."
        )
