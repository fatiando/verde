"""
General utilities for dealing with grids and point generation.
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
    region : list
        The boundaries (``[W, E, S, N]``) of a given region in Cartesian or
        geographic coordinates.

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


def scatter_points(region, size, random_state=None):
    """
    Generate the coordinates for a random scatter of points.

    The points are drawn from a uniform distribution.

    Parameters
    ----------
    region : list
        The boundaries (``[W, E, S, N]``) of a given region in Cartesian or
        geographic coordinates.
    size : int
        The number of points to generate.
    random_state : numpy.random.RandomState or an int seed
        A random number generator used to define the state of the random
        permutations. Use a fixes seed to make sure computations are
        reproducible.


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

    """
    check_region(region)
    random = check_random_state(random_state)
    w, e, s, n = region
    easting = random.uniform(w, e, size)
    northing = random.uniform(s, n, size)
    return easting, northing


def grid_coordinates(region, shape):
    """
    Generate the coordinates for each point on a regular grid.

    Parameters
    ----------
    region : list
        The boundaries (``[W, E, S, N]``) of a given region in Cartesian or
        geographic coordinates.
    shape : tuple
        The number of points in South-North and West-East directions,
        respectively.

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
    >>> for line in east:
    ...     print(', '.join(['{:1.1f}'.format(i) for i in line]))
    0.0, 2.5, 5.0
    0.0, 2.5, 5.0
    0.0, 2.5, 5.0
    0.0, 2.5, 5.0
    0.0, 2.5, 5.0
    >>> for line in north:
    ...     print(', '.join(['{:>4.1f}'.format(i) for i in line]))
     0.0,  0.0,  0.0
     2.5,  2.5,  2.5
     5.0,  5.0,  5.0
     7.5,  7.5,  7.5
    10.0, 10.0, 10.0

    """
    check_region(region)
    w, e, s, n = region
    nnorth, neast = shape
    east_lines = np.linspace(w, e, neast)
    north_lines = np.linspace(s, n, nnorth)
    easting, northing = np.meshgrid(east_lines, north_lines)
    return easting, northing
