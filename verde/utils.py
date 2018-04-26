"""
General utilities for dealing with grids and point generation.
"""
from numpy.random import RandomState


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


def scatter_points(region, size, random_seed=None):
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
    random_seed : None or int
        Value used to seed the random number generator.

    Returns
    -------
    easting, northing : 1d arrays
        The West-East and South-North coordinates of each point.

    Examples
    --------

    >>> # We'll use a seed value will ensure that the same will be generated
    >>> # every time.
    >>> easting, northing = scatter_points((0, 10, -2, -1), 4, random_seed=0)
    >>> print(', '.join(['{:.4f}'.format(i) for i in easting]))
    5.4881, 7.1519, 6.0276, 5.4488
    >>> print(', '.join(['{:.4f}'.format(i) for i in northing]))
    -1.5763, -1.3541, -1.5624, -1.1082

    """
    check_region(region)
    random = RandomState(seed=random_seed)
    w, e, s, n = region
    easting = random.uniform(w, e, size)
    northing = random.uniform(s, n, size)
    return easting, northing
