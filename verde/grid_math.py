"""
Operations on spatial data: block operations, derivatives, etc.
"""
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from .coordinates import grid_coordinates


def distance_mask(data_coordinates, maxdist, coordinates=None, region=None,
                  spacing=None, shape=None, **kwargs):
    """
    Create a mask for points that are too far from the given data points.

    Produces a mask array that is False when a point is more than *maxdist*
    from the closest data point and True otherwise.

    The points that will be masked can be specified by their coordinates or by
    a region and shape/spacing, in which case a regular grid will be generated
    for the mask. If specifying a region, extra keyword arguments to this
    function will be passed along to :func:`verde.grid_coordinates`.

    Parameters
    ----------
    data_coordinates : tuple of arrays
        Same as *coordinates* but for the data points.
    maxdist : float
        The maximum distance that a point can be from the closest data point.
    coordinates : None or tuple of arrays
        Arrays with the coordinates of each point that will be masked. Should
        be in the following order: (easting, northing, vertical, ...). Only
        easting and northing will be used, all subsequent coordinates will be
        ignored. If not given, the *region* and *spacing* or *shape* must be
        provided.
    region : list = [W, E, S, N] or None
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    shape : tuple = (n_north, n_east) or None
        The number of points in the South-North and West-East directions,
        respectively.
    spacing : float, tuple = (s_north, s_east), or None
        The grid spacing in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions.

    Returns
    -------
    mask : array
        The mask boolean array with the same shape as *easting* and *northing*.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> region = (0, 5, -10, -5)
    >>> spacing = 1
    >>> coords = grid_coordinates(region, spacing=spacing)
    >>> mask = distance_mask((2.5, -7.5), maxdist=2, coordinates=coords)
    >>> print(mask)
    [[False False False False False False]
     [False False  True  True False False]
     [False  True  True  True  True False]
     [False  True  True  True  True False]
     [False False  True  True False False]
     [False False False False False False]]
    >>> mask = distance_mask((3.5, -7.5), maxdist=2, region=region,
    ...                      spacing=spacing)
    >>> print(mask)
    [[False False False False False False]
     [False False False  True  True False]
     [False False  True  True  True  True]
     [False False  True  True  True  True]
     [False False False  True  True False]
     [False False False False False False]]

    """
    if coordinates is None:
        if region is None:
            raise ValueError("Either coordinates or region and shape/spacing "
                             "must be given to generate the mask.")
        coordinates = grid_coordinates(region=region, spacing=spacing,
                                       shape=shape, **kwargs)
    data_easting, data_northing = data_coordinates[:2]
    easting, northing = coordinates[:2]
    data_easting = np.atleast_1d(data_easting)
    data_northing = np.atleast_1d(data_northing)
    data_points = np.transpose((data_easting.ravel(), data_northing.ravel()))
    tree = cKDTree(data_points)
    points = np.transpose((easting.ravel(), northing.ravel()))
    distance = tree.query(points)[0].reshape(easting.shape)
    return distance <= maxdist
