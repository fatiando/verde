"""
Operations on spatial data: block operations, derivatives, etc.
"""
import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module


def distance_mask(coordinates, data_coordinates, maxdist):
    """
    Create a mask for points that are too far from the given data points.

    Produces a mask array that is True when a point is more than *maxdist* from
    the closest data point.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each point that will be masked. Should
        be in the following order: (easting, northing, vertical, ...). Only
        easting and northing will be used, all subsequent coordinates will be
        ignored.
    data_coordinates : tuple of arrays
        Same as *coordinates* but for the data points.
    maxdist : float
        The maximum distance that a point can be from the closest data point.

    Returns
    -------
    mask : array
        The mask boolean array with the same shape as *easting* and *northing*.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> coords = grid_coordinates((0, 5, -10, -5), spacing=1)
    >>> mask = distance_mask(coords, (2.5, -7.5), maxdist=2)
    >>> print(mask)
    [[ True  True  True  True  True  True]
     [ True  True False False  True  True]
     [ True False False False False  True]
     [ True False False False False  True]
     [ True  True False False  True  True]
     [ True  True  True  True  True  True]]

    """
    data_easting, data_northing = data_coordinates[:2]
    easting, northing = coordinates[:2]
    data_easting = np.atleast_1d(data_easting)
    data_northing = np.atleast_1d(data_northing)
    data_points = np.transpose((data_easting.ravel(), data_northing.ravel()))
    tree = cKDTree(data_points)
    points = np.transpose((easting.ravel(), northing.ravel()))
    distance = tree.query(points)[0].reshape(easting.shape)
    return distance > maxdist
