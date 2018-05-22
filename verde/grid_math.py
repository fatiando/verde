"""
Operations on spatial data: block operations, derivatives, etc.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from .coordinates import get_region, grid_coordinates


def block_split(coordinates, spacing, adjust='spacing', region=None):
    """
    Split a region into blocks and label points according to where they fall.

    The labels are integers corresponding to the index of the block. The same
    index is used for the coordinates of each block.

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
        Integer label for each data point. The label is the index of the block
        to which that point belongs.

    See also
    --------
    block_reduce : Apply a reduction operation to the data in blocks (windows).

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
    easting, northing = coordinates[:2]
    if region is None:
        region = get_region(easting, northing)
    block_coords = tuple(
        i.ravel() for i in grid_coordinates(
            region, spacing=spacing, adjust=adjust, pixel_register=True)
    )
    # The index of the block with the closest center to each data point
    tree = cKDTree(np.transpose(block_coords))
    labels = tree.query(np.transpose((easting.ravel(), northing.ravel())))[1]
    return block_coords, labels


def block_reduce(easting, northing, data, reduction, spacing, region=None,
                 adjust='spacing', center_coordinates=False):
    """
    Apply a reduction operation to the data in blocks (windows).

    Returns the reduced data value for each block along with the associated
    coordinates, which can be determined through the same reduction applied to
    the coordinates or as the center of each block.

    If a data region to be divided into blocks is not given, it will be the
    bounding region of the data. When using this function to decimate data
    before gridding, it's best to use the same region and spacing as the
    desired grid.

    If the given region is not divisible by the spacing (block size), either
    the region or the spacing will have to be adjusted. By default, the spacing
    will be rounded to the nearest multiple. Optionally, the East and North
    boundaries of the region can be adjusted to fit the exact spacing given.

    Blocks without any data are omitted from the output.

    Parameters
    ----------
    easting : array
        The values of the West-East coordinates of each data point.
    northing : array
        The values of the South-North coordinates of each data point.
    data : array
        The data values at each point.
    reduction : function
        A reduction function that takes an array and returns a single value
        (e.g., ``np.mean``, ``np.median``, etc).
    spacing : float, tuple = (s_north, s_east), or None
        The block size in the South-North and West-East directions,
        respectively. A single value means that the size is equal in both
        directions.
    region : list = [W, E, S, N]
        The boundaries of a given region in Cartesian or geographic
        coordinates.
    adjust : {'spacing', 'region'}
        Whether to adjust the spacing or the region if required. Ignored if
        *shape* is given instead of *spacing*. Defaults to adjusting the
        spacing.
    center_coordinates : bool
        If True, then the returned coordinates correspond to the center of each
        block. Otherwise, the coordinates are calculated by applying the same
        reduction operation to the input coordinates.

    Returns
    -------
    easting, northing, data : arrays
        The reduced coordinates and data values.

    See also
    --------
    block_inside : Split a region into blocks and label points accordingly.

    """
    block_coords, labels = block_split((easting, northing), spacing, adjust,
                                       region)
    if center_coordinates:
        table = pd.DataFrame(dict(data=data.ravel(), block=labels))
        blocked = table.groupby('block').aggregate(reduction)
        unique = np.unique(labels)
        block_east, block_north = [i[unique] for i in block_coords]
    else:
        table = pd.DataFrame(dict(easting=easting.ravel(),
                                  northing=northing.ravel(),
                                  data=data.ravel(),
                                  block=labels))
        blocked = table.groupby('block').aggregate(reduction)
        block_east = blocked.easting.values
        block_north = blocked.northing.values
    return block_east, block_north, blocked.data.values


def distance_mask(easting, northing, data_easting, data_northing, maxdist):
    """
    Create a mask for points that are too far from the given data points.

    Produces a mask array that is True when a point is more than *maxdist* from
    the closest data point.

    Parameters
    ----------
    easting : array
        The West-East coordinates of the points that will be masked.
    northing : array
        The South-North coordinates of the points that will be masked.
    data_easting : array
        The West-East coordinates of the data points.
    data_northing : array
        The South-North coordinates of the data points.
    maxdist : float
        The maximum distance that a point can be from the closest data point.

    Returns
    -------
    mask : array
        The mask boolean array with the same shape as *easting* and *northing*.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> east, north = grid_coordinates((0, 5, -10, -5), spacing=1)
    >>> mask = distance_mask(east, north, 2.5, -7.5, maxdist=2)
    >>> print(mask)
    [[ True  True  True  True  True  True]
     [ True  True False False  True  True]
     [ True False False False False  True]
     [ True False False False False  True]
     [ True  True False False  True  True]
     [ True  True  True  True  True  True]]

    """
    data_easting = np.atleast_1d(data_easting)
    data_northing = np.atleast_1d(data_northing)
    data_points = np.transpose((data_easting.ravel(), data_northing.ravel()))
    tree = cKDTree(data_points)
    points = np.transpose((easting.ravel(), northing.ravel()))
    distance = tree.query(points)[0].reshape(easting.shape)
    return distance > maxdist
