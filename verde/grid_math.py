"""
Operations on spatial data: block operations, derivatives, etc.
"""
import numpy as np
from scipy.spatial import cKDTree

from .coordinates import block_region, inside, get_region


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
    inside : Determine which points fall inside a given region.
    block_region : Divide a region into blocks and yield their boundaries.

    """
    if region is None:
        region = get_region(easting, northing)
    # Allocate the boolean arrays required in 'inside' only once to avoid
    # overhead.
    out = np.empty_like(easting, dtype=np.bool)
    tmp = tuple(np.empty_like(easting, dtype=np.bool) for i in range(4))
    re_east, re_north, re_data = [], [], []
    for block in block_region(region, spacing, adjust):
        inblock = inside(easting, northing, block, out=out, tmp=tmp)
        if np.any(inblock):
            re_data.append(reduction(data[inblock]))
            if center_coordinates:
                re_east.append((block[1] + block[0])/2)
                re_north.append((block[3] + block[2])/2)
            else:
                re_east.append(reduction(easting[inblock]))
                re_north.append(reduction(northing[inblock]))
    return np.array(re_east), np.array(re_north), np.array(re_data)


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

    """
    tree = cKDTree(np.transpose((data_easting.ravel(), data_northing.ravel())))
    points = np.transpose((easting.ravel(), northing.ravel()))
    distance = tree.query(points)[0].reshape(easting.shape)
    return distance > maxdist
