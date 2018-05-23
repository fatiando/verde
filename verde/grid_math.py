"""
Operations on spatial data: block operations, derivatives, etc.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from sklearn.base import BaseEstimator

from .coordinates import get_region, grid_coordinates
from .base import check_fit_input


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
    easting, northing = coordinates[:2]
    if region is None:
        region = get_region((easting, northing))
    block_coords = tuple(
        i.ravel() for i in grid_coordinates(
            region, spacing=spacing, adjust=adjust, pixel_register=True)
    )
    # The index of the block with the closest center to each data point
    tree = cKDTree(np.transpose(block_coords))
    labels = tree.query(np.transpose((easting.ravel(), northing.ravel())))[1]
    return block_coords, labels


class BlockReduce(BaseEstimator):
    """
    Apply a reduction/aggregation operation to the data in blocks (windows).

    Returns the reduced data value for each block along with the associated
    coordinates, which can be determined through the same reduction applied to
    the coordinates or as the center of each block.

    If a data region to be divided into blocks is not given, it will be the
    bounding region of the data. When using this class to decimate data before
    gridding, it's best to use the same region and spacing as the desired grid.

    If the given region is not divisible by the spacing (block size), either
    the region or the spacing will have to be adjusted. By default, the spacing
    will be rounded to the nearest multiple. Optionally, the East and North
    boundaries of the region can be adjusted to fit the exact spacing given.

    Blocks without any data are omitted from the output.

    Implements the :meth:`~verde.BlockReduce.filter` method so it can be used
    with :class:`verde.Chain`. Only acts during data fitting and is ignored
    during prediction.

    Parameters
    ----------
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

    See also
    --------
    block_split : Split a region into blocks and label points accordingly.
    verde.Chain : Apply filter operations successively on data.

    """

    def __init__(self, reduction, spacing, region=None, adjust='spacing',
                 center_coordinates=False, std=False):
        self.reduction = reduction
        self.spacing = spacing
        self.region = region
        self.adjust = adjust
        self.center_coordinates = center_coordinates
        self.std = std

    def filter(self, coordinates, data, weights=None):
        """
        Apply the blocked aggregation to the given data.

        Returns the reduced data value for each block along with the associated
        coordinates, which can be determined through the same reduction applied
        to the coordinates or as the center of each block.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : array
            The data values at each point.
        weights : None or array or tuple of arrays
            If not None, then the weights assigned to each data point. If more
            than one data component is provided, you must provide a weights
            array for each data component (if not None).

        Returns
        -------
        blocked_coordinates : tuple of arrays
            (easting, northing) arrays with the coordinates of each block that
            contains data.
        blocked_data : array
            The block reduced data values.

        """
        coordinates, data, weights = check_fit_input(coordinates, data,
                                                     weights, ravel=True)
        easting, northing = coordinates[:2]
        block_coords, labels = block_split((easting, northing), self.spacing,
                                           self.adjust, self.region)
        if self.center_coordinates:
            unique = np.unique(labels)
            blocked_coords = tuple(i[unique] for i in block_coords)
        else:
            # Doing the coordinates separately because in case of weights the
            # reduction applied to then is different (no weights ever)
            coords = (
                pd.DataFrame(
                    dict(easting=easting.ravel(), northing=northing.ravel(),
                         block=labels)
                ).groupby('block').aggregate(self.reduction))
            blocked_coords = (coords.easting.values, coords.northing.values)
        # if any(w is None for w in weights):
        if weights is None:
            table = pd.DataFrame(dict(data=data.ravel(), block=labels))
            blocked = table.groupby('block')
            blocked_data = blocked.aggregate(self.reduction).data.values
            if self.std:
                blocked_weights = blocked.aggregate(np.std).data.values
            else:
                blocked_weights = None
        else:
            table = pd.DataFrame(
                dict(data=data.ravel(), weights=weights.ravel(),
                     block=labels))
            blocked = table.groupby('block')

            def reduction(value):
                w = table.loc[value.index, "weights"]
                return self.reduction(value, weights=w)

            def variance(w):
                if w.size < 2:
                    return w
                value = table.loc[w.index, "data"]
                mean = np.average(value, weights=w)
                var = np.average((value - mean)**2, weights=w)
                # v1 = w.sum()
                # v2 = (w**2).sum()
                # var *= v1/(v1 - v2/v1)
                return var

            agg = blocked.aggregate(dict(data=reduction, weights=variance))
            blocked_data = agg.data.values.ravel()
            # blocked_weights = (agg.weights.min()/agg.weights).values.ravel()
            blocked_weights = (agg.weights).values.ravel()

        if blocked_weights is None:
            return blocked_coords, blocked_data
        return blocked_coords, blocked_data, blocked_weights




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
