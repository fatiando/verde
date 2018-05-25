"""
Classes for reducing/aggregating data in blocks.
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


def attach_weights(reduction, weights):
    """
    Create a partial application of reduction with the proper weights attached.

    Makes a function that calls *reduction* and gives it the weights
    corresponding to the index of the particular values it receives. Meant for
    used in a groupby aggregation of a pandas.DataFrame. See class BlockReduce.
    """

    def weighted_reduction(values):
        "weighted reduction using the stored from the outer scope weights"
        w = weights[values.index]
        return reduction(values, weights=w)

    return weighted_reduction


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
                 center_coordinates=False):
        self.reduction = reduction
        self.spacing = spacing
        self.region = region
        self.adjust = adjust
        self.center_coordinates = center_coordinates

    def filter(self, coordinates, data, weights=None):
        """
        Apply the blocked aggregation to the given data.

        Returns the reduced data value for each block along with the associated
        coordinates, which can be determined through the same reduction applied
        to the coordinates or as the center of each block.

        If weights are given, the reduction function must accept a ``weights``
        keyword argument. The weights are passed in to the reduction but we
        have no generic way aggregating the weights or reporting uncertainties.
        For that, look to the specialized classes like
        :class:`verde.BlockMean`.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : array or tuple of arrays
            The data values at each point. If you want to reduce more than one
            data component, pass in multiple arrays as elements of a tuple. All
            arrays must have the same shape.
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
                                                     weights, ravel=False)
        blocks, labels = block_split(coordinates, self.spacing, self.adjust,
                                     self.region)
        columns = {'data{}'.format(i): comp.ravel()
                   for i, comp in enumerate(data)}
        columns['block'] = labels
        if any(w is None for w in weights):
            reduction = self.reduction
        else:
            columns.update({'weight{}'.format(i): comp.ravel()
                            for i, comp in enumerate(weights)})
            reduction = {'data{}'.format(i): attach_weights(self.reduction, w)
                         for i, w in enumerate(weights)}
        blocked = pd.DataFrame(columns).groupby('block').aggregate(reduction)
        blocked_data = tuple(blocked['data{}'.format(i)].values.ravel()
                             for i, _ in enumerate(data))
        blocked_coords = self._block_coordinates(coordinates, blocks, labels)
        if len(blocked_data) == 1:
            return blocked_coords, blocked_data[0]
        return blocked_coords, blocked_data

    def _block_coordinates(self, coordinates, block_coordinates, labels):
        """
        Calculate a coordinate assigned to each block.

        If self.center_coordinates, the coordinates will be the center of each
        block. Otherwise, will apply the reduction to the coordinates.

        Blocks without any data will be omitted.

        *block_coordinates* and *labels* should be the outputs of
        :func:`verde.block_split`.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        block_coordinates : tuple of arrays
            (easting, northing) arrays with the coordinates of the center of
            each block.
        labels : array
            integer label for each data point. The label is the index of the
            block to which that point belongs.

        Returns
        -------
        coordinates : tuple of arrays
            (easting, northing) arrays with the coordinates assigned to each
            non-empty block.

        """
        if self.center_coordinates:
            unique = np.unique(labels)
            return tuple(i[unique] for i in block_coordinates)
        # Doing the coordinates separately from the data because in case of
        # weights the reduction applied to then is different (no weights
        # ever)
        easting, northing = coordinates[:2]
        table = pd.DataFrame(dict(easting=easting.ravel(),
                                  northing=northing.ravel(),
                                  block=labels))
        grouped = table.groupby('block').aggregate(self.reduction)
        return grouped.easting.values, grouped.northing.values
