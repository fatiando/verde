# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Nearest neighbor interpolation
"""
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input, n_1d_arrays
from .coordinates import get_region
from .utils import kdtree


class KNeighbors(BaseGridder):
    """
    Nearest neighbor interpolation.

    This gridder assumes Cartesian coordinates.

    Interpolation based on the values of the *k* nearest neighbors of each
    interpolated point. The number of neighbors *k* can be controlled and
    mostly influences the spatial smoothness of the interpolated values.

    The data values of the *k* nearest neighbors are combined into a single
    value by a reduction function, which defaults to the mean. This can also be
    configured.

    Parameters
    ----------
    k : int
        The number of neighbors to use for each interpolated point. Default is
        1.
    reduction : function
        Function used to combine the values of the *k* neighbors into a single
        value. Can be any function that takes a 1D numpy array as input and
        outputs a single value. Default is :func:`numpy.mean`.

    Attributes
    ----------
    tree_ : K-D tree
        An instance of the K-D tree data structure for the data points that is
        used to query for nearest neighbors.
    data_ : 1D array
        A copy of the input data as a 1D array. Used to look up values for
        interpolation/prediction.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.KNeighbors.grid`` method.

    """

    def __init__(self, k=1, reduction=np.mean):
        super().__init__()
        self.k = k
        self.reduction = reduction

    def fit(self, coordinates, data, weights=None):
        """
        Fit the interpolator to the given data.

        The data region is captured and used as default for the
        :meth:`~verde.KNeighbors.grid` method.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : array
            The data values that will be interpolated.
        weights : None or array
            Data weights are **not supported** by this interpolator and will be
            ignored. Only present for compatibility with other gridders.

        Returns
        -------
        self
            Returns this gridder instance for chaining operations.

        """
        if weights is not None:
            warnings.warn(
                "{} does not support weights and they will be ignored.".format(
                    self.__class__.__name__
                ),
                stacklevel=2,
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self.region_ = get_region(coordinates[:2])
        self.tree_ = kdtree(coordinates[:2])
        # Make sure this is an array and not a subclass of array (pandas,
        # xarray, etc) so that we can index it later during predict.
        self.data_ = np.asarray(data).ravel().copy()
        return self

    def predict(self, coordinates):
        """
        Interpolate data on the given set of points.

        Requires a fitted gridder (see :meth:`~verde.KNeighbors.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : array
            The data values interpolated on the given points.

        """
        check_is_fitted(self, ["tree_"])
        distances, indices = self.tree_.query(
            np.transpose(n_1d_arrays(coordinates, 2)), k=self.k
        )
        if indices.ndim == 1:
            indices = np.atleast_2d(indices).T
        neighbor_values = np.reshape(self.data_[indices.ravel()], indices.shape)
        data = self.reduction(neighbor_values, axis=1)
        shape = np.broadcast(*coordinates[:2]).shape
        return data.reshape(shape)
