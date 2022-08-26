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


    Attributes
    ----------
    tree_ : K-D tree
        An instance of the K-D tree data structure for the data points that is
        used to query for nearest neighbors.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.KNeighbors.grid`` method.

    """

    def __init__(self, k=1):
        super().__init__()
        self.k = k

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
            ignored. Only present for compatibility with other gridder.

        Returns
        -------
        self
            Returns this gridder instance for chaining operations.

        """
        if weights is not None:
            warnings.warn(
                "{} does not support weights and they will be ignored.".format(
                    self.__class__.__name__
                )
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self.region_ = get_region(coordinates[:2])
        self.tree_ = kdtree(coordinates[:2])
        self.data_ = n_1d_arrays(data, n=1)[0].copy()
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
        data = np.mean(neighbor_values, axis=1)
        shape = np.broadcast(*coordinates[:2]).shape
        return data.reshape(shape)
