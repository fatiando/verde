# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gridders that use scipy.interpolate as the backend.
"""
from abc import abstractmethod
from warnings import warn

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input
from .coordinates import get_region


class _BaseScipyGridder(BaseGridder):
    """
    A scipy.interpolate base gridder for scalar Cartesian data.

    Used as a base class for each of the SciPy ND based interpolators.

    Attributes
    ----------
    interpolator_ : scipy interpolator class
        An instance of the corresponding scipy interpolator class.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.ScipyGridder.grid` and
        :meth:`~verde.ScipyGridder.scatter` methods.

    """

    @abstractmethod
    def _get_interpolator(self):
        """
        Return the SciPy interpolator class and any extra keyword arguments as
        a dictionary.
        """

    def fit(self, coordinates, data, weights=None):
        """
        Fit the interpolator to the given data.

        The data region is captured and used as default for the
        :meth:`~verde._BaseScipyGridder.grid` method.

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
            warn(
                "{} does not support weights and they will be ignored.".format(
                    self.__class__.__name__
                )
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        easting, northing = coordinates[:2]
        self.region_ = get_region((easting, northing))
        points = np.column_stack((np.ravel(easting), np.ravel(northing)))
        interpolator_class, kwargs = self._get_interpolator()
        self.interpolator_ = interpolator_class(points, np.ravel(data), **kwargs)
        return self

    def predict(self, coordinates):
        """
        Interpolate data on the given set of points.

        Requires a fitted gridder (see :meth:`~verde._BaseScipyGridder.fit`).

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
        check_is_fitted(self, ["interpolator_"])
        easting, northing = coordinates[:2]
        return self.interpolator_((easting, northing))


class Linear(_BaseScipyGridder):
    """
    Piecewise linear interpolation.

    Provides a Verde interface to
    :class:`scipy.interpolate.LinearNDInterpolator`.

    Parameters
    ----------
    rescale : bool
        If ``True``, rescale the data coordinates to [0, 1] range before
        interpolation. Useful when coordinates vary greatly in scale. Default
        is ``False``.

    Attributes
    ----------
    interpolator_ : scipy interpolator class
        An instance of the corresponding scipy interpolator class.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Linear.grid` method.

    """

    def __init__(self, rescale=False):
        super().__init__()
        self.rescale = rescale

    def _get_interpolator(self):
        """
        Return the SciPy interpolator class and any extra keyword arguments as
        a dictionary.
        """
        return LinearNDInterpolator, {"rescale": self.rescale}


class Cubic(_BaseScipyGridder):
    """
    Piecewise cubic interpolation.

    Provides a Verde interface to
    :class:`scipy.interpolate.CloughTocher2DInterpolator`.

    Parameters
    ----------
    rescale : bool
        If ``True``, rescale the data coordinates to [0, 1] range before
        interpolation. Useful when coordinates vary greatly in scale. Default
        is ``False``.

    Attributes
    ----------
    interpolator_ : scipy interpolator class
        An instance of the corresponding scipy interpolator class.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Cubic.grid` method.

    """

    def __init__(self, rescale=False):
        super().__init__()
        self.rescale = rescale

    def _get_interpolator(self):
        """
        Return the SciPy interpolator class and any extra keyword arguments as
        a dictionary.
        """
        return CloughTocher2DInterpolator, {"rescale": self.rescale}
