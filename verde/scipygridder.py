"""
A gridder that uses scipy.interpolate as the backend.
"""
from warnings import warn

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    CloughTocher2DInterpolator,
)
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input
from .coordinates import get_region


class ScipyGridder(BaseGridder):
    """
    A scipy.interpolate based gridder for scalar Cartesian data.

    Provides a verde gridder interface to the scipy interpolators
    :class:`scipy.interpolate.LinearNDInterpolator`,
    :class:`scipy.interpolate.NearestNDInterpolator`, and
    :class:`scipy.interpolate.CloughTocher2DInterpolator` (cubic).

    Parameters
    ----------
    method : str
        The interpolation method. Either ``'linear'``, ``'nearest'``, or
        ``'cubic'``.
    extra_args : None or dict
        Extra keyword arguments to pass to the scipy interpolator class. See
        the documentation for each interpolator for a list of possible
        arguments.

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

    def __init__(self, method="cubic", extra_args=None):
        super().__init__()
        self.method = method
        self.extra_args = extra_args

    def fit(self, coordinates, data, weights=None):
        """
        Fit the interpolator to the given data.

        Any keyword arguments passed as the ``extra_args`` attribute will be
        used when instantiating the scipy interpolator.

        The data region is captured and used as default for the
        :meth:`~verde.ScipyGridder.grid` and
        :meth:`~verde.ScipyGridder.scatter` methods.

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
            If not None, then the weights assigned to each data point.
            Typically, this should be 1 over the data uncertainty squared.
            Ignored for this interpolator. Only present for compatibility with
            other gridder.

        Returns
        -------
        self : verde.ScipyGridder
            Returns this gridder instance for chaining operations.

        """
        classes = dict(
            linear=LinearNDInterpolator,
            nearest=NearestNDInterpolator,
            cubic=CloughTocher2DInterpolator,
        )
        if self.method not in classes:
            raise ValueError(
                "Invalid interpolation method '{}'. Must be one of {}.".format(
                    self.method, str(classes.keys())
                )
            )
        if self.extra_args is None:
            kwargs = {}
        else:
            kwargs = self.extra_args
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
        self.interpolator_ = classes[self.method](points, np.ravel(data), **kwargs)
        return self

    def predict(self, coordinates):
        """
        Interpolate data on the given set of points.

        Requires a fitted gridder (see :meth:`~verde.ScipyGridder.fit`).

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
