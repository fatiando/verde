"""
A gridder that uses scipy.interpolate as the backend.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, \
    CloughTocher2DInterpolator

from .base import BaseGridder


class ScipyGridder(BaseGridder):
    """
    A scipy.interpolate based gridder.

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
        arguments..

    Examples
    --------


    """

    def __init__(self, method='cubic', extra_args=None):
        self.method = method
        self.extra_args = extra_args

    def fit(self, easting, northing, data):
        classes = dict(linear=LinearNDInterpolator,
                       nearest=NearestNDInterpolator,
                       cubic=CloughTocher2DInterpolator)
        if self.method not in classes:
            raise ValueError(
                "Invalid interpolation method '{}'. Must be one of {}."
                .format(method, str(classes.keys())))
        if self.extra_args is None:
            kwargs = {}
        else:
            kwargs = self.extra_args
        self.region_ = (np.min(easting), np.max(easting),
                        np.min(northing), np.max(northing))
        points = np.column_stack((np.ravel(easting), np.ravel(northing)))
        self.interpolator_ = classes[self.method](points, np.ravel(data),
                                                  **kwargs)
        return self

    def predict(self, easting, northing):
        check_is_fitted(self, ['interpolator_'])
        return self.interpolator_((easting, northing))
