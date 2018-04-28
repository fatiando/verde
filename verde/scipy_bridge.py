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
    Gridding using scipy.interpolate as the backend.
    """

    def __init__(self, method='linear', extra_args=None):
        self.method = method
        self.extra_args = extra_args

    def fit(self, easting, northing, data):
        classes = dict(linear=LinearNDInterpolator,
                       nearest=NearestNDInterpolator,
                       cubic=CloughTocher2DInterpolator)
        if self.extra_args is None:
            kwargs = {}
        else:
            kwargs = self.extra_args
        points = np.column_stack((easting.ravel(), northing.ravel()))
        self.interpolator_ = classes[self.method](points, data.ravel(),
                                                  **kwargs)
        return self

    def predict(self, easting, northing):
        check_is_fitted(self, ['interpolator_'])
        return self.interpolator_((easting, northing))
