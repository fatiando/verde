"""
Biharmonic splines in 2D.
"""
from warnings import warn

import numpy as np
import scipy.linalg as spla
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from .base import BaseGridder, check_fit_input
from .coordinates import grid_coordinates, get_region


class Spline(BaseGridder):
    """
    """

    def __init__(self, fudge=1e-5, damping=None, shape=None, spacing=None):
        self.damping = damping
        self.shape = shape
        self.spacing = spacing
        self.fudge = fudge

    def fit(self, coordinates, data, weights=None):
        """
        """
        coordinates, data, weights = check_fit_input(coordinates, data,
                                                     weights)
        exact_solution = all(i is None
                             for i in [self.damping, self.spacing, self.shape])
        if weights is not None and exact_solution:
            warn(' '.join([
                "Weights are ignored for the exact spline solution.",
                "Use damping or specify a shape/spacing for the forces."]))
        easting, northing = coordinates[:2]
        self.region_ = get_region(easting, northing)
        # Set the force positions. If no shape or spacing are given, then they
        # will coincide with the data points.
        if self.shape is None and self.spacing is None:
            self.force_coords_ = (easting.copy(), northing.copy())
        else:
            self.force_coords_ = grid_coordinates(
                self.region_, shape=self.shape, spacing=self.spacing)
        jac = spline_jacobian(easting, northing, self.force_coords_[0],
                              self.force_coords_[1], self.fudge)
        scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
        jac = scaler.fit_transform(jac)
        if self.damping is None:
            regr = LinearRegression(fit_intercept=False, normalize=False)
        else:
            regr = Ridge(alpha=self.damping, fit_intercept=False,
                         normalize=False)
        regr.fit(jac, data.ravel(), sample_weight=weights)
        self.residual_ = data - jac.dot(regr.coef_).reshape(data.shape)
        self.force_ = regr.coef_/scaler.scale_
        return self

    def predict(self, coordinates):
        """
        """
        easting, northing = coordinates[:2]
        check_is_fitted(self, ['force_', 'force_coords_'])
        jac = spline_jacobian(easting.ravel(), northing.ravel(),
                              self.force_coords_[0], self.force_coords_[1],
                              self.fudge)
        shape = np.broadcast(easting, northing).shape
        return jac.dot(self.force_).reshape(shape)


def spline_jacobian(easting, northing, force_easting, force_northing,
                    fudge=1e-5):
    """
    """
    easting = np.atleast_1d(easting)
    northing = np.atleast_1d(northing)
    force_easting = np.atleast_1d(force_easting)
    force_northing = np.atleast_1d(force_northing)
    # Reshaping the data to a column vector will automatically build a
    # Green's function matrix because of the array broadcasting.
    distance = np.hypot(
        easting.reshape((easting.size, 1)) - force_easting.ravel(),
        northing.reshape((northing.size, 1)) - force_northing.ravel())
    # The fudge factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += fudge
    return (distance**2)*(np.log(distance) - 1)
