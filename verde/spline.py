"""
Biharmonic splines in 2D.
"""
import numpy as np
import scipy.linalg as spla
from sklearn.utils.validation import check_is_fitted

from .greens_functions import biharmonic_spline2d
from .base import BaseGridder
from . import grid_coordinates, get_region


class BiharmonicSpline(BaseGridder):
    """
    """

    def __init__(self, fudge=1e-5, damping=None, spacing=None):
        self.damping = damping
        self.spacing = spacing
        self.fudge = fudge

    def fit(self, easting, northing, data, weights=None):
        """
        """
        # Check that the data coordinates are all 1D arrays of same size
        self.region_ = get_region(easting, northing)
        if self.spacing is None:
            self.force_easting_ = easting
            self.force_northing_ = northing
        else:
            coords = grid_coordinates(self.region_, spacing=self.spacing)
            self.force_easting_, self.force_northing_ = (
                i.ravel() for i in coords)
        jac = biharmonic_spline_jacobian(easting, northing,
                                         self.force_easting_,
                                         self.force_northing_,
                                         self.fudge)
        self.forces_ = spla.solve(jac.T.dot(jac), jac.T.dot(data.ravel()),
                                  assume_a='pos')
        return self

    def predict(self, easting, northing):
        """
        """
        check_is_fitted(self, ['forces_', 'force_easting_', 'force_northing_'])
        jac = biharmonic_spline_jacobian(easting.ravel(), northing.ravel(),
                                         self.force_easting_,
                                         self.force_northing_,
                                         self.fudge)
        shape = np.broadcast(easting, northing).shape
        return jac.dot(self.forces_).reshape(shape)


def biharmonic_spline_jacobian(easting, northing, force_easting,
                               force_northing, fudge=1e-5):
    """
    """
    size = easting.size
    # Reshaping the data to a column vector will automatically build a
    # Green's function matrix because of the array broadcasting.
    jac = biharmonic_spline2d(easting.reshape((size, 1)),
                              northing.reshape((size, 1)),
                              force_easting, force_northing,
                              fudge)
    return jac

