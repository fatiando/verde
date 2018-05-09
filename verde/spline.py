"""
Biharmonic splines in 2D.
"""
import numpy as np
import scipy.linalg as spla
from sklearn.utils.validation import check_is_fitted

from .greens_functions.spline import biharmonic_spline2d
from .base.gridder import BaseGridder
from .coordinates import grid_coordinates, get_region


class BiharmonicSpline(BaseGridder):
    """
    """

    def __init__(self, fudge=1e-5, damping=None, shape=None, spacing=None):
        self.damping = damping
        self.shape = shape
        self.spacing = spacing
        self.fudge = fudge

    def fit(self, easting, northing, data, weights=None):
        """
        """
        if easting.shape != northing.shape != data.shape:
            raise ValueError(
                "Coordinate and data arrays must have the same shape.")
        self.region_ = get_region(easting, northing)
        if self.shape is None and self.spacing is None:
            self.force_easting_ = easting
            self.force_northing_ = northing
        else:
            coords = grid_coordinates(self.region_, shape=self.shape,
                                      spacing=self.spacing)
            self.force_easting_, self.force_northing_ = (
                i.ravel() for i in coords)
        ndata = data.size
        nforces = self.force_easting_.size
        nparams = nforces + 3
        jac = np.empty((ndata, nparams))
        jac[:, :-3] = biharmonic_spline_jacobian(easting, northing,
                                                 self.force_easting_,
                                                 self.force_northing_,
                                                 self.fudge)
        jac[:, -3:] = trend_jacobian(easting, northing, degree=1)
        jac, transform = normalize_jacobian(jac)
        self.params_ = spla.solve(jac.T.dot(jac), jac.T.dot(data.ravel()),
                                  assume_a='pos')
        return self

    def predict(self, easting, northing):
        """
        """
        check_is_fitted(self, ['forces_', 'force_easting_', 'force_northing_'])
        ndata = easting.size
        nforces = self.force_easting_.size
        nparams = nforces + 3
        jac = np.empty((ndata, nparams))
        jac[:, :-3] = biharmonic_spline_jacobian(easting.ravel(),
                                                 northing.ravel(),
                                                 self.force_easting_,
                                                 self.force_northing_,
                                                 self.fudge)
        jac[:, -3:] = trend_jacobian(easting, northing, degree=1)
        shape = np.broadcast(easting, northing).shape
        return jac.dot(self.params_).reshape(shape)


def normalize_jacobian(jacobian):
    """
    """
    transform = 1/np.abs(jacobian).max(axis=0)
    # Element-wise multiplication with the diagonal of the scale matrix is the
    # same as A.dot(S)
    jacobian *= transform
    return jacobian, transform


def trend_jacobian(easting, northing, degree):
    """
    """
    if easting.shape != northing.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    ndata = easting.size
    nparams = sum(i + 1 for i in range(degree + 1))
    combinations = [(i, j)
                    for i in range(degree + 1)
                    for j in range(degree + 1 - i)]
    jac = np.empty((ndata, nparams))
    for col, (i, j) in enumerate(combinations):
        jac[:, col] = easting**i*northing**j
    return jac


def biharmonic_spline_jacobian(easting, northing, force_easting,
                               force_northing, fudge=1e-5):
    """
    """
    if easting.shape != northing.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    size = easting.size
    # Reshaping the data to a column vector will automatically build a
    # Green's function matrix because of the array broadcasting.
    jac = biharmonic_spline2d(easting.reshape((size, 1)),
                              northing.reshape((size, 1)),
                              force_easting, force_northing,
                              fudge)
    return jac
