"""
Biharmonic splines in 2D.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base.gridder import BaseGridder
from .coordinates import grid_coordinates, get_region
from .utils import normalize_jacobian, linear_fit


def polynomial_power_combinations(degree):
    """
    Combinations of powers for a 2D polynomial of a given degree.

    Produces the (i, j) pairs to evaluate the polynomial with ``x**i*y**j``.

    Parameters
    ----------
    degree : int
        The degree of the 2D polynomial. Must be >= 1.

    Returns
    -------
    combinations : tuple
        A tuple with ``(i, j)`` pairs.

    Examples
    --------

    >>> print(polynomial_power_combinations(1))
    ((0, 0), (0, 1), (1, 0))
    >>> print(polynomial_power_combinations(2))
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))
    >>> # This is a long polynomial so split it in two lines
    >>> print(polynomial_power_combinations(3)[:5])
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0))
    >>> print(polynomial_power_combinations(3)[5:])
    ((1, 1), (1, 2), (2, 0), (2, 1), (3, 0))

    """
    if degree < 1:
        raise ValueError("Invalid polynomial degree '{}'. Must be >= 1."
                         .format(degree))
    combinations = tuple((i, j)
                         for i in range(degree + 1)
                         for j in range(degree + 1 - i))
    return combinations


class Trend(BaseGridder):
    """
    Document utils functions
    """

    def __init__(self, degree, damping=None):
        self.degree = degree
        self.damping = damping

    def fit(self, easting, northing, data, weights=None):
        """
        """
        if easting.shape != northing.shape != data.shape:
            raise ValueError(
                "Coordinate and data arrays must have the same shape.")
        self.region_ = get_region(easting, northing)
        jac = trend_jacobian(easting, northing, degree=self.degree)
        jac, transform = normalize_jacobian(jac)
        params = linear_fit(jac, data.ravel(), weights=weights,
                            damping=self.damping)
        self.coefs_ = params*transform
        self.residuals_ = data - jac.dot(params).reshape(data.shape)
        return self

    def predict(self, easting, northing):
        """
        """
        check_is_fitted(self, ['coefs_'])
        jac = trend_jacobian(easting, northing, degree=self.degree)
        shape = np.broadcast(easting, northing).shape
        return jac.dot(self.coefs_).reshape(shape)


def trend_jacobian(easting, northing, degree):
    """
    """
    if easting.shape != northing.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    combinations = polynomial_power_combinations(degree)
    ndata = easting.size
    nparams = len(combinations)
    out = np.empty((ndata, nparams))
    for col, (i, j) in enumerate(combinations):
        out[:, col] = (easting.ravel()**i)*(northing.ravel()**j)
    return out
