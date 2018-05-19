"""
2D polynomial trends.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

from .base.gridder import BaseGridder
from .coordinates import get_region
from .utils import linear_fit


class Trend(BaseGridder):
    """
    Fit a 2D polynomial trend to spatial data.

    The trend is estimated through weighted least-squares regression
    with optional damping regularization (ridge regression).

    Regularization will likely be required for high degree polynomials (> 4).
    The Jacobian (design, sensitivity, etc) matrix for the regression is
    normalized to be in the range [-1, 1]. This means that a suitable damping
    parameter will typically be in the range [1e-2, 1e-7]. This may vary for
    different datasets.

    Parameters
    ----------
    degree : int
        The degree of the polynomial. Must be >= 1. High degrees may result in
        ill-conditioned systems so damping may be required.
    damping : None or float
        The damping regularization parameter. If None, no regularization is
        used. Otherwise, must be a positive number. Damping imposes that the
        polynomial coefficients be as small as possible. The damping parameter
        controls how strongly this condition is imposed.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> east, north = grid_coordinates((1, 5, -5, -1), shape=(5, 5))
    >>> data = 10 + 2*east - 0.5*north
    >>> trend = Trend(degree=1).fit(east, north, data)
    >>> print(', '.join(['{:.1f}'.format(i) for i in trend.coefs_]))
    10.0, 2.0, -0.5
    >>> import numpy as np
    >>> np.allclose(trend.predict(east, north), data)
    True

    """

    def __init__(self, degree, damping=None):
        self.degree = degree
        self.damping = damping

    def fit(self, easting, northing, data, weights=None):
        """
        Fit the trend to the given data.

        The data region is captured and used as default for the
        :meth:`~verde.Trend.grid` and :meth:`~verde.Trend.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        easting : array
            The values of the West-East coordinates of each data point.
        northing : array
            The values of the South-North coordinates of each data point.
        data : array
            The data values of each data point.
        weights : None or array
            If not None, then the weights assigned to each data point.
            Typically, this should be 1 over the data uncertainty squared.

        Returns
        -------
        self : verde.Trend
            Returns this estimator instance for chaining operations.

        """
        if easting.shape != northing.shape != data.shape:
            raise ValueError(
                "Coordinate and data arrays must have the same shape.")
        if weights is not None:
            if weights.shape != data.shape:
                raise ValueError(
                    "Weights must have the same shape as the data array.")
            weights = weights.ravel()
        self.region_ = get_region(easting, northing)
        jac = trend_jacobian(easting, northing, degree=self.degree,
                             dtype=data.dtype)
        scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
        jac = scaler.fit_transform(jac)
        params = linear_fit(jac, data.ravel(), weights=weights,
                            damping=self.damping)
        self.coefs_ = params/scaler.scale_
        self.residuals_ = data - jac.dot(params).reshape(data.shape)
        return self

    def predict(self, easting, northing):
        """
        Evaluate the polynomial trend on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.Trend.fit`).

        Parameters
        ----------
        easting : array
            The values of the West-East coordinates of each data point.
        northing : array
            The values of the South-North coordinates of each data point.

        Returns
        -------
        data : array
            The trend values evaluated on the given points.

        """
        check_is_fitted(self, ['coefs_'])
        jac = trend_jacobian(easting, northing, degree=self.degree)
        shape = np.broadcast(easting, northing).shape
        return jac.dot(self.coefs_).reshape(shape)


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
    ((0, 0), (1, 0), (0, 1))
    >>> print(polynomial_power_combinations(2))
    ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    >>> # This is a long polynomial so split it in two lines
    >>> print(" ".join([str(c) for c in polynomial_power_combinations(3)]))
    (0, 0) (1, 0) (0, 1) (2, 0) (1, 1) (0, 2) (3, 0) (2, 1) (1, 2) (0, 3)

    """
    if degree < 1:
        raise ValueError("Invalid polynomial degree '{}'. Must be >= 1."
                         .format(degree))
    combinations = ((i, j)
                    for j in range(degree + 1)
                    for i in range(degree + 1 - j))
    return tuple(sorted(combinations, key=sum))


def trend_jacobian(easting, northing, degree, dtype='float64'):
    """
    Make the Jacobian for a 2D polynomial of the given degree.

    Each column of the Jacobian is ``easting**i * northing**j` for each (i, j)
    pair in the polynomial of the given degree.

    Parameters
    ----------
    easting : array
        The values of the West-East coordinates of each data point.
    northing : array
        The values of the South-North coordinates of each data point.
    degree : int
        The degree of the 2D polynomial. Must be >= 1.
    dtype : str or numpy dtype
        The type of the output Jacobian numpy array.

    Returns
    -------
    jacobian : 2D array
        The Jacobian matrix.

    Examples
    --------

    >>> import numpy as np
    >>> east = np.linspace(0, 4, 5)
    >>> north = np.linspace(-5, -1, 5)
    >>> print(trend_jacobian(east, north, degree=1, dtype=np.int))
    [[ 1  0 -5]
     [ 1  1 -4]
     [ 1  2 -3]
     [ 1  3 -2]
     [ 1  4 -1]]
    >>> print(trend_jacobian(east, north, degree=2, dtype=np.int))
    [[ 1  0 -5  0  0 25]
     [ 1  1 -4  1 -4 16]
     [ 1  2 -3  4 -6  9]
     [ 1  3 -2  9 -6  4]
     [ 1  4 -1 16 -4  1]]

    """
    if easting.shape != northing.shape:
        raise ValueError("Coordinate arrays must have the same shape.")
    combinations = polynomial_power_combinations(degree)
    ndata = easting.size
    nparams = len(combinations)
    out = np.empty((ndata, nparams), dtype=dtype)
    for col, (i, j) in enumerate(combinations):
        out[:, col] = (easting.ravel()**i)*(northing.ravel()**j)
    return out
