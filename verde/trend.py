"""
2D polynomial trends.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input, least_squares, n_1d_arrays
from .coordinates import get_region


class Trend(BaseGridder):
    r"""
    Fit a 2D polynomial trend to spatial data.

    The polynomial of degree :math:`N` is defined as:

    .. math::

        f(e, n) = \sum\limits_{l=0}^{N}\sum\limits_{m=0}^{N - l} e^l n^m

    in which :math:`e` and :math:`n` are the easting and northing coordinates,
    respectively.

    The trend is estimated through weighted least-squares regression. The
    Jacobian (design, sensitivity, feature, etc) matrix for the regression is
    normalized using :class:`sklearn.preprocessing.StandardScaler` without
    centering the mean so that the transformation can be undone in the
    estimated coefficients.

    Parameters
    ----------
    degree : int
        The degree of the polynomial. Must be >= 0 (a degree of zero would
        estimate the mean of the data).

    Attributes
    ----------
    coef_ : array
        The estimated polynomial coefficients that fit the observed data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Trend.grid` and :meth:`~verde.Trend.scatter` methods.

    Examples
    --------

    >>> from verde import grid_coordinates
    >>> import numpy as np
    >>> coordinates = grid_coordinates((1, 5, -5, -1), shape=(5, 5))
    >>> data = 10 + 2*coordinates[0] - 0.4*coordinates[1]
    >>> trend = Trend(degree=1).fit(coordinates, data)
    >>> print(
    ...     "Coefficients:",
    ...     ', '.join(['{:.1f}'.format(i) for i in trend.coef_])
    ... )
    Coefficients: 10.0, 2.0, -0.4
    >>> np.allclose(trend.predict(coordinates), data)
    True

    A zero degree polynomial estimates the mean of the data:

    >>> mean = Trend(degree=0).fit(coordinates, data)
    >>> np.allclose(mean.predict(coordinates), data.mean())
    True
    >>> print("Data mean:", '{:.2f}'.format(data.mean()))
    Data mean: 17.20
    >>> print("Coefficient:", '{:.2f}'.format(mean.coef_[0]))
    Coefficient: 17.20

    We can use weights to account for outliers or data points with variable
    uncertainties (see :func:`verde.variance_to_weights`):

    >>> data_out = data.copy()
    >>> data_out[2, 2] += 500
    >>> weights = np.ones_like(data)
    >>> weights[2, 2] = 1e-10
    >>> trend_out = Trend(degree=1).fit(coordinates, data_out, weights)
    >>> # Still recover the coefficients even with the added outlier
    >>> print(
    ...     "Coefficients:",
    ...     ', '.join(['{:.1f}'.format(i) for i in trend_out.coef_])
    ... )
    Coefficients: 10.0, 2.0, -0.4
    >>> # The residual at the outlier location should be values we added to
    >>> # that point
    >>> residual = data_out - trend_out.predict(coordinates)
    >>> print('{:.2f}'.format(residual[2, 2]))
    500.00

    """

    def __init__(self, degree):
        super().__init__()
        self.degree = degree

    def fit(self, coordinates, data, weights=None):
        """
        Fit the trend to the given data.

        The data region is captured and used as default for the
        :meth:`~verde.Trend.grid` and :meth:`~verde.Trend.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : array
            The data values of each data point.
        weights : None or array
            If not None, then the weights assigned to each data point.
            Typically, this should be 1 over the data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        easting, northing = n_1d_arrays(coordinates, 2)
        self.region_ = get_region((easting, northing))
        jac = self.jacobian((easting, northing), dtype=data.dtype)
        self.coef_ = least_squares(jac, data, weights, damping=None)
        return self

    def predict(self, coordinates):
        """
        Evaluate the polynomial trend on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.Trend.fit`).

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
            The trend values evaluated on the given points.

        """
        check_is_fitted(self, ["coef_"])
        easting, northing = n_1d_arrays(coordinates, 2)
        shape = np.broadcast(*coordinates[:2]).shape
        data = np.zeros(easting.size, dtype=easting.dtype)
        combinations = polynomial_power_combinations(self.degree)
        for coef, (i, j) in zip(self.coef_, combinations):
            data += (easting ** i) * (northing ** j) * coef
        return data.reshape(shape)

    def jacobian(self, coordinates, dtype="float64"):
        """
        Make the Jacobian matrix for a 2D polynomial.

        Each column of the Jacobian is ``easting**i * northing**j`` for each
        (i, j) pair in the polynomial.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        dtype : str or numpy dtype
            The type of the output Jacobian numpy array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_coefficients) Jacobian matrix.

        Examples
        --------

        >>> import numpy as np
        >>> east = np.linspace(0, 4, 5)
        >>> north = np.linspace(-5, -1, 5)
        >>> print(Trend(degree=1).jacobian((east, north), dtype=np.int))
        [[ 1  0 -5]
         [ 1  1 -4]
         [ 1  2 -3]
         [ 1  3 -2]
         [ 1  4 -1]]
        >>> print(Trend(degree=2).jacobian((east, north), dtype=np.int))
        [[ 1  0 -5  0  0 25]
         [ 1  1 -4  1 -4 16]
         [ 1  2 -3  4 -6  9]
         [ 1  3 -2  9 -6  4]
         [ 1  4 -1 16 -4  1]]

        """
        easting, northing = n_1d_arrays(coordinates, 2)
        if easting.shape != northing.shape:
            raise ValueError("Coordinate arrays must have the same shape.")
        combinations = polynomial_power_combinations(self.degree)
        ndata = easting.size
        nparams = len(combinations)
        out = np.empty((ndata, nparams), dtype=dtype)
        for col, (i, j) in enumerate(combinations):
            out[:, col] = (easting ** i) * (northing ** j)
        return out


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
    >>> # A degree zero polynomial would be just the mean
    >>> print(polynomial_power_combinations(0))
    ((0, 0),)

    """
    if degree < 0:
        raise ValueError("Invalid polynomial degree '{}'. Must be >= 0.".format(degree))
    combinations = ((i, j) for j in range(degree + 1) for i in range(degree + 1 - j))
    return tuple(sorted(combinations, key=sum))
