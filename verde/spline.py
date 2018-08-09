"""
Biharmonic splines in 2D.
"""
from warnings import warn

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input, least_squares
from .coordinates import grid_coordinates, get_region
from .utils import n_1d_arrays, parse_engine

try:
    import numba
    from numba import jit
except ImportError:
    numba = None
    from .utils import dummy_jit as jit


class Spline(BaseGridder):
    r"""
    Biharmonic spline interpolation using Green's functions.

    This gridder assumes Cartesian coordinates.

    Implements the 2D splines of [Sandwell1987]_. The Green's function for the spline
    corresponds to the elastic deflection of a thin sheet subject to a vertical force.
    For an observation point at the origin and a force at the coordinates given by the
    vector :math:`\mathbf{x}`, the Green's function is:

    .. math::

        g(\mathbf{x}) = \|\mathbf{x}\|^2 \left(\log \|\mathbf{x}\| - 1\right)

    In practice, this function is not defined for data points that coincide with a
    force. To prevent this, a fudge factor is added to :math:`\|\mathbf{x}\|`.

    The interpolation is performed by estimating forces that produce deflections that
    fit the observed data (using least-squares). Then, the interpolated points can be
    evaluated at any location.

    By default, the forces will be placed at the same points as the input data given to
    :meth:`~verde.Spline.fit`. This configuration provides an exact solution on top of
    the data points. However, this solution can be unstable for certain configurations
    of data points.

    Approximate (and more stable) solutions can be obtained by applying damping
    regularization to smooth the estimated forces (and interpolated values) or by
    distributing the forces on a regular grid instead (using arguments *spacing*,
    *shape*, and/or *region*).

    Data weights can be used during fitting but only have an any effect when using the
    approximate solutions.

    The Jacobian (design, sensitivity, feature, etc) matrix for the spline is normalized
    using :class:`sklearn.preprocessing.StandardScaler` without centering the mean so
    that the transformation can be undone in the estimated forces.

    Parameters
    ----------
    mindist : float
        A minimum distance between the point forces and data points. Needed because the
        Green's functions are singular when forces and data points coincide. Acts as a
        fudge factor.
    damping : None or float
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated forces. If None, no regularization is used.
    shape : None or tuple
        If not None, then should be the shape of the regular grid of forces. See
        :func:`verde.grid_coordinates` for details.
    spacing : None or float or tuple
        If not None, then should be the spacing of the regular grid of forces. See
        :func:`verde.grid_coordinates` for details.
    region : None or tuple
        If not None, then the boundaries (``[W, E, S, N]``) used to generate a regular
        grid of forces. If None is given, then will use the boundaries of data given to
        the first call to :meth:`~verde.Spline.fit`.
    engine : str
        Computation engine for the Jacobian matrix. Can be ``'auto'``, ``'numba'``, or
        ``'numpy'``. If ``'auto'``, will use numba if it is installed or numpy
        otherwise. The numba version is multi-threaded and considerably faster, which
        makes fitting and predicting faster.

    Attributes
    ----------
    forces_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the forces.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Spline.grid` and :meth:`~verde.Spline.scatter` methods.

    """

    def __init__(
        self,
        mindist=1e-5,
        damping=None,
        shape=None,
        spacing=None,
        region=None,
        engine="auto",
    ):
        self.damping = damping
        self.shape = shape
        self.spacing = spacing
        self.mindist = mindist
        self.region = region
        self.engine = engine

    def fit(self, coordinates, data, weights=None):
        """
        Fit the biharmonic spline to the given data.

        The data region is captured and used as default for the
        :meth:`~verde.Spline.grid` and :meth:`~verde.Spline.scatter` methods.

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
        warn_weighted_exact_solution(self, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        self.force_coords_ = get_force_coordinates(self, coordinates)
        jacobian = self.jacobian(coordinates[:2], self.force_coords_)
        self.force_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        Evaluate the estimated spline on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.Spline.fit`).

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
            The data values evaluated on the given points.

        """
        check_is_fitted(self, ["force_", "force_coords_"])
        jac = self.jacobian(coordinates[:2], self.force_coords_)
        shape = np.broadcast(*coordinates[:2]).shape
        return jac.dot(self.force_).reshape(shape)

    def jacobian(self, coordinates, force_coordinates, dtype="float64"):
        """
        Make the Jacobian matrix for the 2D biharmonic spline.

        Each column of the Jacobian is the Green's function for a single force evaluated
        on all observation points [Sandwell1987]_.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting and
            northing will be used, all subsequent coordinates will be ignored.
        force_coordinates : tuple of arrays
            Arrays with the coordinates for the forces. Should be in the same order as
            the coordinate arrays.
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_forces) Jacobian matrix.

        """
        force_east, force_north = n_1d_arrays(force_coordinates, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        if parse_engine(self.engine) == "numba":
            jac = np.empty((east.size, force_east.size), dtype=dtype)
            jacobian_numba(east, north, force_east, force_north, self.mindist, jac)
        else:
            jac = jacobian_numpy(
                east, north, force_east, force_north, self.mindist, dtype
            )
        return jac


def warn_weighted_exact_solution(spline, weights):
    """
    Warn the user that a weights doesn't work for the exact solution.

    Parameters
    ----------
    spline : estimator
        The spline instance that we'll check. Needs to have ``shape``, ``spacing``, and
        ``damping`` attributes.
    weights : array or None
        The weights given to fit.

    """
    # Check if we're using weights with the exact solution and warn the
    # user that it won't have any effect.
    exact = all(i is None for i in [spline.damping, spline.spacing, spline.shape])
    if weights is not None and exact:
        warn(
            "Weights are ignored for the exact solution. "
            "Use damping or specify a shape/spacing for the forces."
        )


def get_force_coordinates(spline, coordinates):
    """
    Make arrays for force coordinates depending on the spline configuration.

    If no ``shape`` and ``spacing`` are given, then will use the data coordinates as
    force coordinates. Otherwise, will generate a grid based on the given spacing/shape
    and the data region.

    Parameters
    ----------
    spline : estimator
        The spline instance. Needs to have ``shape`` and ``spacing`` attributes. If
        ``region`` is present, will use it to define the grid region. If not, will use
        the data region.
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.

    Returns
    -------
    force_coordinates : tuple of arrays
        The coordinate arrays for the forces.

    """
    # Set the force positions. If no shape or spacing are given, then they
    # will coincide with the data points. This will only happen once so the
    # model won't change during later fits.
    if spline.shape is None and spline.spacing is None:
        coords = tuple(i.copy() for i in n_1d_arrays(coordinates, n=2))
    else:
        if spline.region is None:
            region = get_region(coordinates)
        else:
            region = spline.region
        coords = tuple(
            i.ravel()
            for i in grid_coordinates(
                region, shape=spline.shape, spacing=spline.spacing
            )
        )
    return coords


@jit(nopython=True, target="cpu", fastmath=True, parallel=True)
def jacobian_numba(east, north, force_east, force_north, mindist, jac):
    """
    Calculate the Jacobian matrix using numba to speed things up.
    """
    for i in numba.prange(east.size):  # pylint: disable=not-an-iterable
        for j in range(force_east.size):
            distance = np.sqrt(
                (east[i] - force_east[j]) ** 2 + (north[i] - force_north[j]) ** 2
            )
            distance += mindist
            jac[i, j] = (distance ** 2) * (np.log(distance) - 1)
    return jac


def jacobian_numpy(east, north, force_east, force_north, mindist, dtype):
    """
    Calculate the Jacobian using numpy broadcasting. Is slightly slower than the numba
    version.
    """
    # Reshaping the data to a column vector will automatically build a
    # distance matrix between each data point and force.
    distance = np.hypot(
        east.reshape((east.size, 1)) - force_east,
        north.reshape((north.size, 1)) - force_north,
        dtype=dtype,
    )
    # The mindist factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += mindist
    return (distance ** 2) * (np.log(distance) - 1)
