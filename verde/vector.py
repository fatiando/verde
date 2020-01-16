"""
Classes for dealing with vector data.
"""
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import n_1d_arrays, check_fit_input, least_squares, BaseGridder
from .spline import warn_weighted_exact_solution
from .utils import parse_engine
from .coordinates import get_region

try:
    import numba
    from numba import jit
except ImportError:
    numba = None
    from .utils import dummy_jit as jit


# Otherwise, DeprecationWarning won't be shown, kind of defeating the purpose.
warnings.simplefilter("default")


class Vector(BaseGridder):
    """
    Fit an estimator to each component of multi-component vector data.

    Provides a convenient way of fitting and gridding vector data using scalar
    gridders and estimators.

    Each data component provided to :meth:`~verde.Vector.fit` is fitted to a
    separated estimator. Methods like :meth:`~verde.Vector.grid` and
    :meth:`~verde.Vector.predict` will operate on the multiple components
    simultaneously.

    .. warning::

        Never pass code like this as input to this class: ``[vd.Trend(1)]*3``.
        This creates 3 references to the **same instance** of ``Trend``, which
        means that they will all get the same coefficients after fitting. Use a
        list comprehension instead: ``[vd.Trend(1) for i in range(3)]``.

    Parameters
    ----------
    components : tuple or list
        A tuple or list of the estimator/gridder instances used for each
        component. The estimators will be applied for each data component in
        the same order that they are given here.

    Attributes
    ----------
    components : tuple
        Tuple of the fitted estimators on each component of the data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Vector.grid` and :meth:`~verde.Vector.scatter` methods.

    See also
    --------
    verde.Chain : Chain filtering operations to fit on each subsequent output.

    """

    def __init__(self, components):
        super().__init__()
        self.components = components

    def fit(self, coordinates, data, weights=None):
        """
        Fit the estimators to the given multi-component data.

        The data region is captured and used as default for the
        :meth:`~verde.Vector.grid` and :meth:`~verde.Vector.scatter` methods.

        All input arrays must have the same shape. If weights are given, there
        must be a separate array for each component of the data.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : tuple of array
            The data values of each component at each data point. Must be a
            tuple.
        weights : None or tuple of array
            If not None, then the weights assigned to each data point of each
            data component. Typically, this should be 1 over the data
            uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        if not isinstance(data, tuple):
            raise ValueError(
                "Data must be a tuple of arrays. {} given.".format(type(data))
            )
        if weights is not None and not isinstance(weights, tuple):
            raise ValueError(
                "Weights must be a tuple of arrays. {} given.".format(type(weights))
            )
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self.region_ = get_region(coordinates[:2])
        for estimator, data_comp, weight_comp in zip(self.components, data, weights):
            estimator.fit(coordinates, data_comp, weight_comp)
        return self

    def predict(self, coordinates):
        """
        Evaluate each data component on a set of points.

        Requires a fitted estimator (see :meth:`~verde.Vector.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : tuple of array
            The values for each vector component evaluated on the given points.
            The order of components will be the same as was provided to
            :meth:`~verde.Vector.fit`.

        """
        check_is_fitted(self, ["region_"])
        return tuple(comp.predict(coordinates) for comp in self.components)


class VectorSpline2D(BaseGridder):
    r"""
    Elastically coupled interpolation of 2-component vector data.

    .. warning::

        The :class:`~verde.VectorSpline2D` class is deprecated and will be
        removed in Verde v2.0.0. Its usage is restricted to GPS/GNSS data and
        not in the general scope of Verde. Please use the implementation in the
        `Erizo <https://github.com/fatiando/erizo>`__ package instead.

    This gridder assumes Cartesian coordinates.

    Uses the Green's functions based on elastic deformation from
    [SandwellWessel2016]_. The interpolation is done by estimating point forces
    that generate an elastic deformation that fits the observed vector data.
    The deformation equations are based on a 2D elastic sheet with a constant
    Poisson's ratio. The data can then be predicted at any desired location.

    The east and north data components are coupled through the elastic
    deformation equations. This coupling is controlled by the Poisson's ratio,
    which is usually between -1 and 1. The special case of Poisson's ratio -1
    leads to an uncoupled interpolation, meaning that the east and north
    components don't interfere with each other.

    The point forces are traditionally placed under each data point. The force
    locations are set the first time :meth:`~verde.VectorSpline2D.fit` is
    called. Subsequent calls will fit using the same force locations as the
    first call. This configuration results in an exact prediction at the data
    points but can be unstable.

    [SandwellWessel2016]_ stabilize the solution using Singular Value
    Decomposition but we use ridge regression instead. The regularization can
    be controlled using the *damping* argument. Alternatively, you can specify
    the position of the forces manually using the *force_coords* argument.
    Regularization or forces not coinciding with data points will result in a
    least-squares estimate, not an exact solution. Note that the least-squares
    solution is required for data weights to have any effect.

    Before fitting, the Jacobian (design, sensitivity, feature, etc) matrix for
    the spline is normalized using
    :class:`sklearn.preprocessing.StandardScaler` without centering the mean so
    that the transformation can be undone in the estimated forces.

    Parameters
    ----------
    poisson : float
        The Poisson's ratio for the elastic deformation Green's functions.
        Default is 0.5. A value of -1 will lead to uncoupled interpolation of
        the east and north data components.
    mindist : float
        A minimum distance between the point forces and data points. Needed
        because the Green's functions are singular when forces and data points
        coincide. Acts as a fudge factor. A good rule of thumb is to use the
        average spacing between data points.
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated forces. If None, no
        regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), then will be set to the data coordinates the first time
        :meth:`~verde.VectorSpline2D.fit` is called.
    engine : str
        Computation engine for the Jacobian matrix and predictions. Can be
        ``'auto'``, ``'numba'``, or ``'numpy'``. If ``'auto'``, will use numba
        if it is installed or numpy otherwise. The numba version is
        multi-threaded and usually faster, which makes fitting and predicting
        faster.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.VectorSpline2D.grid` and
        :meth:`~verde.VectorSpline2D.scatter` methods.

    """

    def __init__(
        self, poisson=0.5, mindist=10e3, damping=None, force_coords=None, engine="auto"
    ):
        super().__init__()
        self.poisson = poisson
        self.mindist = mindist
        self.damping = damping
        self.force_coords = force_coords
        self.engine = engine
        warnings.warn(
            "VectorSpline2D is deprecated and will be removed in Verde v2.0.0."
            " Please use the implementation in the Erizo package instead "
            "(https://github.com/fatiando/erizo).",
            DeprecationWarning,
        )

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given 2-component vector data.

        The data region is captured and used as default for the
        :meth:`~verde.VectorSpline2D.grid` and
        :meth:`~verde.VectorSpline2D.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : tuple of array
            A tuple ``(east_component, north_component)`` of arrays with the
            vector data values at each point.
        weights : None or tuple array
            If not None, then the weights assigned to each data point. Must be
            one array per data component. Typically, this should be 1 over the
            data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        if len(data) != 2:
            raise ValueError(
                "Need two data components. Only {} given.".format(len(data))
            )
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        if any(w is not None for w in weights):
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        warn_weighted_exact_solution(self, weights)
        data = np.concatenate([i.ravel() for i in data])
        if self.force_coords is None:
            self.force_coords = tuple(i.copy() for i in n_1d_arrays(coordinates, n=2))
        jacobian = self.jacobian(coordinates[:2], self.force_coords)
        self.force_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.VectorSpline2D.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : tuple of arrays
            A tuple ``(east_component, north_component)`` of arrays with the
            predicted vector data values at each point.

        """
        check_is_fitted(self, ["force_"])
        force_east, force_north = self.force_coords
        east, north = n_1d_arrays(coordinates, n=2)
        cast = np.broadcast(*coordinates[:2])
        npoints = cast.size
        components = (
            np.empty(npoints, dtype=east.dtype),
            np.empty(npoints, dtype=east.dtype),
        )
        if parse_engine(self.engine) == "numba":
            components = predict_2d_numba(
                east,
                north,
                force_east,
                force_north,
                self.mindist,
                self.poisson,
                self.force_,
                components[0],
                components[1],
            )
        else:
            components = predict_2d_numpy(
                east,
                north,
                force_east,
                force_north,
                self.mindist,
                self.poisson,
                self.force_,
                components[0],
                components[1],
            )
        return tuple(comp.reshape(cast.shape) for comp in components)

    def jacobian(self, coordinates, force_coords, dtype="float64"):
        """
        Make the Jacobian matrix for the 2D coupled elastic deformation.

        The Jacobian is segmented into 4 parts, each relating a force component
        to a data component [SandwellWessel2016]_::

            | J_ee  J_ne |*|f_e| = |d_e|
            | J_ne  J_nn | |f_n|   |d_n|

        The forces and data are assumed to be stacked into 1D arrays with the
        east component on top of the north component.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        force_coords : tuple of arrays
            Arrays with the coordinates for the forces. Should be in the same
            order as the coordinate arrays.
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data*2, n_forces*2) Jacobian matrix.

        """
        force_east, force_north = n_1d_arrays(force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        jac = np.empty((east.size * 2, force_east.size * 2), dtype=dtype)
        if parse_engine(self.engine) == "numba":
            jac = jacobian_2d_numba(
                east, north, force_east, force_north, self.mindist, self.poisson, jac
            )
        else:
            jac = jacobian_2d_numpy(
                east, north, force_east, force_north, self.mindist, self.poisson, jac
            )
        return jac


def greens_func_2d(east, north, mindist, poisson):
    "Calculate the Green's functions for the 2D elastic case."
    distance = np.sqrt(east ** 2 + north ** 2)
    # The mindist factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += mindist
    # Pre-compute common terms for the Green's functions of each component
    ln_r = (3 - poisson) * np.log(distance)
    over_r2 = (1 + poisson) / distance ** 2
    green_ee = ln_r + over_r2 * north ** 2
    green_nn = ln_r + over_r2 * east ** 2
    green_ne = -over_r2 * east * north
    return green_ee, green_nn, green_ne


def predict_2d_numpy(
    east, north, force_east, force_north, mindist, poisson, forces, vec_east, vec_north
):
    "Calculate the predicted data using numpy."
    vec_east[:] = 0
    vec_north[:] = 0
    nforces = forces.size // 2
    for j in range(nforces):
        green_ee, green_nn, green_ne = greens_func_2d(
            east - force_east[j], north - force_north[j], mindist, poisson
        )
        vec_east += green_ee * forces[j] + green_ne * forces[j + nforces]
        vec_north += green_ne * forces[j] + green_nn * forces[j + nforces]
    return vec_east, vec_north


def jacobian_2d_numpy(east, north, force_east, force_north, mindist, poisson, jac):
    "Calculate the Jacobian matrix using numpy broadcasting."
    npoints = east.size
    nforces = force_east.size
    # Reshaping the data coordinates to a column vector will automatically
    # build a Green's functions matrix between each data point and force.
    green_ee, green_nn, green_ne = greens_func_2d(
        east.reshape((npoints, 1)) - force_east,
        north.reshape((npoints, 1)) - force_north,
        mindist,
        poisson,
    )
    jac[:npoints, :nforces] = green_ee
    jac[npoints:, nforces:] = green_nn
    jac[:npoints, nforces:] = green_ne
    jac[npoints:, :nforces] = green_ne  # J is symmetric
    return jac


@jit(nopython=True, fastmath=True, parallel=True)
def predict_2d_numba(
    east, north, force_east, force_north, mindist, poisson, forces, vec_east, vec_north
):
    "Calculate the predicted data using numba to speed things up."
    nforces = forces.size // 2
    for i in numba.prange(east.size):  # pylint: disable=not-an-iterable
        vec_east[i] = 0
        vec_north[i] = 0
        for j in range(nforces):
            green_ee, green_nn, green_ne = GREENS_FUNC_2D_JIT(
                east[i] - force_east[j], north[i] - force_north[j], mindist, poisson
            )
            vec_east[i] += green_ee * forces[j] + green_ne * forces[j + nforces]
            vec_north[i] += green_ne * forces[j] + green_nn * forces[j + nforces]
    return vec_east, vec_north


@jit(nopython=True, fastmath=True, parallel=True)
def jacobian_2d_numba(east, north, force_east, force_north, mindist, poisson, jac):
    "Calculate the Jacobian matrix using numba to speed things up."
    nforces = force_east.size
    npoints = east.size
    for i in numba.prange(npoints):  # pylint: disable=not-an-iterable
        for j in range(nforces):
            green_ee, green_nn, green_ne = GREENS_FUNC_2D_JIT(
                east[i] - force_east[j], north[i] - force_north[j], mindist, poisson
            )
            jac[i, j] = green_ee
            jac[i + npoints, j + nforces] = green_nn
            jac[i, j + nforces] = green_ne
            jac[i + npoints, j] = green_ne  # J is symmetric
    return jac


# JIT compile the Greens functions for use in numba functions
GREENS_FUNC_2D_JIT = jit(nopython=True, fastmath=True)(greens_func_2d)
