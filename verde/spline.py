"""
Biharmonic splines in 2D.
"""
import itertools
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import n_1d_arrays, BaseGridder, check_fit_input, least_squares
from .coordinates import get_region
from .utils import dispatch, parse_engine
from .model_selection import cross_val_score

try:
    import numba
    from numba import jit
except ImportError:
    numba = None
    from .utils import dummy_jit as jit


# Otherwise, DeprecationWarning won't be shown, kind of defeating the purpose.
warnings.simplefilter("default")


class SplineCV(BaseGridder):
    r"""
    Cross-validated biharmonic spline interpolation.

    Similar to :class:`verde.Spline` but automatically chooses the best
    *damping* and *mindist* parameters using cross-validation. Tests all
    combinations of the given *dampings* and *mindists* and selects the maximum
    (or minimum) mean cross-validation score (i.e., a grid search).

    This can optionally run in parallel using :mod:`dask`. To do this, use
    ``delayed=True`` to dispatch computations with :func:`dask.delayed`.
    In this case, each fit and score operation of the grid search will be
    performed in parallel.

    .. note::

        When using *delayed*, the ``scores_`` attribute will be
        :func:`dask.delayed` objects instead of the actual scores. This is
        because the scores are an intermediate step in the computations and
        their results are not stored. If you need the scores, run
        :func:`dask.compute` on ``scores_`` to calculate them. Be warned that
        **this will run the grid search again**. It might still be faster than
        serial execution but not necessarily.

    .. warning::

        The ``client`` parameter is deprecated and will be removed in Verde
        v2.0.0. Use ``delayed`` instead.

    Other cross-validation generators from :mod:`sklearn.model_selection` can
    be used by passing them through the *cv* argument.

    Parameters
    ----------
    mindists : iterable or 1d array
        List (or other iterable) of *mindist* parameter values to try. Can be
        considered a minimum distance between the point forces and data points.
        Needed because the Green's functions are singular when forces and data
        points coincide. Acts as a fudge factor.
    dampings : iterable or 1d array
        List (or other iterable) of *damping* parameter values to try. Is the
        positive damping regularization parameter. Controls how much smoothness
        is imposed on the estimated forces. If None, no regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), then will be set to the data coordinates the first time
        :meth:`~verde.SplineCV.fit` is called.
    engine : str
        Computation engine for the Jacobian matrix and prediction. Can be
        ``'auto'``, ``'numba'``, or ``'numpy'``. If ``'auto'``, will use numba
        if it is installed or numpy otherwise. The numba version is
        multi-threaded and usually faster, which makes fitting and predicting
        faster.
    cv : None or cross-validation generator
        Any scikit-learn cross-validation generator. If not given, will use the
        default set by :func:`verde.cross_val_score`.
    client : None or dask.distributed.Client
        **DEPRECATED:** This option is deprecated and will be removed in Verde
        v2.0.0. If None, then computations are run serially. Otherwise, should
        be a dask ``Client`` object. It will be used to dispatch computations
        to the dask cluster.
    delayed : bool
        If True, will use :func:`dask.delayed` to dispatch computations and
        allow mod:`dask` to execute the grid search in parallel (see note
        above).

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the point forces. Same as
        *force_coords* if it is not None. Otherwise, same as the data locations
        used to fit the spline.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.SplineCV.grid` and :meth:`~verde.SplineCV.scatter`
        methods.
    scores_ : array
        The mean cross-validation score for each parameter combination. If
        ``delayed=True``, will be a list of :func:`dask.delayed` objects (see
        note above).
    mindist_ : float
        The optimal value for the *mindist* parameter.
    damping_ : float
        The optimal value for the *damping* parameter.
    spline_ : :class:`verde.Spline`
        A fitted :class:`~verde.Spline` with the optimal configuration
        parameters.

    See also
    --------
    Spline : The bi-harmonic spline
    cross_val_score : Score an estimator/gridder using cross-validation

    """

    def __init__(
        self,
        mindists=(1e3, 10e3, 100e3),
        dampings=(1e-10, 1e-5, 1e-1),
        force_coords=None,
        engine="auto",
        cv=None,
        client=None,
        delayed=False,
    ):
        super().__init__()
        self.dampings = dampings
        self.mindists = mindists
        self.force_coords = force_coords
        self.engine = engine
        self.cv = cv
        self.client = client
        self.delayed = delayed
        if client is not None:
            warnings.warn(
                "The 'client' parameter of 'verde.SplineCV' is "
                "deprecated and will be removed in Verde 2.0.0. "
                "Use the 'delayed' parameter instead.",
                DeprecationWarning,
            )

    def fit(self, coordinates, data, weights=None):
        """
        Fit the spline to the given data and automatically tune parameters.

        For each combination of the parameters given, computes the mean cross
        validation score using :func:`verde.cross_val_score` and the given CV
        splitting class (the *cv* parameter of this class). The configuration
        with the best score is then chosen and used to fit the entire dataset.

        The data region is captured and used as default for the
        :meth:`~verde.SplineCV.grid` and :meth:`~verde.SplineCV.scatter`
        methods.

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
        parameter_sets = [
            dict(
                mindist=combo[0],
                damping=combo[1],
                engine=self.engine,
                force_coords=self.force_coords,
            )
            for combo in itertools.product(self.mindists, self.dampings)
        ]
        if self.client is not None:
            # Deprecated and will be removed in v2.0.0
            scores = []
            for params in parameter_sets:
                spline = Spline(**params)
                scores.append(
                    self.client.submit(
                        cross_val_score,
                        spline,
                        coordinates=coordinates,
                        data=data,
                        weights=weights,
                        cv=self.cv,
                    )
                )
            scores = [np.mean(score.result()) for score in scores]
        else:
            scores = []
            for params in parameter_sets:
                spline = Spline(**params)
                score = cross_val_score(
                    spline,
                    coordinates=coordinates,
                    data=data,
                    weights=weights,
                    cv=self.cv,
                    delayed=self.delayed,
                )
                scores.append(dispatch(np.mean, delayed=self.delayed)(score))
        best = dispatch(np.argmax, delayed=self.delayed)(scores)
        if self.delayed:
            best = best.compute()
        else:
            scores = np.asarray(scores)
        self.spline_ = Spline(**parameter_sets[best])
        self.spline_.fit(coordinates, data, weights=weights)
        self.scores_ = scores
        return self

    @property
    def force_(self):
        "The estimated forces that fit the data."
        return self.spline_.force_

    @property
    def region_(self):
        "The bounding region of the data used to fit the spline"
        return self.spline_.region_

    @property
    def damping_(self):
        "The optimal damping parameter"
        return self.spline_.damping

    @property
    def mindist_(self):
        "The optimal mindist parameter"
        return self.spline_.mindist

    @property
    def force_coords_(self):
        "The optimal force locations"
        return self.spline_.force_coords_

    def predict(self, coordinates):
        """
        Evaluate the best estimated spline on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.SplineCV.fit`).

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
        check_is_fitted(self, ["spline_"])
        return self.spline_.predict(coordinates)


class Spline(BaseGridder):
    r"""
    Biharmonic spline interpolation using Green's functions.

    This gridder assumes Cartesian coordinates.

    Implements the 2D splines of [Sandwell1987]_. The Green's function for the
    spline corresponds to the elastic deflection of a thin sheet subject to a
    vertical force. For an observation point at the origin and a force at the
    coordinates given by the vector :math:`\mathbf{x}`, the Green's function
    is:

    .. math::

        g(\mathbf{x}) = \|\mathbf{x}\|^2 \left(\log \|\mathbf{x}\| - 1\right)

    In practice, this function is not defined for data points that coincide
    with a force. To prevent this, a fudge factor is added to
    :math:`\|\mathbf{x}\|`.

    The interpolation is performed by estimating forces that produce
    deflections that fit the observed data (using least-squares). Then, the
    interpolated points can be evaluated at any location.

    By default, the forces will be placed at the same points as the input data
    given to :meth:`~verde.Spline.fit`. This configuration provides an exact
    solution on top of the data points. However, this solution can be unstable
    for certain configurations of data points.

    Approximate (and more stable) solutions can be obtained by applying damping
    regularization to smooth the estimated forces (and interpolated values) or
    by not using the data coordinates to position the forces (use the
    *force_coords* parameter).

    Data weights can be used during fitting but only have an any effect when
    using the approximate solutions.

    Before fitting, the Jacobian (design, sensitivity, feature, etc) matrix for
    the spline is normalized using
    :class:`sklearn.preprocessing.StandardScaler` without centering the mean so
    that the transformation can be undone in the estimated forces.

    Parameters
    ----------
    mindist : float
        A minimum distance between the point forces and data points. Needed
        because the Green's functions are singular when forces and data points
        coincide. Acts as a fudge factor.
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated forces. If None, no
        regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), then will be set to the data coordinates used to fit the
        spline.
    engine : str
        Computation engine for the Jacobian matrix and prediction. Can be
        ``'auto'``, ``'numba'``, or ``'numpy'``. If ``'auto'``, will use numba
        if it is installed or numpy otherwise. The numba version is
        multi-threaded and usually faster, which makes fitting and predicting
        faster.

    Attributes
    ----------
    force_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the point forces. Same as
        *force_coords* if it is not None. Otherwise, same as the data locations
        used to fit the spline.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Spline.grid` and :meth:`~verde.Spline.scatter` methods.

    See also
    --------
    SplineCV : Cross-validated version of the bi-harmonic spline

    """

    def __init__(self, mindist=1e-5, damping=None, force_coords=None, engine="auto"):
        super().__init__()
        self.mindist = mindist
        self.damping = damping
        self.force_coords = force_coords
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
        if self.force_coords is None:
            self.force_coords_ = tuple(i.copy() for i in n_1d_arrays(coordinates, n=2))
        else:
            self.force_coords_ = self.force_coords
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
        check_is_fitted(self, ["force_"])
        shape = np.broadcast(*coordinates[:2]).shape
        force_east, force_north = n_1d_arrays(self.force_coords_, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        data = np.empty(east.size, dtype=east.dtype)
        if parse_engine(self.engine) == "numba":
            data = predict_numba(
                east, north, force_east, force_north, self.mindist, self.force_, data
            )
        else:
            data = predict_numpy(
                east, north, force_east, force_north, self.mindist, self.force_, data
            )
        return data.reshape(shape)

    def jacobian(self, coordinates, force_coords, dtype="float64"):
        """
        Make the Jacobian matrix for the 2D biharmonic spline.

        Each column of the Jacobian is the Green's function for a single force
        evaluated on all observation points [Sandwell1987]_.

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
            The (n_data, n_forces) Jacobian matrix.

        """
        force_east, force_north = n_1d_arrays(force_coords, n=2)
        east, north = n_1d_arrays(coordinates, n=2)
        jac = np.empty((east.size, force_east.size), dtype=dtype)
        if parse_engine(self.engine) == "numba":
            jac = jacobian_numba(
                east, north, force_east, force_north, self.mindist, jac
            )
        else:
            jac = jacobian_numpy(
                east, north, force_east, force_north, self.mindist, jac
            )
        return jac


def warn_weighted_exact_solution(spline, weights):
    """
    Warn the user that a weights doesn't work for the exact solution.

    Parameters
    ----------
    spline : estimator
        The spline instance that we'll check. Needs to have the ``damping``
        attribute.
    weights : array or None
        The weights given to fit.

    """
    # Check if we're using weights without damping and warn the user that it
    # might not have any effect.
    if weights is not None and spline.damping is None:
        warnings.warn(
            "Weights might have no effect if no regularization is used. "
            "Use damping or specify force positions that are different from the data."
        )


def greens_func(east, north, mindist):
    "Calculate the Green's function for the Bi-Harmonic Spline"
    distance = np.sqrt(east ** 2 + north ** 2)
    # The mindist factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += mindist
    return (distance ** 2) * (np.log(distance) - 1)


def predict_numpy(east, north, force_east, force_north, mindist, forces, result):
    "Calculate the predicted data using numpy."
    result[:] = 0
    for j in range(forces.size):
        green = greens_func(east - force_east[j], north - force_north[j], mindist)
        result += green * forces[j]
    return result


def jacobian_numpy(east, north, force_east, force_north, mindist, jac):
    "Calculate the Jacobian using numpy broadcasting."
    # Reshaping the data to a column vector will automatically build a distance
    # matrix between each data point and force.
    jac[:] = greens_func(
        east.reshape((east.size, 1)) - force_east,
        north.reshape((north.size, 1)) - force_north,
        mindist,
    )
    return jac


@jit(nopython=True, fastmath=True, parallel=True)
def predict_numba(east, north, force_east, force_north, mindist, forces, result):
    "Calculate the predicted data using numba to speed things up."
    for i in numba.prange(east.size):  # pylint: disable=not-an-iterable
        result[i] = 0
        for j in range(forces.size):
            green = GREENS_FUNC_JIT(
                east[i] - force_east[j], north[i] - force_north[j], mindist
            )
            result[i] += green * forces[j]
    return result


@jit(nopython=True, fastmath=True, parallel=True)
def jacobian_numba(east, north, force_east, force_north, mindist, jac):
    "Calculate the Jacobian matrix using numba to speed things up."
    for i in numba.prange(east.size):  # pylint: disable=not-an-iterable
        for j in range(force_east.size):
            jac[i, j] = GREENS_FUNC_JIT(
                east[i] - force_east[j], north[i] - force_north[j], mindist
            )
    return jac


# Jit compile the Green's functions for use in the numba functions
GREENS_FUNC_JIT = jit(nopython=True, fastmath=True)(greens_func)
