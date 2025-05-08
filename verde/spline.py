# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Biharmonic splines in 2D.
"""
import warnings

import numba
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_fit_input, least_squares, n_1d_arrays
from .coordinates import get_region
from .model_selection import cross_val_score
from .utils import dispatch


class SplineCV(BaseGridder):
    r"""
    Cross-validated biharmonic spline interpolation.

    Similar to :class:`verde.Spline` but automatically chooses the best
    *damping* parameter using cross-validation. Tests all the given *dampings*
    and selects the maximum (or minimum) mean cross-validation score (i.e.,
    a grid search).

    This can optionally run in parallel using :mod:`dask`. To do this, use
    ``delayed=True`` to dispatch computations with
    :func:`dask.delayed.delayed`. In this case, each fit and score operation of
    the grid search will be performed in parallel.

    .. note::

        When using *delayed*, the ``scores_`` attribute will be
        :func:`dask.delayed.delayed` objects instead of the actual scores. This
        is because the scores are an intermediate step in the computations and
        their results are not stored. If you need the scores, run
        :func:`dask.compute` on ``scores_`` to calculate them. Be warned that
        **this will run the grid search again**. It might still be faster than
        serial execution but not necessarily.

    Other cross-validation generators from :mod:`sklearn.model_selection` can
    be used by passing them through the *cv* argument.

    Parameters
    ----------
    dampings : iterable or 1d array
        List (or other iterable) of *damping* parameter values to try. Is the
        positive damping regularization parameter. Controls how much smoothness
        is imposed on the estimated forces. If None, no regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), then will be set to the data coordinates the first time
        :meth:`~verde.SplineCV.fit` is called.
    cv : None or cross-validation generator
        Any scikit-learn cross-validation generator. If not given, will use the
        default set by :func:`verde.cross_val_score`.
    delayed : bool
        If True, will use :func:`dask.delayed.delayed` to dispatch computations
        and allow mod:`dask` to execute the grid search in parallel (see note
        above).
    scoring : None, str or callable
        The scoring function (or name of a function) used for cross-validation.
        Must be known to scikit-learn. See the description of *scoring* in
        :func:`sklearn.model_selection.cross_val_score` for details. If None,
        will fall back to the :meth:`verde.Spline.score` method.

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
        ``delayed=True``, will be a list of :func:`dask.delayed.delayed`
        objects (see note above).
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
        dampings=(1e-10, 1e-5, 1e-1),
        force_coords=None,
        cv=None,
        delayed=False,
        scoring=None,
    ):
        super().__init__()
        self.dampings = dampings
        self.force_coords = force_coords
        self.cv = cv
        self.delayed = delayed
        self.scoring = scoring

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
            {"damping": damping, "force_coords": self.force_coords}
            for damping in self.dampings
        ]
        if self.delayed:
            parallel = False
        else:
            parallel = True
        scores = []
        for params in parameter_sets:
            spline = Spline(**params, parallel=parallel)
            score = cross_val_score(
                spline,
                coordinates=coordinates,
                data=data,
                weights=weights,
                cv=self.cv,
                delayed=self.delayed,
                scoring=self.scoring,
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
        self.force_ = self.spline_.force_
        self.region_ = self.spline_.region_
        self.damping_ = self.spline_.damping
        self.force_coords_ = self.spline_.force_coords_
        return self

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
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated forces. If None, no
        regularization is used.
    force_coords : None or tuple of arrays
        The easting and northing coordinates of the point forces. If None
        (default), then will be set to the data coordinates used to fit the
        spline.
    parallel : bool
        Whether or not to run computations in parallel (multithreaded).
        **WARNING:** Using ``parallel=True`` inside a ``ThreadPoolExecutor`` or
        :func:`dask.delayed` context will cause crashes on Mac if :mod:`numba`
        was installed with pip (this won't happen if you use conda). Default is
        True.

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

    def __init__(self, damping=None, force_coords=None, parallel=True):
        super().__init__()
        self.damping = damping
        self.force_coords = force_coords
        self.parallel = parallel

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
        if self.parallel:
            predict = predict_parallel
        else:
            predict = predict_serial
        data = predict(east, north, force_east, force_north, self.force_, data)
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
        if self.parallel:
            jacobian = jacobian_parallel
        else:
            jacobian = jacobian_serial
        jac = jacobian(east, north, force_east, force_north, jac)
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
            "Use damping or specify force positions that are different from the data.",
            stacklevel=2,
        )


@numba.jit(nopython=True)
def greens_function(east, north):
    "Calculate the Green's function for the Bi-Harmonic Spline"
    distance = np.sqrt(east**2 + north**2)
    # Calculate this way instead of x²(log(x) - 1) to avoid calculating log of
    # 0. The limit for this as x->0 is 0 anyway. This is good for small values
    # of distance but for larger distances it fails because of an overflow in
    # the power operator. Saw this on Wikipedia but there was no citation in
    # December 2022: https://en.wikipedia.org/wiki/Radial_basis_function
    if distance < 1:
        result = distance * (np.log(distance**distance) - distance)
    else:
        result = distance**2 * (np.log(distance) - 1)
    return result


def _predict(east, north, force_east, force_north, forces, result):
    "Calculate the predicted data using numba to speed things up."
    for i in numba.prange(east.size):
        result[i] = 0
        for j in range(forces.size):
            green = greens_function(east[i] - force_east[j], north[i] - force_north[j])
            result[i] += green * forces[j]
    return result


def _jacobian(east, north, force_east, force_north, jac):
    "Calculate the Jacobian matrix using numba to speed things up."
    for i in numba.prange(east.size):
        for j in range(force_east.size):
            jac[i, j] = greens_function(
                east[i] - force_east[j], north[i] - force_north[j]
            )
    return jac


predict_serial = numba.jit(_predict, nopython=True)
predict_parallel = numba.jit(_predict, nopython=True, parallel=True)
jacobian_serial = numba.jit(_jacobian, nopython=True)
jacobian_parallel = numba.jit(_jacobian, nopython=True, parallel=True)
