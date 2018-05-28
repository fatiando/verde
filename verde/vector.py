"""
Vector gridding using elasticity Green's functions from Sandwell and Wessel
(2016).
"""
from warnings import warn

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from .base import BaseGridder, check_fit_input
from .coordinates import grid_coordinates, get_region


class Vector2D(BaseGridder):
    r"""
    Coupled interpolation of 2-component vector data.

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
    locations are set the first time :meth:`~verde.Vector2D.fit` is called.
    Subsequent calls will fit using the same force locations as the first call.
    This configuration results in an exact prediction at the data points but
    can be unstable.

    [SandwellWessel2016]_ stabilize the solution using Singular Value
    Decomposition but we use ridge regression instead. The regularization can
    be controlled using the *damping* argument. Alternatively, we also allow
    forces to be placed on a regular grid using the *spacing*, *shape*, and/or
    *region* arguments. Regularization or forces on a grid will result in a
    least-squares estimate at the data points, not an exact solution. Note that
    the least-squares solution is required for data weights to have any effect.

    The Jacobian (design, sensitivity, feature, etc) matrix for the spline
    is normalized using :class:`sklearn.preprocessing.StandardScaler` without
    centering the mean so that the transformation can be undone in the
    estimated forces.

    Parameters
    ----------
    poisson : float
        The Poisson's ratio for the elastic deformation Green's functions.
        Default is 0.5. A value of -1 will lead to uncoupled interpolation of
        the east and north data components.
    fudge : float
        The positive fudge factor applied to the Green's function to avoid
        singularities.
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated forces. If None, no
        regularization is used.
    shape : None or tuple
        If not None, then should be the shape of the regular grid of forces.
        See :func:`verde.grid_coordinates` for details.
    spacing : None or float or tuple
        If not None, then should be the spacing of the regular grid of forces.
        See :func:`verde.grid_coordinates` for details.
    region

    Attributes
    ----------
    forces_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the forces.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Vector.grid` and :meth:`~verde.Vector.scatter` methods.

    See also
    --------
    verde.vector2d_jacobian : Jacobian matrix for the 2D elastic deformation

    """

    def __init__(self, poisson=0.5, fudge=1e-5, damping=None, shape=None,
                 spacing=None, force_region=None):
        self.poisson = poisson
        self.damping = damping
        self.shape = shape
        self.spacing = spacing
        self.fudge = fudge
        self.force_region = force_region

    def fit(self, coordinates, data, weights=None):
        """

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
        coordinates, data, weights = check_fit_input(coordinates, data,
                                                     weights)
        exact_solution = all(i is None
                             for i in [self.damping, self.spacing, self.shape])
        if any(w is not None for w in weights):
            if exact_solution:
                warn(' '.join([
                    "Weights are ignored for the exact solution.",
                    "Use damping or specify a shape/spacing for the forces."]))
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        self.region_ = get_region(coordinates[:2])
        if self.force_region is None:
            self.force_region = self.region_
        # Set the force positions. If no shape or spacing are given, then they
        # will coincide with the data points.
        if self.shape is None and self.spacing is None:
            self.force_coords_ = tuple(i.copy() for i in coordinates[:2])
        else:
            self.force_coords_ = grid_coordinates(self.force_region,
                                                  shape=self.shape,
                                                  spacing=self.spacing)
        jac = vector2d_jacobian(coordinates[:2], self.force_coords_,
                                self.poisson, self.fudge)
        if jac.shape[0] < jac.shape[1]:
            warn(" ".join([
                "Under-determined problem detected:",
                "(ndata, nparams)={}".format(jac.shape),
                "spacing={} shape={}".format(self.spacing, self.shape)]))
        scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
        jac = scaler.fit_transform(jac)
        if self.damping is None:
            regr = LinearRegression(fit_intercept=False, normalize=False)
        else:
            regr = Ridge(alpha=self.damping, fit_intercept=False,
                         normalize=False)
        regr.fit(jac, np.concatenate([i.ravel() for i in data]),
                 sample_weight=weights)
        self.force_ = regr.coef_/scaler.scale_
        return self

    def predict(self, coordinates):
        """
        Evaluate the estimated spline on the given set of points.

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
            The data values evaluated on the given points.

        """
        check_is_fitted(self, ['force_', 'force_coords_'])
        jac = vector2d_jacobian(coordinates[:2], self.force_coords_,
                                self.poisson, self.fudge)
        cast = np.broadcast(*coordinates[:2])
        npoints = cast.size
        components = jac.dot(self.force_).reshape((2, npoints))
        return tuple(comp.reshape(cast.shape) for comp in components)


def vector2d_jacobian(coordinates, force_coordinates, poisson, fudge=1e-5,
                      dtype="float32"):
    """

    """
    force_easting, force_northing = (np.atleast_1d(i).ravel()
                                     for i in force_coordinates[:2])
    easting, northing = (np.atleast_1d(i).ravel() for i in coordinates[:2])

    npoints = easting.size
    nforces = force_easting.size

    # Reshaping the data to a column vector will automatically build a
    # distance matrix between each data point and force.
    e = easting.reshape((npoints, 1)) - force_easting
    n = northing.reshape((npoints, 1)) - force_northing
    distance = np.hypot(e, n, dtype=dtype)
    # The fudge factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += fudge
    # Pre-compute common terms
    ln_r = (3 - poisson)*np.log(distance)
    over_r2 = (1 + poisson)/distance**2
    # Fill in the Jacobian
    jac = np.empty((npoints*2, nforces*2), dtype=dtype)
    jac[:npoints, :nforces] = ln_r + over_r2*n**2  # EE
    jac[npoints:, nforces:] = ln_r + over_r2*e**2  # NN
    jac[:npoints, nforces:] = jac[npoints:, :nforces] = -over_r2*e*n  # EN
    return jac
