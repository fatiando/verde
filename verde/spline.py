"""
Biharmonic splines in 2D.
"""
from warnings import warn

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from .base import BaseGridder, check_fit_input
from .coordinates import grid_coordinates, get_region


class Spline(BaseGridder):
    r"""
    Biharmonic spline interpolation using Green's functions.

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
    by distributing the forces on a regular grid instead (using arguments
    *spacing*, *shape*, and/or *region*).

    Data weights can be used during fitting but only have an any effect when
    using the approximate solutions.

    The Jacobian (design, sensitivity, feature, etc) matrix for the spline
    is normalized using :class:`sklearn.preprocessing.StandardScaler` without
    centering the mean so that the transformation can be undone in the
    estimated forces.

    Parameters
    ----------
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
    region : None or tuple
        If not None, then the boundaries (``[W, E, S, N]``) used to generate a
        regular grid of forces. If None is given, then will use the boundaries
        of data given to the first call to :meth:`~verde.Spline.fit`.

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

    def __init__(self, fudge=1e-5, damping=None, shape=None, spacing=None, region=None):
        self.damping = damping
        self.shape = shape
        self.spacing = spacing
        self.fudge = fudge
        self.region = region

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
        # Remove pre-existing force coordinates when fitting for a second time.
        if hasattr(self, "force_coords_"):
            del self.force_coords_
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        self._check_weighted_exact_solution(weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        jacobian = self.jacobian(coordinates[:2])
        self.force_ = self._estimate_forces(jacobian, data, weights)
        return self

    def _estimate_forces(self, jacobian, data, weights):
        """
        Estimate forces that fit the data using least-squares. Scales the
        Jacobian matrix to have unit standard deviation. This helps balance the
        regularization and the difference between forces.
        """
        if jacobian.shape[0] < jacobian.shape[1]:
            warn(
                " ".join(
                    [
                        "Under-determined problem detected",
                        "(ndata, nparams)={}.".format(jacobian.shape),
                        "Configuration of forces:",
                        "spacing={} shape={}".format(self.spacing, self.shape),
                    ]
                )
            )
        scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
        jacobian = scaler.fit_transform(jacobian)
        if self.damping is None:
            regr = LinearRegression(fit_intercept=False, normalize=False)
        else:
            regr = Ridge(alpha=self.damping, fit_intercept=False, normalize=False)
        regr.fit(jacobian, data.ravel(), sample_weight=weights)
        # Undo the scaling so that we can use forces on the unscaled Jacobian
        # later on.
        forces = regr.coef_ / scaler.scale_
        return forces

    def _get_force_coordinates(self, coordinates):
        """
        Generate arrays for the force coordinates depending on the
        configuration of the spline.
        """
        # Set the force positions. If no shape or spacing are given, then they
        # will coincide with the data points. This will only happen once so the
        # model won't change during later fits.
        if self.shape is None and self.spacing is None:
            coords = tuple(i.copy() for i in coordinates[:2])
        else:
            if self.region is None:
                self.region = get_region(coordinates)
            coords = grid_coordinates(
                self.region, shape=self.shape, spacing=self.spacing
            )
        return coords

    def _check_weighted_exact_solution(self, weights):
        """
        If a weighted exact solution is requested, warn the user that it won't
        work.
        """
        # Check if we're using weights with the exact solution and warn the
        # user that it won't have any effect.
        exact = all(i is None for i in [self.damping, self.spacing, self.shape])
        if weights is not None and exact:
            warn(
                "Weights are ignored for the exact solution. "
                "Use damping or specify a shape/spacing for the forces."
            )

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
        jac = self.jacobian(coordinates[:2])
        shape = np.broadcast(*coordinates[:2]).shape
        return jac.dot(self.force_).reshape(shape)

    def jacobian(self, coordinates, dtype="float64"):
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
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_forces) Jacobian matrix.

        """
        if not hasattr(self, "force_coords_"):
            self.force_coords_ = self._get_force_coordinates(coordinates)
        force_easting, force_northing = (
            np.atleast_1d(i).ravel() for i in self.force_coords_[:2]
        )
        easting, northing = (np.atleast_1d(i).ravel() for i in coordinates[:2])
        # Reshaping the data to a column vector will automatically build a
        # distance matrix between each data point and force.
        distance = np.hypot(
            easting.reshape((easting.size, 1)) - force_easting.ravel(),
            northing.reshape((northing.size, 1)) - force_northing.ravel(),
            dtype=dtype,
        )
        # The fudge factor helps avoid singular matrices when the force and
        # computation point are too close
        distance += self.fudge
        return (distance ** 2) * (np.log(distance) - 1)
