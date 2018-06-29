"""
Vector gridding using elasticity Green's functions from Sandwell and Wessel
(2016).
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import check_fit_input
from .spline import Spline
from .coordinates import get_region


class Vector2D(Spline):
    r"""
    Elastically coupled interpolation of 2-component vector data.

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
    region : None or tuple
        If not None, then the boundaries (``[W, E, S, N]``) used to generate a
        regular grid of forces. If None is given, then will use the boundaries
        of data given to the first call to :meth:`~verde.Vector2D.fit`.

    Attributes
    ----------
    forces_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the forces.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Vector2D.grid` and :meth:`~verde.Vector2D.scatter`
        methods.

    See also
    --------
    verde.vector2d_jacobian : Jacobian matrix for the 2D elastic deformation

    """

    def __init__(
        self,
        poisson=0.5,
        fudge=1e-5,
        damping=None,
        shape=None,
        spacing=None,
        region=None,
    ):
        self.poisson = poisson
        super().__init__(
            fudge=fudge, damping=damping, shape=shape, spacing=spacing, region=region
        )

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given 2-component vector data.

        The data region is captured and used as default for the
        :meth:`~verde.Vector2D.grid` and :meth:`~verde.Vector2D.scatter`
        methods.

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
        self.force_coords_ = self._get_force_coordinates(coordinates)
        if any(w is not None for w in weights):
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        self._check_weighted_exact_solution(weights)
        data = np.concatenate([i.ravel() for i in data])
        jacobian = vector2d_jacobian(
            coordinates[:2], self.force_coords_, self.poisson, self.fudge
        )
        self.force_ = self._estimate_forces(jacobian, data, weights)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.Vector2D.fit`).

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
        check_is_fitted(self, ["force_", "force_coords_"])
        jac = vector2d_jacobian(
            coordinates[:2], self.force_coords_, self.poisson, self.fudge
        )
        cast = np.broadcast(*coordinates[:2])
        npoints = cast.size
        components = jac.dot(self.force_).reshape((2, npoints))
        return tuple(comp.reshape(cast.shape) for comp in components)


def vector2d_jacobian(
    coordinates, force_coordinates, poisson, fudge=1e-5, dtype="float32"
):
    """
    Make the Jacobian matrix for the 2D coupled elastic deformation.

    Follows [SandwellWessel2016]_.

    The Jacobian is segmented into 4 parts, each relating a force component to
    a data component::

        | J_ee  J_ne |*|f_e| = |d_e|
        | J_ne  J_nn | |f_n|   |d_n|

    The forces and data are assumed to be stacked into 1D arrays with the east
    component on top of the north component.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    force_coordinates : tuple of arrays
        Arrays with the coordinates of each vertical force. Should be in the
        following order: (easting, northing, vertical, ...). Only easting and
        northing will be used, all subsequent coordinates will be ignored.
    poisson ; float
        The Poisson's ratio for the elastic deformation Green's functions.
        A value of -1 will lead to uncoupled interpolation of
        the east and north data components (the ``J_ne`` component of the
        Jacobian is null).
    fudge : float
        The positive fudge factor applied to the Green's function to avoid
        singularities.
    dtype : str or numpy dtype
        The type of the Jacobian array.

    Returns
    -------
    jacobian : 2D array
        The (n_data*2, n_forces*2) Jacobian matrix.

    See also
    --------
    verde.Vector2D : Coupled gridder for 2-component vector data

    """
    force_coordinates = [np.atleast_1d(i).ravel() for i in force_coordinates[:2]]
    coordinates = [np.atleast_1d(i).ravel() for i in coordinates[:2]]
    npoints = coordinates[0].size
    nforces = force_coordinates[0].size
    # Reshaping the data coordinates to a column vector will automatically
    # build a distance matrix between each data point and force.
    east, north = (
        datac.reshape((npoints, 1)) - forcec
        for datac, forcec in zip(coordinates, force_coordinates)
    )
    distance = np.hypot(east, north, dtype=dtype)
    # The fudge factor helps avoid singular matrices when the force and
    # computation point are too close
    distance += fudge
    # Pre-compute common terms for the Green's functions of each component
    ln_r = (3 - poisson) * np.log(distance)
    over_r2 = (1 + poisson) / distance ** 2
    jac = np.empty((npoints * 2, nforces * 2), dtype=dtype)
    jac[:npoints, :nforces] = ln_r + over_r2 * north ** 2  # J_ee
    jac[npoints:, nforces:] = ln_r + over_r2 * east ** 2  # J_nn
    jac[:npoints, nforces:] = -over_r2 * east * north  # J_ne
    jac[npoints:, :nforces] = jac[:npoints, nforces:]  # J is symmetric
    return jac
