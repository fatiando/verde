"""
Base classes for all gridders.
"""
from warnings import warn

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from .coordinates import grid_coordinates, profile_coordinates, scatter_points
from .utils import check_data


class BaseGridder(BaseEstimator):
    """
    Base class for gridders.

    Most methods of this class requires the implementation of a
    :meth:`~verde.base.BaseGridder.predict` method. The data returned by it should
    be a 1d or 2d numpy array for scalar data or a tuple with 1d or 2d numpy
    arrays for each component of vector data.

    The :meth:`~verde.base.BaseGridder.filter` method requires the implementation of
    a :meth:`~verde.base.BaseGridder.fit` method to fit the gridder model to data.

    Doesn't define any new attributes.

    This is a subclass of :class:`sklearn.base.BaseEstimator` and must abide by
    the same rules of the scikit-learn classes. Mainly:

    * ``__init__`` must **only** assign values to attributes based on the
      parameters it receives. All parameters must have default values.
      Parameter checking should be done in ``fit``.
    * Estimated parameters should be stored as attributes with names ending in
      ``_``.

    The child class can define the following attributes to control the names of
    coordinates and how distances are calculated:

    * ``coordinate_system``: either ``'cartesian'`` or ``'geographic'``. Will
      influence dimension names and distance calculations. Defaults to
      ``'cartesian'``.

    Examples
    --------

    Let's create a class that interpolates by attributing the mean value of the
    data to every single point (it's not a very good interpolator).

    >>> import verde as vd
    >>> import numpy as np
    >>> from sklearn.utils.validation import check_is_fitted
    >>> class MeanGridder(vd.base.BaseGridder):
    ...     "Gridder that always produces the mean of all data values"
    ...     def __init__(self, multiplier=1):
    ...         # Init should only assign the parameters to attributes
    ...         self.multiplier = multiplier
    ...     def fit(self, coordiantes, data):
    ...         # Argument checking should be done in fit
    ...         if self.multiplier <= 0:
    ...             raise ValueError('Invalid multiplier {}'
    ...                              .format(self.multiplier))
    ...         self.mean_ = data.mean()*self.multiplier
    ...         # fit should return self so that we can chain operations
    ...         return self
    ...     def predict(self, coordinates):
    ...         # We know the gridder has been fitted if it has the mean
    ...         check_is_fitted(self, ['mean_'])
    ...         return np.ones_like(coordinates[0])*self.mean_
    >>> # Try it on some synthetic data
    >>> synthetic = vd.datasets.CheckerBoard(region=(0, 5, -10, 8))
    >>> data = synthetic.scatter()
    >>> print('{:.4f}'.format(data.scalars.mean()))
    -32.2182
    >>> # Fit the gridder to our synthetic data
    >>> grd = MeanGridder().fit((data.easting, data.northing), data.scalars)
    >>> grd
    MeanGridder(multiplier=1)
    >>> # Interpolate on a regular grid
    >>> grid = grd.grid(region=(0, 5, -10, -8), shape=(30, 20))
    >>> type(grid)
    <class 'xarray.core.dataset.Dataset'>
    >>> np.allclose(grid.scalars, -32.2182)
    True
    >>> # Interpolate along a profile
    >>> profile = grd.profile(point1=(0, -10), point2=(5, -8), size=10)
    >>> type(profile)
    <class 'pandas.core.frame.DataFrame'>
    >>> print(', '.join(['{:.2f}'.format(i) for i in profile.distance]))
    0.00, 0.60, 1.20, 1.80, 2.39, 2.99, 3.59, 4.19, 4.79, 5.39
    >>> print(', '.join(['{:.1f}'.format(i) for i in profile.scalars]))
    -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2

    """

    def predict(self, coordinates):
        """
        Predict data on the given coordinate values. NOT IMPLEMENTED.

        This is a dummy placeholder for an actual method.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).

        Returns
        -------
        data : array
            The data predicted at the give coordinates.

        """
        raise NotImplementedError()

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to observed data. NOT IMPLEMENTED.

        This is a dummy placeholder for an actual method.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).
        data : array or tuple of arrays
            The data values of each data point. If the data has more than one
            component, *data* must be a tuple of arrays (one for each
            component).
        weights : None or array or tuple of arrays
            If not None, then the weights assigned to each data point. If more
            than one data component is provided, you must provide a weights
            array for each data component (if not None).

        Returns
        -------
        self
            This instance of the gridder. Useful to chain operations.

        """
        raise NotImplementedError()

    def filter(self, coordinates, data, weights=None):
        """
        Filter the data through the gridder and produce residuals.

        Calls ``fit`` on the data, evaluates the residuals (data - predicted
        data), and returns the coordinates, residuals, and weights.

        No very useful by itself but this interface makes gridders compatible
        with other processing operations and is used by :class:`verde.Chain` to
        join them together (for example, so you can fit a spline on the
        residuals of a trend).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).
        data : array or tuple of arrays
            The data values of each data point. If the data has more than one
            component, *data* must be a tuple of arrays (one for each
            component).
        weights : None or array or tuple of arrays
            If not None, then the weights assigned to each data point. If more
            than one data component is provided, you must provide a weights
            array for each data component (if not None).

        Returns
        -------
        coordinates, residuals, weights
            The coordinates and weights are same as the input. Residuals are
            the input data minus the predicted data.

        """
        self.fit(coordinates, data, weights)
        data = check_data(data)
        pred = check_data(self.predict(coordinates))
        residuals = tuple(
            datai - predi.reshape(datai.shape) for datai, predi in zip(data, pred)
        )
        if len(residuals) == 1:
            residuals = residuals[0]
        return coordinates, residuals, weights

    def score(self, coordinates, data, weights=None):
        """
        Score the gridder predictions against the given data.

        Calculates the R^2 coefficient of determination of between the
        predicted values and the given data values. A maximum score of 1 means
        a perfect fit. The score can be negative.

        If the data has more than 1 component, the scores of each component
        will be averaged.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).
        data : array or tuple of arrays
            The data values of each data point. If the data has more than one
            component, *data* must be a tuple of arrays (one for each
            component).
        weights : None or array or tuple of arrays
            If not None, then the weights assigned to each data point. If more
            than one data component is provided, you must provide a weights
            array for each data component (if not None).

        Returns
        -------
        score : float
            The R^2 score

        """
        coordinates, data, weights = check_fit_input(
            coordinates, data, weights, unpack=False
        )
        pred = check_data(self.predict(coordinates))
        result = np.mean(
            [
                r2_score(datai.ravel(), predi.ravel(), sample_weight=weighti)
                for datai, predi, weighti in zip(data, pred, weights)
            ]
        )
        return result

    def grid(
        self,
        region=None,
        shape=None,
        spacing=None,
        adjust="spacing",
        dims=None,
        data_names=None,
        projection=None,
    ):
        """
        Interpolate the data onto a regular grid.

        The grid can be specified by either the number of points in each
        dimension (the *shape*) or by the grid node spacing.

        If the given region is not divisible by the desired spacing, either the
        region or the spacing will have to be adjusted. By default, the spacing
        will be rounded to the nearest multiple. Optionally, the East and North
        boundaries of the region can be adjusted to fit the exact spacing
        given. See :func:`verde.grid_coordinates` for more details.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names are provided.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The boundaries of a given region in Cartesian or geographic
            coordinates.
        shape : tuple = (n_north, n_east) or None
            The number of points in the South-North and West-East directions,
            respectively. If *None* and *spacing* is not given, defaults to
            ``(101, 101)``.
        spacing : tuple = (s_north, s_east) or None
            The grid spacing in the South-North and West-East directions,
            respectively.
        adjust : {'spacing', 'region'}
            Whether to adjust the spacing or the region if required. Ignored if
            *shape* is given instead of *spacing*. Defaults to adjusting the
            spacing.
        dims : list or None
            The names of the northing and easting data dimensions, respectively, in the
            output grid. Defaults to ``['northing', 'easting']`` for Cartesian grids and
            ``['latitude', 'longitude']`` for geographic grids. **NOTE: This is an
            exception to the "easting" then "northing" pattern but is required for
            compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output grid. Defaults to
            ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated grid coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic grid from a Cartesian gridder.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        See also
        --------
        verde.grid_coordinates : Generate the coordinate values for the grid.

        """
        if shape is None and spacing is None:
            shape = (101, 101)
        dims = get_dims(self, dims)
        region = get_instance_region(self, region)
        coordinates = grid_coordinates(
            region, shape=shape, spacing=spacing, adjust=adjust
        )
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates)))
        data_names = get_data_names(data, data_names)
        coords = {dims[1]: coordinates[0][0, :], dims[0]: coordinates[1][:, 0]}
        attrs = {"metadata": "Generated by {}".format(repr(self))}
        data_vars = {
            name: (dims, value, attrs) for name, value in zip(data_names, data)
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def scatter(
        self,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        projection=None,
    ):
        """
        Interpolate values onto a random scatter of points.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The boundaries of a given region in Cartesian or geographic
            coordinates.
        size : int
            The number of points to generate.
        random_state : numpy.random.RandomState or an int seed
            A random number generator used to define the state of the random
            permutations. Use a fixed seed to make sure computations are
            reproducible. Use ``None`` to choose a seed automatically
            (resulting in different numbers with each run).
        dims : list or None
            The names of the northing and easting data dimensions, respectively, in the
            output dataframe. Defaults to ``['northing', 'easting']`` for Cartesian
            grids and ``['latitude', 'longitude']`` for geographic grids. **NOTE: This
            is an exception to the "easting" then "northing" pattern but is required for
            compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated scatter coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic scatter from a Cartesian gridder.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values on a random set of points.

        """
        dims = get_dims(self, dims)
        region = get_instance_region(self, region)
        coordinates = scatter_points(region, size, random_state)
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates)))
        data_names = get_data_names(data, data_names)
        columns = [(dims[0], coordinates[1]), (dims[1], coordinates[0])]
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def profile(
        self, point1, point2, size, dims=None, data_names=None, projection=None
    ):
        """
        Interpolate data along a profile between two points.

        Generates the profile using a straight line if the interpolator assumes
        Cartesian data or a great circle if geographic data.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Includes the calculated distance to *point1* for each data point in the
        profile.

        Parameters
        ----------
        point1 : tuple
            The easting and northing coordinates, respectively, of the first
            point.
        point2 : tuple
            The easting and northing coordinates, respectively, of the second
            point.
        size : int
            The number of points to generate.
        dims : list or None
            The names of the northing and easting data dimensions, respectively, in the
            output dataframe. Defaults to ``['northing', 'easting']`` for Cartesian
            grids and ``['latitude', 'longitude']`` for geographic grids. **NOTE: This
            is an exception to the "easting" then "northing" pattern but is required for
            compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated profile coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic profile from a Cartesian gridder.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values along the profile.

        """
        coordsys = getattr(self, "coordinate_system", "cartesian")
        dims = get_dims(self, dims)
        east, north, distances = profile_coordinates(
            point1, point2, size, coordinate_system=coordsys
        )
        if projection is None:
            data = check_data(self.predict((east, north)))
        else:
            data = check_data(self.predict(projection(east, north)))
        data_names = get_data_names(data, data_names)
        columns = [(dims[0], north), (dims[1], east), ("distance", distances)]
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])


def get_dims(instance, dims):
    """
    Get default dimension names for an instance if not given as arguments.

    Examples
    --------

    >>> grd = BaseGridder()
    >>> get_dims(grd, dims=None)
    ('northing', 'easting')
    >>> grd.coordinate_system = 'geographic'
    >>> get_dims(grd, dims=None)
    ('latitude', 'longitude')
    >>> grd.coordinate_system = 'cartesian'
    >>> get_dims(grd, dims=('john', 'paul'))
    ('john', 'paul')

    """
    if dims is not None:
        return dims
    valid_coords = ["cartesian", "geographic"]
    coords = getattr(instance, "coordinate_system", "cartesian")
    if coords not in valid_coords:
        raise ValueError(
            "Invalid coordinate system for {}: '{}'. Must be one of {}.".format(
                instance.__class__.__name__, coords, str(valid_coords)
            )
        )
    if coords == "geographic":
        return ("latitude", "longitude")
    return ("northing", "easting")


def get_data_names(data, data_names):
    """
    Get default names for data fields if none are given based on the data.

    Examples
    --------

    >>> import numpy as np
    >>> east, north, up = [np.arange(10)]*3
    >>> get_data_names((east,), data_names=None)
    ('scalars',)
    >>> get_data_names((east, north), data_names=None)
    ('east_component', 'north_component')
    >>> get_data_names((east, north, up), data_names=None)
    ('east_component', 'north_component', 'vertical_component')
    >>> get_data_names((up, north), data_names=('ringo', 'george'))
    ('ringo', 'george')

    """
    if data_names is not None:
        if len(data) != len(data_names):
            raise ValueError(
                "Data has {} components but only {} names provided: {}".format(
                    len(data), len(data_names), str(data_names)
                )
            )
        return data_names
    data_types = [
        ("scalars",),
        ("east_component", "north_component"),
        ("east_component", "north_component", "vertical_component"),
    ]
    if len(data) > len(data_types):
        raise ValueError(
            " ".join(
                [
                    "Default data names only available for up to 3 components.",
                    "Must provide custom names through the 'data_names' argument.",
                ]
            )
        )
    return data_types[len(data) - 1]


def get_instance_region(instance, region):
    """
    Get the region attribute stored in instance if one is not provided.
    """
    if region is None:
        if not hasattr(instance, "region_"):
            raise ValueError("No default region found. Argument must be supplied.")
        region = getattr(instance, "region_")
    return region


def check_fit_input(coordinates, data, weights, unpack=True):
    """
    Validate the inputs to the fit method of gridders.

    Checks that the coordinates, data, and weights (if given) all have the same
    shape. Weights arrays are raveled.

    Parameters
    ----------
    coordinates : tuple of arrays
        Arrays with the coordinates of each data point. Should be in the
        following order: (easting, northing, vertical, ...).
    data : array or tuple of arrays
        The data values of each data point. Data can have more than one
        component. In such cases, data should be a tuple of arrays.
    weights : None or array
        If not None, then the weights assigned to each data point.
        Typically, this should be 1 over the data uncertainty squared.
        If the data has multiple components, the weights have the same number
        of components.
    unoack : bool
        If False, data and weights will be tuples always. If they are single
        arrays, then they will be returned as a 1-element tuple. If True, will
        unpack the tuples if there is only 1 array in each.

    Returns
    -------
    validated_inputs
        The validated inputs in the same order. If weights are given, will
        ravel the array before returning.

    """
    data = check_data(data)
    weights = check_data(weights)
    if any(i.shape != j.shape for i in coordinates for j in data):
        raise ValueError("Coordinate and data arrays must have the same shape.")
    if any(w is not None for w in weights):
        if len(weights) != len(data):
            raise ValueError(
                "Number of data '{}' and weights '{}' must be equal.".format(
                    len(data), len(weights)
                )
            )
        if any(i.size != j.size for i in weights for j in data):
            raise ValueError("Weights must have the same size as the data array.")
        weights = tuple(i.ravel() for i in weights)
    else:
        weights = tuple([None] * len(data))
    if unpack:
        if len(weights) == 1:
            weights = weights[0]
        if len(data) == 1:
            data = data[0]
    return coordinates, data, weights


def least_squares(jacobian, data, weights, damping=None):
    """
    Estimate forces that fit the data using least-squares. Scales the
    Jacobian matrix to have unit standard deviation. This helps balance the
    regularization and the difference between forces.
    """
    if jacobian.shape[0] < jacobian.shape[1]:
        warn(
            "Under-determined problem detected (ndata, nparams)={}.".format(
                jacobian.shape
            )
        )
    scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
    jacobian = scaler.fit_transform(jacobian)
    if damping is None:
        regr = LinearRegression(fit_intercept=False, normalize=False)
    else:
        regr = Ridge(alpha=damping, fit_intercept=False, normalize=False)
    regr.fit(jacobian, data.ravel(), sample_weight=weights)
    params = regr.coef_ / scaler.scale_
    return params
