"""
Base classes for all gridders.
"""
import xarray as xr
import pandas as pd
from sklearn.base import BaseEstimator

from .coordinates import grid_coordinates, profile_coordinates, scatter_points


class BaseGridder(BaseEstimator):
    """
    Base class for gridders.

    Requires the implementation of the ``predict(coordiantes)`` method. The
    data returned by it should be a 1d or 2d numpy array for scalar data or a
    tuple with 1d or 2d numpy arrays for each component of vector data.

    Subclasses should define a ``residual_`` attribute after fitting that
    contains the data residuals ``self.residual_ = data -
    self.predict(coordinates)``. This is required for compatibility with
    :class:`verde.Chain`.

    Doesn't define any new attributes.

    This is a subclass of :class:`sklearn.base.BaseEstimator` and must abide by
    the same rules of the scikit-learn classes. Mainly:

    * ``__init__`` must **only** assign values to attributes based on the
      parameters it receives. All parameters must have default values.
      Parameter checking should be done in ``fit``.
    * Estimated parameters should be stored as attributes with names ending in
      ``_``.

    The child class can define the following attributes to control the names of
    coordinates and data values in the output ``xarray.Dataset`` and
    ``pandas.DataFrame`` and how distances are calculated:

    * ``coordinate_system``: either ``'cartesian'`` or ``'geographic'``. Will
      influence dimension names and distance calculations. Defaults to
      ``'cartesian'``.
    * ``data_type``: one of ``'scalar'``, ``'vector2d'``, or ``'vector3d'``.
      Defaults to ``'scalar'``.

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
    >>> synthetic = vd.datasets.CheckerBoard().fit(region=(0, 5, -10, 8))
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
        Interpolate data on the given set of points.

        This method is required for all other methods related to interpolation.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each computation point. Should be in
            the following order: (easting, northing, vertical, ...)

        Returns
        -------
        data : array or tuple of arrays
            The data values interpolated on the given points. If the data are
            scalars, then it should be a single array. If data are vectors,
            then should be a tuple with a separate array for each vector
            component. The order of components should be: east, north,
            vertical.

        """
        raise NotImplementedError()

    def grid(self, region=None, shape=None, spacing=None, adjust='spacing',
             dims=None, data_names=None, projection=None):
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
            The names of the northing and easting data dimensions,
            respectively, in the output grid. Defaults to
            ``['northing', 'easting']`` for Cartesian grids and
            ``['latitude', 'longitude']`` for geographic grids.
        data_names : list of None
            The name(s) of the data variables in the output grid. Defaults to
            ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.

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
        data_names = get_data_names(self, data_names)
        region = get_region(self, region)
        coordinates = grid_coordinates(region, shape=shape, spacing=spacing,
                                       adjust=adjust)
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates)))
        coords = {dims[1]: coordinates[0][0, :], dims[0]: coordinates[1][:, 0]}
        attrs = {'metadata': 'Generated by {}'.format(repr(self))}
        data_vars = {name: (dims, value, attrs)
                     for name, value in zip(data_names, data)}
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def scatter(self, region=None, size=300, random_state=0, dims=None,
                data_names=None, projection=None):
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
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Defaults to
            ``['northing', 'easting']`` for Cartesian grids and
            ``['latitude', 'longitude']`` for geographic grids.
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values on a random set of points.

        """
        dims = get_dims(self, dims)
        data_names = get_data_names(self, data_names)
        region = get_region(self, region)
        coordinates = scatter_points(region, size, random_state)
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates)))
        columns = [(dims[0], coordinates[1]), (dims[1], coordinates[0])]
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def profile(self, point1, point2, size, dims=None, data_names=None,
                projection=None):
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
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Defaults to
            ``['northing', 'easting']`` for Cartesian grids and
            ``['latitude', 'longitude']`` for geographic grids.
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values along the profile.

        """
        coordsys = getattr(self, 'coordinate_system', 'cartesian')
        dims = get_dims(self, dims)
        data_names = get_data_names(self, data_names)
        east, north, distances = profile_coordinates(
            point1, point2, size, coordinate_system=coordsys)
        if projection is None:
            data = check_data(self.predict((east, north)))
        else:
            data = check_data(self.predict(projection(east, north)))
        columns = [(dims[0], north), (dims[1], east), ('distance', distances)]
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
    valid_coords = ['cartesian', 'geographic']
    coords = getattr(instance, 'coordinate_system', 'cartesian')
    if coords not in valid_coords:
        raise ValueError(
            "Invalid coordinate system for {}: '{}'. Must be one of {}."
            .format(instance.__class__.__name__, coords, str(valid_coords)))
    if coords == 'geographic':
        return ('latitude', 'longitude')
    return ('northing', 'easting')


def get_data_names(instance, data_names):
    """
    Get default names for data fields for an instance if not given as arguments

    Examples
    --------

    >>> grd = BaseGridder()
    >>> get_data_names(grd, data_names=None)
    ('scalars',)
    >>> grd.data_type = 'vector2d'
    >>> get_data_names(grd, data_names=None)
    ('east_component', 'north_component')
    >>> grd.data_type = 'vector3d'
    >>> get_data_names(grd, data_names=None)
    ('east_component', 'north_component', 'vertical_component')
    >>> get_data_names(grd, data_names=('ringo', 'george'))
    ('ringo', 'george')

    """
    if data_names is not None:
        return data_names
    valid_types = ['scalar', 'vector2d', 'vector3d']
    data_type = getattr(instance, 'data_type', 'scalar')
    if data_type not in valid_types:
        raise ValueError(
            "Invalid data type for {}: '{}'. Must be one of {}."
            .format(instance.__class__.__name__, data_type, str(valid_types)))
    if data_type == 'vector2d':
        return ('east_component', 'north_component')
    elif data_type == 'vector3d':
        return ('east_component', 'north_component', 'vertical_component')
    return ('scalars',)


def get_region(instance, region):
    """
    Get the region attribute stored in instance if one is not provided.
    """
    if region is None:
        if not hasattr(instance, 'region_'):
            raise ValueError(
                "No default region found. Argument must be supplied.")
        region = getattr(instance, 'region_')
    return region


def check_data(data):
    """
    Check the data returned by predict.
    If the data is a single array, return it as a tuple with a single element.

    Examples
    --------

    >>> check_data([1, 2, 3])
    ([1, 2, 3],)
    >>> check_data(([1, 2], [3, 4]))
    ([1, 2], [3, 4])
    """
    if not isinstance(data, tuple):
        data = (data,)
    return data
