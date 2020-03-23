"""
Base classes for all gridders.
"""
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

from ..coordinates import grid_coordinates, profile_coordinates, scatter_points
from .utils import check_data, check_fit_input


class BaseGridder(BaseEstimator):
    """
    Base class for gridders.

    Most methods of this class requires the implementation of a
    :meth:`~verde.base.BaseGridder.predict` method. The data returned by it
    should be a 1d or 2d numpy array for scalar data or a tuple with 1d or 2d
    numpy arrays for each component of vector data.

    The :meth:`~verde.base.BaseGridder.filter` method requires the
    implementation of a :meth:`~verde.base.BaseGridder.fit` method to fit the
    gridder model to data.

    Doesn't define any new attributes.

    This is a subclass of :class:`sklearn.base.BaseEstimator` and must abide by
    the same rules of the scikit-learn classes. Mainly:

    * ``__init__`` must **only** assign values to attributes based on the
      parameters it receives. All parameters must have default values.
      Parameter checking should be done in ``fit``.
    * Estimated parameters should be stored as attributes with names ending in
      ``_``.

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
    >>> np.allclose(grid.scalars, -32.2182)
    True
    >>> # Interpolate along a profile
    >>> profile = grd.profile(point1=(0, -10), point2=(5, -8), size=10)
    >>> print(', '.join(['{:.2f}'.format(i) for i in profile.distance]))
    0.00, 0.60, 1.20, 1.80, 2.39, 2.99, 3.59, 4.19, 4.79, 5.39
    >>> print(', '.join(['{:.1f}'.format(i) for i in profile.scalars]))
    -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2, -32.2

    """

    # The default dimension names for generated outputs
    # (pd.DataFrame, xr.Dataset, etc)
    dims = ("northing", "easting")

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
            For the specific definition of coordinate systems and what these
            names mean, see the class docstring.
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
            For the specific definition of coordinate systems and what these
            names mean, see the class docstring.
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
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        Interpolate the data onto a regular grid.

        The grid can be specified by either the number of points in each
        dimension (the *shape*) or by the grid node spacing. See
        :func:`verde.grid_coordinates` for details. Other arguments for
        :func:`verde.grid_coordinates` can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names will be provided if none are given.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        shape : tuple = (n_north, n_east) or None
            The number of points in the South-North and West-East directions,
            respectively.
        spacing : tuple = (s_north, s_east) or None
            The grid spacing in the South-North and West-East directions,
            respectively.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output grid. Default is determined from the
            ``dims`` attribute of the class. Must be defined in the following
            order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
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
        dims = self._get_dims(dims)
        region = get_instance_region(self, region)
        coordinates = grid_coordinates(region, shape=shape, spacing=spacing, **kwargs)
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
        **kwargs
    ):
        """
        Interpolate values onto a random scatter of points.

        Point coordinates are generated by :func:`verde.scatter_points`. Other
        arguments for this function can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        size : int
            The number of points to generate.
        random_state : numpy.random.RandomState or an int seed
            A random number generator used to define the state of the random
            permutations. Use a fixed seed to make sure computations are
            reproducible. Use ``None`` to choose a seed automatically
            (resulting in different numbers with each run).
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
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
        dims = self._get_dims(dims)
        region = get_instance_region(self, region)
        coordinates = scatter_points(region, size, random_state=random_state, **kwargs)
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(self.predict(projection(*coordinates)))
        data_names = get_data_names(data, data_names)
        columns = [(dims[0], coordinates[1]), (dims[1], coordinates[0])]
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def profile(
        self,
        point1,
        point2,
        size,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        Interpolate data along a profile between two points.

        Generates the profile along a straight line assuming Cartesian
        distances. Point coordinates are generated by
        :func:`verde.profile_coordinates`. Other arguments for this function
        can be passed as extra keyword arguments (``kwargs``) to this method.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Includes the calculated Cartesian distance from *point1* for each data
        point in the profile.

        To specify *point1* and *point2* in a coordinate system that would
        require projection to Cartesian (geographic longitude and latitude, for
        example), use the ``projection`` argument. With this option, the input
        points will be projected using the given projection function prior to
        computations. The generated Cartesian profile coordinates will be
        projected back to the original coordinate system. **Note that the
        profile points are evenly spaced in projected coordinates, not the
        original system (e.g., geographic)**.

        .. warning::

            **The profile calculation method with a projection has changed in
            Verde 1.4.0**. Previous versions generated coordinates (assuming
            they were Cartesian) and projected them afterwards. This led to
            "distances" being incorrectly handled and returned in unprojected
            coordinates. For example, if ``projection`` is from geographic to
            Mercator, the distances would be "angles" (incorrectly calculated
            as if they were Cartesian). After 1.4.0, *point1* and *point2* are
            projected prior to generating coordinates for the profile,
            guaranteeing that distances are properly handled in a Cartesian
            system. **With this change, the profile points are now evenly
            spaced in projected coordinates and the distances are returned in
            projected coordinates as well.**

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
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object ``projection(easting,
            northing, inverse=False) -> (proj_easting, proj_northing)`` that
            takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. Should also take
            an optional keyword argument ``inverse`` (default to False) that if
            True will calculate the inverse transform instead. This function
            will be used to project the profile end points before generating
            coordinates and passing them into ``predict``. It will also be used
            to undo the projection of the coordinates before returning the
            results.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values along the profile.

        """
        dims = self._get_dims(dims)
        # Project the input points to generate the profile in Cartesian
        # coordinates (the distance calculation doesn't make sense in
        # geographic coordinates since we don't do actual distances on a
        # sphere).
        if projection is not None:
            point1 = projection(*point1)
            point2 = projection(*point2)
        coordinates, distances = profile_coordinates(point1, point2, size, **kwargs)
        data = check_data(self.predict(coordinates))
        # Project the coordinates back to have geographic coordinates for the
        # profile but Cartesian distances.
        if projection is not None:
            coordinates = projection(*coordinates, inverse=True)
        data_names = get_data_names(data, data_names)
        columns = [
            (dims[0], coordinates[1]),
            (dims[1], coordinates[0]),
            ("distance", distances),
        ]
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def _get_dims(self, dims):
        """
        Get default dimension names.
        """
        if dims is not None:
            return dims
        return self.dims


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
