# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Base classes for all gridders.
"""
import warnings
from abc import ABCMeta, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from ..coordinates import grid_coordinates, profile_coordinates, scatter_points
from ..utils import (
    check_meshgrid,
    get_ndim_horizontal_coords,
    make_xarray_grid,
    meshgrid_from_1d,
)
from .utils import check_data, check_data_names, score_estimator


class BaseBlockCrossValidator(BaseCrossValidator, metaclass=ABCMeta):
    """
    Base class for spatially blocked cross-validators.

    Parameters
    ----------
    spacing : float, tuple = (s_north, s_east), or None
        The block size in the South-North and West-East directions,
        respectively. A single value means that the spacing is equal in both
        directions. If None, then *shape* **must be provided**.
    shape : tuple = (n_north, n_east) or None
        The number of blocks in the South-North and West-East directions,
        respectively. If None, then *spacing* **must be provided**.
    n_splits : int
        Number of splitting iterations.

    """

    def __init__(
        self,
        spacing=None,
        shape=None,
        n_splits=10,
    ):
        if spacing is None and shape is None:
            raise ValueError("Either 'spacing' or 'shape' must be provided.")
        self.spacing = spacing
        self.shape = shape
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):  # noqa: N803
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        if X.shape[1] != 2:
            raise ValueError(
                "X must have exactly 2 columns ({} given).".format(X.shape[1])
            )
        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: U100,N803
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    @abstractmethod
    def _iter_test_indices(self, X=None, y=None, groups=None):  # noqa: U100,N803
        """
        Generates integer indices corresponding to test sets.

        MUST BE IMPLEMENTED BY DERIVED CLASSES.

        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            Columns should be the easting and northing coordinates of data
            points, respectively.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems. Always
            ignored.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Always ignored.

        Yields
        ------
        test : ndarray
            The testing set indices for that split.

        """


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
    ...     def fit(self, coordinates, data):
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
    >>> synthetic = vd.synthetic.CheckerBoard(region=(0, 5, -10, 8))
    >>> data = synthetic.scatter()
    >>> print('{:.4f}'.format(data.scalars.mean()))
    -32.2182
    >>> # Fit the gridder to our synthetic data
    >>> grd = MeanGridder().fit((data.easting, data.northing), data.scalars)
    >>> grd
    MeanGridder()
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

    # The default name for any extra coordinates given to methods below
    # through the `extra_coords` keyword argument. Coordinates are
    # included in the outputs (pandas.DataFrame or xarray.Dataset)
    # using this name as a basis.
    extra_coords_name = "extra_coord"

    # Define default values for data_names depending on the number of data
    # arrays returned by predict method.
    data_names_defaults = [
        ("scalars",),
        ("east_component", "north_component"),
        ("east_component", "north_component", "vertical_component"),
    ]

    def predict(self, coordinates):  # noqa: U100
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

    def fit(self, coordinates, data, weights=None):  # noqa: U100
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

    def filter(self, coordinates, data, weights=None):  # noqa: A003
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

        .. warning::

            The default scoring will change from R² to negative root mean
            squared error (RMSE) in Verde 2.0.0. This may change model
            selection results slightly. The negative version will be used to
            maintain the behaviour of larger scores being better, which is more
            compatible with current model selection code.

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
        warnings.warn(
            "The default scoring will change from R² to negative root mean "
            "squared error (RMSE) in Verde 2.0.0. "
            "This may change model selection results slightly.",
            FutureWarning,
        )
        return score_estimator("r2", self, coordinates, data, weights=weights)

    def grid(
        self,
        region=None,
        shape=None,
        spacing=None,
        dims=None,
        data_names=None,
        projection=None,
        coordinates=None,
        **kwargs,
    ):
        """
        Interpolate the data onto a regular grid.

        The grid can be specified by two methods:

        - Pass the actual *coordinates* of the grid points, as generated by
          :func:`verde.grid_coordinates` or from an existing
          :class:`xarray.Dataset` grid.
        - Let the method define a new grid by either passing the number of
          points in each dimension (the *shape*) or by the grid node *spacing*.
          If the interpolator collected the input data region, then it will be
          used if ``region=None``. Otherwise, you must specify the grid region.
          See :func:`verde.grid_coordinates` for details. Other arguments for
          :func:`verde.grid_coordinates` can be passed as extra keyword
          arguments (``kwargs``) to this method.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names will be provided if none are given.

        Parameters
        ----------
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
            Use only if ``coordinates`` is None.
        shape : tuple = (n_north, n_east) or None
            The number of points in the South-North and West-East directions,
            respectively.
            Use only if ``coordinates`` is None.
        spacing : tuple = (s_north, s_east) or None
            The grid spacing in the South-North and West-East directions,
            respectively.
            Use only if ``coordinates`` is None.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output grid. Default is determined from the
            ``dims`` attribute of the class. Must be defined in the following
            order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : str, list or None
            The name(s) of the data variables in the output grid. Defaults to
            ``'scalars'`` for scalar data,
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
        coordinates : tuple of arrays
            Tuple of arrays containing the coordinates of the grid in the
            following order: (easting, northing, vertical, ...).
            The easting and northing arrays could be 1d or 2d arrays, if
            they are 2d they must be part of a meshgrid.
            If coordinates are passed, ``region``, ``shape``, and ``spacing``
            are ignored.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        See also
        --------
        verde.grid_coordinates : Generate the coordinate values for the grid.

        """
        if coordinates is not None and (spacing is not None or shape is not None):
            raise ValueError(
                "Both coordinates and spacing or shape were provided. "
                + "Please pass only coordinates or either the spacing or the shape."
            )
        if coordinates is not None and region is not None:
            raise ValueError(
                "Both coordinates and region were provided. "
                + "Please pass region only if spacing or shape is specified."
            )
        # Raise deprecation warning for the region, shape and spacing arguments
        if spacing is not None or shape is not None or region is not None:
            warnings.warn(
                "The 'spacing', 'shape' and 'region' arguments will be removed "
                + "in Verde v2.0.0. "
                + "Please use the 'verde.grid_coordinates' function to define "
                + "grid coordinates and pass them as the 'coordinates' argument.",
                FutureWarning,
            )
        # Get grid coordinates from coordinates parameter
        if coordinates is not None:
            ndim = get_ndim_horizontal_coords(*coordinates[:2])
            if ndim == 1:
                # Build a meshgrid if easting and northing are 1d
                coordinates = meshgrid_from_1d(coordinates)
            else:
                check_meshgrid(coordinates)
        # Build the grid coordinates through vd.grid_coordinates
        else:
            region = get_instance_region(self, region)
            coordinates = grid_coordinates(
                region, shape=shape, spacing=spacing, **kwargs
            )
        # Predict on the grid points (project the coordinates if needed)
        if projection is None:
            data = check_data(self.predict(coordinates))
        else:
            data = check_data(
                self.predict(project_coordinates(coordinates, projection))
            )
        # Get names for dims, data and any extra coordinates
        dims = self._get_dims(dims)
        data_names = self._get_data_names(data, data_names)
        extra_coords_names = self._get_extra_coords_names(coordinates)
        # Create xarray.Dataset
        dataset = make_xarray_grid(
            coordinates,
            data,
            data_names,
            dims=dims,
            extra_coords_names=extra_coords_names,
        )
        # Add metadata as attrs
        metadata = "Generated by {}".format(repr(self))
        dataset.attrs["metadata"] = metadata
        for array in dataset:
            dataset[array].attrs["metadata"] = metadata
        return dataset

    def scatter(
        self,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
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
        data_names : str, list or None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``'scalars'`` for scalar data,
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
            data = check_data(
                self.predict(project_coordinates(coordinates, projection))
            )
        data_names = self._get_data_names(data, data_names)
        columns = [(dims[0], coordinates[1]), (dims[1], coordinates[0])]
        extra_coords_names = self._get_extra_coords_names(coordinates)
        columns.extend(zip(extra_coords_names, coordinates[2:]))
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
        **kwargs,
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
        data_names : str, list or None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``'scalars'`` for scalar data,
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
            point1 = project_coordinates(point1, projection)
            point2 = project_coordinates(point2, projection)
        coordinates, distances = profile_coordinates(point1, point2, size, **kwargs)
        data = check_data(self.predict(coordinates))
        # Project the coordinates back to have geographic coordinates for the
        # profile but Cartesian distances.
        if projection is not None:
            coordinates = project_coordinates(coordinates, projection, inverse=True)
        data_names = self._get_data_names(data, data_names)
        columns = [
            (dims[0], coordinates[1]),
            (dims[1], coordinates[0]),
            ("distance", distances),
        ]
        extra_coords_names = self._get_extra_coords_names(coordinates)
        columns.extend(zip(extra_coords_names, coordinates[2:]))
        columns.extend(zip(data_names, data))
        return pd.DataFrame(dict(columns), columns=[c[0] for c in columns])

    def _get_dims(self, dims):
        """
        Get default dimension names.
        """
        if dims is not None:
            return dims
        return self.dims

    def _get_extra_coords_names(self, coordinates):
        """
        Return names for extra coordinates

        Examples
        --------

        >>> coordinates = (-5, 4, 3, 5, 1)
        >>> grd = BaseGridder()
        >>> grd._get_extra_coords_names(coordinates)
        ['extra_coord', 'extra_coord_1', 'extra_coord_2']

        >>> coordinates = (-5, 4, 3)
        >>> grd = BaseGridder()
        >>> grd.extra_coords_name = "upward"
        >>> grd._get_extra_coords_names(coordinates)
        ['upward']

        """
        names = []
        for i in range(len(coordinates[2:])):
            name = self.extra_coords_name
            if i > 0:
                name += "_{}".format(i)
            names.append(name)
        return names

    def _get_data_names(self, data, data_names):
        """
        Get default names for data fields if none are given based on the data.

        Examples
        --------

        >>> import numpy as np
        >>> east, north, up = [np.arange(10)]*3
        >>> gridder = BaseGridder()
        >>> gridder._get_data_names((east,), data_names=None)
        ('scalars',)
        >>> gridder._get_data_names((east, north), data_names=None)
        ('east_component', 'north_component')
        >>> gridder._get_data_names((east, north, up), data_names=None)
        ('east_component', 'north_component', 'vertical_component')
        >>> gridder._get_data_names((east,), data_names="john")
        ('john',)
        >>> gridder._get_data_names((east,), data_names=("paul",))
        ('paul',)
        >>> gridder._get_data_names(
        ...     (up, north), data_names=('ringo', 'george')
        ... )
        ('ringo', 'george')
        >>> gridder._get_data_names((north,), data_names=["brian"])
        ['brian']

        """
        # Return the defaults data_names for the class
        if data_names is None:
            if len(data) > len(self.data_names_defaults):
                raise ValueError(
                    "Default data names only available for up to 3 components. "
                    + "Must provide custom names through the 'data_names' argument."
                )
            return self.data_names_defaults[len(data) - 1]
        # Return the passed data_names if valid
        data_names = check_data_names(data, data_names)
        return data_names


def project_coordinates(coordinates, projection, **kwargs):
    """
    Apply projection to given coordinates

    Allows to apply projections to any number of coordinates, assuming
    that the first ones are ``longitude`` and ``latitude``.

    Examples
    --------

    >>> # Define a custom projection function
    >>> def projection(lon, lat, inverse=False):
    ...     "Simple projection of geographic coordinates"
    ...     radius = 1000
    ...     if inverse:
    ...         return (lon / radius, lat / radius)
    ...     return (lon * radius, lat * radius)

    >>> # Apply the projection to a set of coordinates containing:
    >>> # longitude, latitude and height
    >>> coordinates = (1., -2., 3.)
    >>> project_coordinates(coordinates, projection)
    (1000.0, -2000.0, 3.0)

    >>> # Apply the inverse projection
    >>> coordinates = (-500.0, 1500.0, -19.0)
    >>> project_coordinates(coordinates, projection, inverse=True)
    (-0.5, 1.5, -19.0)

    """
    proj_coordinates = projection(*coordinates[:2], **kwargs)
    if len(coordinates) > 2:
        proj_coordinates += tuple(coordinates[2:])
    return proj_coordinates


def get_instance_region(instance, region):
    """
    Get the region attribute stored in instance if one is not provided.
    """
    if region is None:
        if not hasattr(instance, "region_"):
            raise ValueError("No default region found. Argument must be supplied.")
        region = instance.region_
    return region
