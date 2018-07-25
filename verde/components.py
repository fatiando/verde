"""
Components meta-estimator to use different estimators on multi-component data.
"""
from .base import BaseGridder, check_fit_input


class Components(BaseGridder):
    """
    Fit an estimator to each component of multi-component vector data.

    Provides a convenient way of fitting and gridding vector data using scalar gridders
    and estimators.

    Each data component provided to :meth:`~verde.Components.fit` is fitted to a
    separated estimator. Methods like :meth:`~verde.Components.grid` and
    :meth:`~verde.Components.predict` will operate on the multiple components
    simultaneously.

    Parameters
    ----------
    components : tuple or list
        A tuple or list of the estimator/gridder instances used for each components. The
        estimators will be applied for each data component in the same order that they
        are given here.

    Attributes
    ----------
    components : tuple
        Tuple of the fitted estimators on each component of the data.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the interpolator. Used
        as the default region for the :meth:`~verde.Components.grid` and
        :meth:`~verde.Components.scatter` methods.

    See also
    --------
    verde.Chain : Chain filtering operations to fit on each subsequent output.

    """

    def __init__(self, degree):
        self.degree = degree

    def fit(self, coordinates, data, weights=None):
        """
        Fit the trend to the given multi-component data.

        The data region is captured and used as default for the
        :meth:`~verde.VectorTrend.grid` and :meth:`~verde.VectorTrend.scatter`
        methods.

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
        self.component_ = [
            Trend(degree=self.degree).fit(coordinates, data_comp, weight_comp)
            for data_comp, weight_comp in zip(data, weights)
        ]
        self.residual_ = tuple(
            data_comp - comp.predict(coordinates).reshape(data_comp.shape)
            for data_comp, comp in zip(data, self.component_)
        )
        return self

    def predict(self, coordinates):
        """
        Evaluate the polynomial trend of each component on a of points.

        Requires a fitted estimator (see :meth:`~verde.VectorTrend.fit`).

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
            The trend values for each vector component evaluated on the given
            points. The order of components will be the same as was provided to
            :meth:`~verde.VectorTrend.fit`.

        """
        check_is_fitted(self, ["component_"])
        return tuple(comp.predict(coordinates) for comp in self.component_)

