"""
Class for chaining gridders.
"""
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder, check_data
from .coordinates import get_region


class Chain(BaseGridder):
    """
    Chain gridders to fit on each others residuals.

    Given a set of gridders or trend estimators, :meth:`~verde.Chain.fit` will
    fit each estimator on the data residuals of the previous one. When
    predicting data, the predictions of each estimator will be added together.

    This provides a convenient way to chaining operations like trend estimation
    to a given gridder.

    Parameters
    ----------
    steps : list
        A list of ``('name', gridder)`` pairs where ``gridder`` is any verde
        class that implements the gridder interface (including ``Chain``
        itself).

    Attributes
    ----------
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the chain.
        Used as the default region for the :meth:`~verde.Chain.grid` and
        :meth:`~verde.Chain.scatter` methods.
    residual_ : array or tuple of arrays
        The data residual after all chained operations are applied to the data.
    named_steps : dict
        A dictionary version of *steps* where the ``'name'``  strings are keys
        and the ``gridder`` objects are the values.

    """

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        """
        A dictionary version of steps.
        """
        return dict(self.steps)

    def fit(self, coordinates, data, weights=None):
        """
        Fit the chained estimators to the given data.

        Each estimator in the chain is fitted to the residuals of the previous
        estimator. The coordinates are preserved. Only the data values are
        changed.

        The data region is captured and used as default for the
        :meth:`~verde.Chain.grid` and :meth:`~verde.Chain.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).
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
        self.region_ = get_region(coordinates[:2])
        residuals = data
        for _, step in self.steps:
            step.fit(coordinates, residuals, weights)
            residuals = step.residual_
        self.residual_ = residuals
        return self

    def predict(self, coordinates):
        """
        Interpolate data on the given set of points.

        Requires a fitted gridder (see :meth:`~verde.ScipyGridder.fit`).

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
            The data values interpolated on the given points.

        """
        check_is_fitted(self, ['region_', 'residual_'])
        result = None
        for _, step in self.steps:
            predicted = check_data(step.predict(coordinates))
            if result is None:
                result = [0 for i in range(len(predicted))]
            for i, pred in enumerate(predicted):
                result[i] += pred
        if len(result) == 1:
            result = result[0]
        return result
