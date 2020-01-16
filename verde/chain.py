"""
Class for chaining gridders.
"""
from sklearn.utils.validation import check_is_fitted

from .base import BaseGridder
from .base.utils import check_data
from .coordinates import get_region


class Chain(BaseGridder):
    """
    Chain filtering operations to fit on each subsequent output.

    The :meth:`~verde.base.BaseGridder.filter` method of each element of the
    set is called with the outputs of the previous one. For gridders and trend
    estimators this means that each element fits the residuals (input data
    minus predicted data) of the previous one.

    When predicting data, the predictions of each step in the chain are added
    together. Steps that don't implement a
    :meth:`~verde.base.BaseGridder.predict` method are ignored.

    This provides a convenient way to chaining operations like trend estimation
    to a given gridder.

    Parameters
    ----------
    steps : list
        A list of ``('name', step)`` pairs where ``step`` is any verde
        class that implements a ``filter(coordinates, data, weights)`` method
        (including ``Chain`` itself).

    Attributes
    ----------
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the chain.
        Used as the default region for the :meth:`~verde.Chain.grid` and
        :meth:`~verde.Chain.scatter` methods.
    named_steps : dict
        A dictionary version of *steps* where the ``'name'``  strings are keys
        and the estimator/gridder/processor objects are the values.

    See also
    --------
    verde.Vector : Fit an estimator to each component of vector data

    """

    def __init__(self, steps):
        super().__init__()
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
        data : array or tuple of arrays
            the data values of each data point. if the data has more than one
            component, *data* must be a tuple of arrays (one for each
            component).
        weights : none or array or tuple of arrays
            if not none, then the weights assigned to each data point. if more
            than one data component is provided, you must provide a weights
            array for each data component (if not none).

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        self.region_ = get_region(coordinates[:2])
        args = coordinates, data, weights
        for _, step in self.steps:
            args = step.filter(*args)
        return self

    def predict(self, coordinates):
        """
        Evaluates the data predicted by the chain on the given set of points.

        Predictions from each step in the chain are added together. Any step
        that doesn't implement a ``predict`` method is ignored.

        Requires a fitted gridder (see :meth:`~verde.Chain.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...).

        Returns
        -------
        data : array
            The data values predicted on the given points.

        """
        check_is_fitted(self, ["region_"])
        result = None
        for _, step in self.steps:
            if hasattr(step, "predict"):
                predicted = check_data(step.predict(coordinates))
                if result is None:
                    result = [0 for i in range(len(predicted))]
                for i, pred in enumerate(predicted):
                    result[i] += pred
        if len(result) == 1:
            return result[0]
        return tuple(result)
