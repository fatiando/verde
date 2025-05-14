# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utility functions for building gridders and checking arguments.
"""
import numpy as np
from sklearn.metrics import check_scoring


def score_estimator(scoring, estimator, coordinates, data, weights=None):
    """
    Score the given gridder against the given data using the given metric.

    If the data and predictions have more than 1 component, the scores of each
    component will be averaged.

    Parameters
    ----------
    scoring : str or callable
        A scoring specification known to scikit-learn. See
        :func:`sklearn.metrics.check_scoring`.
    estimator : a Verde gridder
        The gridder to score. Usually derived from
        :class:`verde.base.BaseGridder`.
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
        The score.

    """
    coordinates, data, weights = check_fit_input(
        coordinates, data, weights, unpack=False
    )
    predicted = check_data(estimator.predict(coordinates))
    scorer = check_scoring(DummyEstimator, scoring=scoring)
    result = np.mean(
        [
            scorer(
                DummyEstimator(np.ravel(pred)),
                coordinates,
                np.ravel(data[i]),
                sample_weight=weights[i],
            )
            for i, pred in enumerate(predicted)
        ]
    )
    return result


class DummyEstimator:
    """
    Dummy estimator that does nothing but pass along the predicted data.
    Used to fool the scikit-learn scorer functions to fit our API
    (multi-component estimators return a tuple on .predict).

    >>> est = DummyEstimator([1, 2, 3])
    >>> print(est.fit().predict())
    [1, 2, 3]

    """

    def __init__(self, predicted):
        self._predicted = predicted

    def predict(self, *args, **kwargs):  # noqa: U100
        "Return the stored predicted values"
        return self._predicted

    def fit(self, *args, **kwards):  # noqa: U100
        "Does nothing. Just here to satisfy the API."
        return self


def n_1d_arrays(arrays, n):
    """
    Get the first n elements from a tuple/list, convert to arrays, and ravel.

    Use this function to make sure that coordinate and data arrays are ready
    for building Jacobian matrices and least-squares fitting.

    Parameters
    ----------
    arrays : tuple of arrays
        The arrays. Can be lists or anything that can be converted to a numpy
        array (including numpy arrays).
    n : int
        How many arrays to return.

    Returns
    -------
    1darrays : tuple of arrays
        The converted 1D numpy arrays.

    Examples
    --------

    >>> import numpy as np
    >>> arrays = [np.arange(4).reshape(2, 2)]*3
    >>> n_1d_arrays(arrays, n=2)
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """
    return tuple(np.ravel(np.atleast_1d(i)) for i in arrays[:n])
