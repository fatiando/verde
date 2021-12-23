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
                DummyEstimator(pred.ravel()),
                coordinates,
                data[i].ravel(),
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


def check_data(data):
    """
    Check the *data* argument and make sure it's a tuple.
    If the data is a single array, return it as a tuple with a single element.

    This is the default format accepted and used by all gridders and processing
    functions.

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


def check_data_names(data, data_names):
    """
    Check *data_names* against *data*.

    Also, convert ``data_names`` to a tuple if it's a single string.

    Examples
    --------

    >>> import numpy as np
    >>> east, north, scalar = [np.array(10)]*3
    >>> check_data_names((scalar,), "dummy")
    ('dummy',)
    >>> check_data_names((scalar,), ("dummy",))
    ('dummy',)
    >>> check_data_names((scalar,), ["dummy"])
    ['dummy']
    >>> check_data_names((east, north), ("component_x", "component_y"))
    ('component_x', 'component_y')
    """
    # Convert single string to tuple
    if isinstance(data_names, str):
        data_names = (data_names,)
    # Raise error if data_names is None
    if data_names is None:
        raise ValueError("Invalid data_names equal to None.")
    # Raise error if data and data_names don't have the same number of elements
    if len(data) != len(data_names):
        raise ValueError(
            "Data has {} components but only {} names provided: {}".format(
                len(data), len(data_names), str(data_names)
            )
        )
    return data_names


def check_coordinates(coordinates):
    """
    Check that the given coordinate arrays are what we expect them to be.
    Should be a tuple with arrays of the same shape.
    """
    shapes = [coord.shape for coord in coordinates]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(
            "Coordinate arrays must have the same shape. Coordinate shapes: {}".format(
                shapes
            )
        )
    return coordinates


def check_extra_coords_names(coordinates, extra_coords_names):
    """
    Check extra_coords_names against coordinates.

    Also, convert ``extra_coords_names`` to a tuple if it's a single string.
    Assume that there are extra coordinates on the ``coordinates`` tuple.

    Examples
    --------

    >>> import numpy as np
    >>> coordinates = [np.array(10)]*3
    >>> check_extra_coords_names(coordinates, "upward")
    ('upward',)
    >>> check_extra_coords_names(coordinates, ("upward",))
    ('upward',)
    >>> coordinates = [np.array(10)]*4
    >>> check_extra_coords_names(coordinates, ("upward", "time"))
    ('upward', 'time')
    """
    # Convert single string to a tuple
    if isinstance(extra_coords_names, str):
        extra_coords_names = (extra_coords_names,)
    # Check if it's not None
    if extra_coords_names is None:
        raise ValueError(
            "Invalid extra_coords_names equal to None. "
            + "When passing one or more extra coordinate, "
            + "extra_coords_names cannot be None."
        )
    # Check if there are the same number of extra_coords than extra_coords_name
    if len(coordinates[2:]) != len(extra_coords_names):
        raise ValueError(
            "Invalid extra_coords_names '{}'. ".format(extra_coords_names)
            + "Number of extra coordinates names must match the number of "
            + "additional coordinates ('{}').".format(len(coordinates[2:]))
        )
    return extra_coords_names


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
    unpack : bool
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
    coordinates = check_coordinates(coordinates)
    if any(i.shape != coordinates[0].shape for i in data):
        raise ValueError(
            "Data arrays must have the same shape {} as coordinates. Data shapes: {}.".format(
                coordinates[0].shape, [i.shape for i in data]
            )
        )
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
    return tuple(np.atleast_1d(i).ravel() for i in arrays[:n])
