"""
Utility functions for building gridders and checking arguments.
"""
import numpy as np


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
