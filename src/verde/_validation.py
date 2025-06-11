# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to check and conform inputs into our functions and classes.
"""

import numpy as np


def check_tuple_of_arrays(arrays, name):
    """
    Check a given tuple of arrays (data, coordinates, or weights).

    Make sure all arrays have the same shape. Convert any lists/tuples to
    arrays. If a single array is given, return it as a tuple with a single
    element. If arrays is None, then this function will return a tuple with the
    element None.

    Parameters
    ----------
    arrays : tuple or array or list or None
        If a tuple, will just return the given values. If an array, will place
        the array as the single element of a tuple and return. If a list, will
        first convert the list to an array, then place it in a tuple and
        return. If the elements of the tuple are lists, they will be converted
        to arrays.
    name : str
        The name of the type of arrays being checked. Used in the exception
        that is raised. For example, if checking data arrays use
        ``name="data"``.

    Returns
    -------
    arrays : tuple of arrays
        A tuple with the arrays that all have the same shape. If the input
        arrays is None, then return a tuple with None.

    Raises
    ------
    ValueError
        If the arrays don't all have the same shape.

    Examples
    --------
    >>> check_tuple_of_arrays(1, name="data")
    (array([1]),)
    >>> check_tuple_of_arrays(None, name="data")
    (None,)
    >>> check_tuple_of_arrays((1, 2), name="data")
    (array([1]), array([2]))
    >>> check_tuple_of_arrays([1, 2, 3], name="data")
    (array([1, 2, 3]),)
    >>> check_tuple_of_arrays(([1, 2], [3, 4]), name="data")
    (array([1, 2]), array([3, 4]))
    """
    if arrays is None:
        return (None,)
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
    arrays = tuple(np.atleast_1d(a) for a in arrays)
    shapes = [a.shape for a in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        message = (
            f"Invalid {name.lower()} arrays. "
            "All arrays must have the same shape. "
            f"Given shapes: {shapes}"
        )
        raise ValueError(message)
    return arrays


def check_names(data, names, name):
    """
    Make sure the names are consistent with elements in the array.

    Also, convert ``data_names`` to a tuple if it's a single string.

    Parameters
    ----------
    data : tuple
        Tuple of arrays.
    names : str or tuple or list
        Either a single name or a list/tuple of names for each array in data.
    name : str
        The name of the type of data given. Used in the error message. For
        example, if checking data names, use ``name="data"`` or if using
        coordinates, ``name="coordinates"``.

    Returns
    -------
    names : tuple
        The names given but always as a tuple.

    Raises
    ------
    ValueError
        If the number of elements in data and names are not the same or if any
        of the names is None.

    Examples
    --------
    >>> data = [1, 2, 3]
    >>> check_names((data,), "dummy", "data")
    ('dummy',)
    >>> check_names((data,), ("dummy",), "data")
    ('dummy',)
    >>> check_names((data,), ["dummy"], "data")
    ('dummy',)
    >>> check_names((data, data), ("component_x", "component_y"), "data")
    ('component_x', 'component_y')
    """
    # Convert single string to tuple
    if isinstance(names, str):
        names = (names,)
    if names is None or any(n is None for n in names):
        message = (
            f"Using None as {name.lower()} name is invalid. Names should be strings."
        )
        raise ValueError(message)
    # Raise error if data and names don't have the same number of elements
    if len(data) != len(names):
        message = (
            f"{name.title()} has {len(data)} components but only {len(names)} names "
            f"were provided: {str(names)}"
        )
        raise ValueError(message)
    return tuple(names)


def check_fit_input(coordinates, data, weights):
    """
    Validate the inputs to the fit method of interpolators.

    Checks that the coordinates, data, and weights (if given) all have the same
    shape. Ravel all inputs to make sure they are 1D arrays. Make sure data and
    weights are always tuples, even if only given one array. Transform lists to
    numpy arrays if necessary.

    Parameters
    ----------
    coordinates : tuple = (easting, northing, ...)
        Tuple of arrays with the coordinates of each point. Arrays can be
        Python lists or any numpy-compatible array type. Arrays can be of any
        shape but must all have the same shape.
    data : array or tuple of arrays
        The data values of each data point. Data that have multiple components
        (for example, measurements of vector fields) should be passed as
        a tuple of arrays. Arrays can be Python lists or any numpy-compatible
        array type. Arrays can be of any shape but must all have the same shape
        and must have the same shape as the coordinates.
    weights : None or array or tuple of arrays
        If not None, then an array with the weights assigned to each data
        point. Arrays can be Python lists or any numpy-compatible array type.
        Arrays can be of any shape but must all have the same shape and must
        have the same shape as the data arrays. If the data has multiple
        components, weights should be a tuple with the same number of
        components as the data. Typically, weights are 1 over the data
        uncertainty squared but can take any value. It's recommended to
        normalize weights to the range [0, 1].

    Returns
    -------
    coordinates, data, weights : tuple
        The validated inputs in the same order.
    """
    null_weights = weights is None or all(w is None for w in weights)
    data = check_tuple_of_arrays(data, name="data")
    weights = check_tuple_of_arrays(weights, name="weights")
    coordinates = check_tuple_of_arrays(coordinates, name="coordinates")
    if data[0].shape != coordinates[0].shape:
        message = (
            "Data arrays must have the same shape as coordinate arrays. "
            f"Given data with shape {data[0].shape} and coordinates with shape "
            f"{coordinates[0].shape}."
        )
        raise ValueError(message)
    if null_weights:
        weights = tuple(None for _ in range(len(data)))
    else:
        if len(weights) != len(data):
            message = (
                f"Number of data arrays '{len(data)}' and weight arrays "
                f"'{len(weights)}' must be equal."
            )
            raise ValueError(message)
        if data[0].shape != weights[0].shape:
            message = (
                "Weights arrays must have the same shape as data arrays. "
                f"Given data with shape {data[0].shape} and weights with shape "
                f"{weights[0].shape}."
            )
            raise ValueError(message)
        weights = tuple(w.ravel() for w in weights)
    data = tuple(d.ravel() for d in data)
    coordinates = tuple(c.ravel() for c in coordinates)
    return coordinates, data, weights
