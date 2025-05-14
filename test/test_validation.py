# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the input validation functions
"""

import pytest

from verde._validation import check_fit_input, check_names, check_tuple_of_arrays


@pytest.mark.parametrize(
    ["arrays", "name"],
    [
        (([1, 2], [1, 2, 3]), "data"),
        ((1, 2, [1, 2]), "data"),
        (([1, 2], [1, 2, 3]), "DatA"),
        (([[1, 2], [3, 4]], [1, 2, 3]), "DATA"),
    ],
)
def test_check_tuple_of_arrays_raises(arrays, name):
    "Make sure an exception is raised when the inputs are invalid."
    with pytest.raises(ValueError, match="Invalid data arrays"):
        check_tuple_of_arrays(arrays, name)


@pytest.mark.parametrize(
    ["arrays", "names", "name"],
    [
        (([1, 2], [1, 2]), "one", "data"),
        (([1, 2], [1, 2]), ("one",), "data"),
        (([1, 2], [1, 2]), ["one"], "data"),
        (([1, 2], [1, 2]), ["one", "two", "three"], "DatA"),
    ],
)
def test_names_raises(arrays, names, name):
    "Make sure an exception is raised when the inputs are invalid."
    with pytest.raises(ValueError, match="Data has"):
        check_names(arrays, names, name)


@pytest.mark.parametrize(
    ["coordinates", "data", "weights"],
    [
        (([1, 2], [1, 2]), [1, 2], None),
        (([1, 2], [1, 2]), [1, 2], [1, 2]),
        (([1, 2], [1, 2]), ([1, 2],), None),
        (([1, 2], [1, 2]), ([1, 2],), ([1, 2],)),
        (([1, 2], [1, 2]), ([1, 2], [1, 2]), None),
        (([1, 2], [1, 2]), ([1, 2], [1, 2]), ([1, 2], [1, 2])),
        (([[1, 2], [3, 4]], [[1, 2], [3, 4]]), [[1, 2], [3, 4]], None),
        (([[1, 2], [3, 4]], [[1, 2], [3, 4]]), [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
    ],
)
def test_check_fit_input_passes(coordinates, data, weights):
    "Make sure no exceptions are raised for standard cases"
    coordinates_val, data_val, weights_val = check_fit_input(
        coordinates,
        data,
        weights,
    )
    assert len(coordinates_val) == len(coordinates)
    assert all(len(i.shape) == 1 for i in coordinates_val)
    if isinstance(data, tuple):
        assert len(data_val) == len(data)
        assert len(weights_val) == len(data)
    else:
        assert len(data_val) == 1
        assert len(weights_val) == 1
    assert all(len(i.shape) == 1 for i in data_val)
    if weights is not None:
        assert all(len(i.shape) == 1 for i in weights_val)


@pytest.mark.parametrize(
    ["coordinates", "data"],
    [
        (([1, 2], [1, 2]), [1, 2, 3]),
        (([1, 2, 3], [1, 2, 3]), [1]),
        (([1, 2, 3], [1, 2, 3]), [[1, 2], [1, 2]]),
        (([[1, 2], [3, 4]], [[1, 2], [3, 4]]), [[1, 2]]),
        (([[1, 2], [3, 4]], [[1, 2], [3, 4]]), [1, 2]),
    ],
)
def test_check_fit_input_raises_data_coords_shape(coordinates, data):
    "Make sure exceptions are raised for invalid cases"
    with pytest.raises(ValueError, match="Data arrays must have the same shape as"):
        check_fit_input(coordinates, data, None)


@pytest.mark.parametrize(
    ["data", "weights"],
    [
        ([1, 2], ([1, 2], [1, 2])),
        (([1, 2], [1, 2]), [1, 2]),
        (([1, 2], [1, 2]), ([1, 2],)),
    ],
)
def test_check_fit_input_raises_len_data_weights(data, weights):
    "Make sure exceptions are raised for invalid cases"
    with pytest.raises(ValueError, match="Number of data arrays "):
        check_fit_input(([1, 2], [1, 2]), data, weights)


@pytest.mark.parametrize(
    ["coordinates", "data", "weights"],
    [
        (([1, 2], [1, 2]), [1, 2], [1, 2, 3]),
        (([1, 2, 3], [1, 2, 3]), [1, 2, 3], [1, 2]),
    ],
)
def test_check_fit_input_raises_data_weights_shape(coordinates, data, weights):
    "Make sure exceptions are raised for invalid cases"
    with pytest.raises(ValueError, match="Weights arrays must have the same"):
        check_fit_input(coordinates, data, weights)
