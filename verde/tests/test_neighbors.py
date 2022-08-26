# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the nearest neighbors interpolator.
"""
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from ..coordinates import grid_coordinates
from ..neighbors import KNeighbors
from ..synthetic import CheckerBoard


def test_neighbors_same_points():
    "See if the gridder recovers known points."
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1000, random_state=0)
    coords = (data.easting, data.northing)
    # The interpolation should be perfect on top of the data points
    gridder = KNeighbors()
    gridder.fit(coords, data.scalars)
    predicted = gridder.predict(coords)
    npt.assert_allclose(predicted, data.scalars)
    npt.assert_allclose(gridder.score(coords, data.scalars), 1)


@pytest.mark.parametrize(
    "gridder",
    [
        KNeighbors(),
        KNeighbors(k=2),
        KNeighbors(k=3),
    ],
    ids=[
        "k=default",
        "k=2",
        "k=3",
    ],
)
def test_neighbors(gridder):
    "See if the gridder recovers known points."
    region = (1000, 5000, -8000, -6000)
    synth = CheckerBoard(region=region)
    data_coords = grid_coordinates(region, shape=(100, 100))
    data = synth.predict(data_coords)
    coords = grid_coordinates(region, shape=(95, 95))
    true_data = synth.predict(coords)
    # nearest will never be too close to the truth
    gridder.fit(data_coords, data)
    npt.assert_almost_equal(gridder.predict(coords), true_data, decimal=1)


def test_neighbors_weights_warning():
    "Check that a warning is issued when using weights."
    data = CheckerBoard().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    grd = KNeighbors()
    msg = "KNeighbors does not support weights and they will be ignored."
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message) == msg
