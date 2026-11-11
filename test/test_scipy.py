# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the scipy based interpolator.
"""
import warnings

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from verde.coordinates import grid_coordinates
from verde.scipygridder import Cubic, Linear
from verde.synthetic import CheckerBoard


@pytest.mark.parametrize(
    "gridder",
    [
        Linear(),
        Linear(rescale=False),
        Cubic(),
        Cubic(rescale=False),
    ],
    ids=[
        "linear",
        "linear_norescale",
        "cubic",
        "cubic_norescale",
    ],
)
def test_scipy_gridder_same_points(gridder):
    "See if the gridder recovers known points."
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1000, random_state=0)
    coords = (data.easting, data.northing)
    # The interpolation should be perfect on top of the data points
    gridder.fit(coords, data.scalars)
    predicted = gridder.predict(coords)
    npt.assert_allclose(predicted, data.scalars)
    npt.assert_allclose(gridder.score(coords, data.scalars), 1)


@pytest.mark.parametrize(
    "gridder",
    [
        Linear(),
        Linear(rescale=False),
        Cubic(),
        Cubic(rescale=False),
    ],
    ids=[
        "linear",
        "linear_norescale",
        "cubic",
        "cubic_norescale",
    ],
)
def test_scipy_gridder(gridder):
    "See if the gridder recovers known points."
    synth = CheckerBoard(region=(1000, 5000, -8000, -6000))
    data = synth.scatter(size=20000, random_state=0)
    coords = (data.easting, data.northing)
    pt_coords = (3000, -7000)
    true_data = synth.predict(pt_coords)
    # nearest will never be too close to the truth
    gridder.fit(coords, data.scalars)
    npt.assert_almost_equal(gridder.predict(pt_coords), true_data, decimal=1)


@pytest.mark.parametrize(
    "gridder",
    [
        Cubic(),
        Linear(),
    ],
    ids=["cubic", "linear"],
)
def test_scipy_gridder_region_xarray(gridder):
    "See if the region gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    synth = CheckerBoard(region=region)
    grid = synth.grid(shape=(101, 101))
    coords = grid_coordinates(region, grid.scalars.shape)
    gridder.fit(coords, grid.scalars)
    npt.assert_allclose(gridder.region_, region)


@pytest.mark.parametrize(
    "gridder",
    [
        Cubic(),
        Linear(),
    ],
    ids=["cubic", "linear"],
)
def test_scipy_gridder_region_pandas(gridder):
    "See if the region is gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    synth = CheckerBoard(region=region)
    grid = synth.grid(shape=(101, 101))
    coords = grid_coordinates(region, grid.scalars.shape)
    data = pd.DataFrame(
        {
            "easting": coords[0].ravel(),
            "northing": coords[1].ravel(),
            "scalars": grid.scalars.values.ravel(),
        }
    )
    gridder.fit((data.easting, data.northing), data.scalars)
    npt.assert_allclose(gridder.region_, region)


@pytest.mark.parametrize(
    "gridder",
    [
        Cubic(),
        Linear(),
    ],
    ids=["cubic", "linear"],
)
def test_scipy_gridder_warns(gridder):
    "Check that a warning is issued when using weights."
    data = CheckerBoard().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    msg = "does not support weights and they will be ignored."
    with warnings.catch_warnings(record=True) as warn:
        gridder.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).endswith(msg)
