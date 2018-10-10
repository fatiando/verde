"""
Test the scipy based interpolator.
"""
import warnings

import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest

from ..scipygridder import ScipyGridder
from ..coordinates import grid_coordinates
from ..datasets.synthetic import CheckerBoard


def test_scipy_gridder_same_points():
    "See if the gridder recovers known points."
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1000, random_state=0)
    coords = (data.easting, data.northing)
    # The interpolation should be perfect on top of the data points
    for method in ["nearest", "linear", "cubic"]:
        grd = ScipyGridder(method=method)
        grd.fit(coords, data.scalars)
        predicted = grd.predict(coords)
        npt.assert_allclose(predicted, data.scalars)
        npt.assert_allclose(grd.score(coords, data.scalars), 1)


def test_scipy_gridder():
    "See if the gridder recovers known points."
    synth = CheckerBoard(region=(1000, 5000, -8000, -6000))
    data = synth.scatter(size=20000, random_state=0)
    coords = (data.easting, data.northing)
    pt_coords = (3000, -7000)
    true_data = synth.predict(pt_coords)
    # nearest will never be too close to the truth
    grd = ScipyGridder("cubic").fit(coords, data.scalars)
    npt.assert_almost_equal(grd.predict(pt_coords), true_data, decimal=2)
    grd = ScipyGridder("linear").fit(coords, data.scalars)
    npt.assert_almost_equal(grd.predict(pt_coords), true_data, decimal=1)


def test_scipy_gridder_region():
    "See if the region is gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    synth = CheckerBoard(region=region)
    # Test using xarray objects
    grid = synth.grid(shape=(101, 101))
    coords = grid_coordinates(region, grid.scalars.shape)
    grd = ScipyGridder().fit(coords, grid.scalars)
    npt.assert_allclose(grd.region_, region)
    # Test using pandas objects
    data = pd.DataFrame(
        {
            "easting": coords[0].ravel(),
            "northing": coords[1].ravel(),
            "scalars": grid.scalars.values.ravel(),
        }
    )
    grd = ScipyGridder().fit((data.easting, data.northing), data.scalars)
    npt.assert_allclose(grd.region_, region)


def test_scipy_gridder_extra_args():
    "Passing in extra arguments to scipy"
    data = CheckerBoard().scatter(random_state=100)
    coords = (data.easting, data.northing)
    grd = ScipyGridder(method="linear", extra_args=dict(rescale=True))
    grd.fit(coords, data.scalars)
    predicted = grd.predict(coords)
    npt.assert_allclose(predicted, data.scalars)


def test_scipy_gridder_fails():
    "fit should fail for invalid method name"
    data = CheckerBoard().scatter(random_state=0)
    grd = ScipyGridder(method="some invalid method name")
    with pytest.raises(ValueError):
        grd.fit((data.easting, data.northing), data.scalars)


def test_scipy_gridder_warns():
    "Check that a warning is issued when using weights."
    data = CheckerBoard().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    grd = ScipyGridder()
    msg = "ScipyGridder does not support weights and they will be ignored."
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message) == msg
