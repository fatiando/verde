"""
Test the biharmonic splines
"""
import warnings

import numpy as np
import numpy.testing as npt

from ..spline import Spline
from ..datasets.synthetic import CheckerBoard


def test_spline():
    "See if the exact spline solution."
    region = (100, 500, -800, -700)
    synth = CheckerBoard().fit(region=region)
    data = synth.scatter(size=1500, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be perfect on top of the data points
    spline = Spline().fit(coords, data.scalars)
    npt.assert_allclose(spline.predict(coords), data.scalars, rtol=1e-5)
    npt.assert_allclose(spline.residual_, 0, atol=1e-5)
    # There should be 1 force per data point
    assert data.scalars.size == spline.force_.size
    npt.assert_allclose(spline.force_coords_, coords)
    shape = (5, 5)
    region = (270, 320, -770, -720)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(spline.grid(region, shape=shape).scalars,
                        synth.grid(region, shape=shape).scalars,
                        rtol=5e-2)


def test_spline_region():
    "See if the region is gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    grid = CheckerBoard().fit(region=region).grid(shape=(10, 10))
    coords = np.meshgrid(grid.easting, grid.northing)
    grd = Spline().fit(coords, grid.scalars.values)
    npt.assert_allclose(grd.region_, region)


def test_spline_damping():
    "Test the spline solution with a bit of damping"
    region = (100, 500, -800, -700)
    synth = CheckerBoard().fit(region=region)
    data = synth.scatter(size=1200, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be close on top of the data points
    spline = Spline(damping=1e-15).fit(coords, data.scalars)
    npt.assert_allclose(spline.residual_.mean(), 0, atol=1e-2)
    npt.assert_allclose(spline.residual_.std(), 0, atol=1e-2)
    shape = (5, 5)
    region = (270, 320, -770, -720)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(spline.grid(region, shape=shape).scalars,
                        synth.grid(region, shape=shape).scalars,
                        rtol=5e-2)


def test_spline_grid():
    "Test the spline solution with forces on a grid"
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard().fit(region=region)
    data = synth.scatter(size=1500, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be close on top of the data points
    spline = Spline(shape=(35, 35)).fit(coords, data.scalars)
    npt.assert_allclose(spline.residual_.mean(), 0, atol=1e-2)
    npt.assert_allclose(spline.residual_.std(), 0, atol=2)
    assert spline.force_.size == 35**2
    shape = (3, 3)
    region = (2700, 3200, -7700, -7200)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(spline.grid(region, shape=shape).scalars,
                        synth.grid(region, shape=shape).scalars,
                        rtol=5e-2)


def test_spline_warns():
    "Check that a warning is issued when using weights and the exact solution."
    data = CheckerBoard().fit().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    grd = Spline()
    msg = "Weights are ignored for the exact spline solution"
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split('.')[0] == msg
