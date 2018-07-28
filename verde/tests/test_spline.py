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
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1500, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be perfect on top of the data points
    spline = Spline().fit(coords, data.scalars)
    npt.assert_allclose(spline.predict(coords), data.scalars, rtol=1e-5)
    npt.assert_allclose(spline.score(coords, data.scalars), 1)
    # There should be 1 force per data point
    assert data.scalars.size == spline.force_.size
    npt.assert_allclose(spline.force_coords_, coords)
    shape = (5, 5)
    region = (270, 320, -770, -720)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(
        spline.grid(region, shape=shape).scalars,
        synth.grid(region, shape=shape).scalars,
        rtol=5e-2,
    )


def test_spline_twice():
    "Check that the forces are updated when fitting twice"
    grid = CheckerBoard(region=(1000, 5000, -8000, -6000)).grid(shape=(10, 10))
    coords = np.meshgrid(grid.easting, grid.northing)
    spline = Spline()
    spline.fit(coords, grid.scalars.values)
    npt.assert_allclose(spline.force_coords_, coords)
    grid2 = CheckerBoard(region=(-15, -5, 10, 20)).grid(shape=(10, 10))
    coords2 = np.meshgrid(grid2.easting, grid2.northing)
    spline.fit(coords2, grid2.scalars.values)
    npt.assert_allclose(spline.force_coords_, coords2)
    assert not np.allclose(spline.force_coords_, coords)


def test_spline_region():
    "See if the region is gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    grid = CheckerBoard(region=region).grid(shape=(10, 10))
    coords = np.meshgrid(grid.easting, grid.northing)
    grd = Spline().fit(coords, grid.scalars.values)
    npt.assert_allclose(grd.region_, region)


def test_spline_damping():
    "Test the spline solution with a bit of damping"
    region = (100, 500, -800, -700)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1200, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be close on top of the data points
    spline = Spline(damping=1e-15).fit(coords, data.scalars)
    npt.assert_allclose(spline.predict(coords), data.scalars, rtol=1e-5)
    shape = (5, 5)
    region = (270, 320, -770, -720)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(
        spline.grid(region, shape=shape).scalars,
        synth.grid(region, shape=shape).scalars,
        rtol=5e-2,
    )


def test_spline_grid():
    "Test the spline solution with forces on a grid"
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=1500, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be close on top of the data points
    spline = Spline(shape=(35, 35)).fit(coords, data.scalars)
    npt.assert_allclose(spline.score(coords, data.scalars), 1, rtol=1e-2)
    assert spline.force_.size == 35 ** 2
    npt.assert_allclose(spline.force_coords_[0].min(), data.easting.min())
    npt.assert_allclose(spline.force_coords_[1].min(), data.northing.min())
    shape = (3, 3)
    region = (2700, 3200, -7700, -7200)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(
        spline.grid(region, shape=shape).scalars,
        synth.grid(region, shape=shape).scalars,
        rtol=5e-2,
    )


def test_spline_warns_weights():
    "Check that a warning is issued when using weights and the exact solution."
    data = CheckerBoard().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    grd = Spline()
    msg = "Weights are ignored for the exact solution"
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split(".")[0] == msg


def test_spline_warns_underdetermined():
    "Check that a warning is issued when the problem is underdetermined"
    data = CheckerBoard().scatter(size=50, random_state=100)
    grd = Spline(shape=(10, 10))
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).startswith("Under-determined problem")
