"""
Test the biharmonic splines
"""
import warnings

import numpy as np
import numpy.testing as npt

from ..spline import Spline
from ..datasets.synthetic import CheckerBoard
from .utils import requires_numba


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
    npt.assert_allclose(spline.force_coords, coords)
    shape = (5, 5)
    region = (270, 320, -770, -720)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(
        spline.grid(region, shape=shape).scalars,
        synth.grid(region, shape=shape).scalars,
        rtol=5e-2,
    )


def test_spline_weights():
    "Use weights to ignore an outlier"
    data = CheckerBoard().scatter(size=2000, random_state=1)
    data_outlier = data.scalars.copy()
    outlier = 500
    outlier_value = 100e3
    data_outlier[outlier] += outlier_value
    weights = np.ones_like(data_outlier)
    weights[outlier] = 1e-10
    coords = (data.easting, data.northing)
    spline = Spline(damping=1e-8).fit(coords, data_outlier, weights=weights)
    npt.assert_allclose(spline.score(coords, data.scalars), 1, atol=0.01)
    predicted = spline.predict(coords)
    npt.assert_allclose(predicted, data.scalars, rtol=1e-2, atol=10)
    npt.assert_allclose(
        data_outlier[outlier] - predicted[outlier], outlier_value, rtol=1e-2
    )


def test_spline_region():
    "See if the region is gotten from the data is correct."
    region = (1000, 5000, -8000, -6000)
    grid = CheckerBoard(region=region).grid(shape=(10, 10))
    coords = np.meshgrid(grid.easting, grid.northing)
    grd = Spline().fit(coords, grid.scalars.values)
    npt.assert_allclose(grd.region_, region)


def test_spline_damping():
    "Test the spline solution with a bit of damping"
    region = (1000, 5000, -8000, -6000)
    synth = CheckerBoard(region=region)
    data = synth.scatter(size=3000, random_state=1)
    coords = (data.easting, data.northing)
    # The interpolation should be close on top of the data points
    spline = Spline(damping=1e-8, mindist=1000).fit(coords, data.scalars)
    npt.assert_allclose(spline.predict(coords), data.scalars, rtol=1e-2, atol=1)
    shape = (5, 5)
    region = (2000, 4000, -7500, -6500)
    # Tolerance needs to be kind of high to allow for error due to small
    # dataset
    npt.assert_allclose(
        spline.grid(region, shape=shape).scalars,
        synth.grid(region, shape=shape).scalars,
        rtol=1e-2,
        atol=10,
    )


def test_spline_warns_weights():
    "Check that a warning is issued when using weights and the exact solution."
    data = CheckerBoard().scatter(random_state=100)
    weights = np.ones_like(data.scalars)
    grd = Spline()
    msg = "Weights might have no effect if no regularization is used"
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars, weights=weights)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).split(".")[0] == msg


def test_spline_warns_underdetermined():
    "Check that a warning is issued when the problem is underdetermined"
    data = CheckerBoard().scatter(size=50, random_state=100)
    grd = Spline(force_coords=(np.arange(60), np.arange(60)))
    with warnings.catch_warnings(record=True) as warn:
        grd.fit((data.easting, data.northing), data.scalars)
        assert len(warn) == 1
        assert issubclass(warn[-1].category, UserWarning)
        assert str(warn[-1].message).startswith("Under-determined problem")


@requires_numba
def test_spline_jacobian_implementations():
    "Compare the numba and numpy implementations."
    data = CheckerBoard().scatter(size=1500, random_state=1)
    coords = (data.easting, data.northing)
    jac_numpy = Spline(engine="numpy").jacobian(coords, coords)
    jac_numba = Spline(engine="numba").jacobian(coords, coords)
    npt.assert_allclose(jac_numpy, jac_numba)
