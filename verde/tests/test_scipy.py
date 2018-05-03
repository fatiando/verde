"""
Test the scipy based interpolator.
"""
import pytest
import numpy.testing as npt

from .. import ScipyGridder
from ..datasets import CheckerBoard


def test_scipy_gridder_same_points():
    "See if the gridder recovers known points."
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard().fit(region=region)
    data = synth.scatter(size=1000, random_state=0)
    # The interpolation should be perfect on top of the data points
    for method in ['nearest', 'linear', 'cubic']:
        grd = ScipyGridder(method=method)
        grd.fit(data.easting, data.northing, data.scalars)
        predicted = grd.predict(data.easting, data.northing)
        npt.assert_allclose(predicted, data.scalars)


def test_scipy_gridder():
    "See if the gridder recovers known points."
    synth = CheckerBoard().fit(region=(1000, 5000, -8000, -6000))
    data = synth.scatter(size=20000, random_state=0)
    east, north = 3000, -7000
    true_data = synth.predict(east, north)
    # nearest will never be too close to the truth
    grd = ScipyGridder('cubic').fit(data.easting, data.northing, data.scalars)
    npt.assert_almost_equal(grd.predict(east, north), true_data, decimal=2)
    grd = ScipyGridder('linear').fit(data.easting, data.northing, data.scalars)
    npt.assert_almost_equal(grd.predict(east, north), true_data, decimal=1)


def test_scipy_gridder_extra_args():
    "Passing in extra arguments to scipy"
    data = CheckerBoard().fit().scatter(random_state=100)
    grd = ScipyGridder(method='linear', extra_args=dict(rescale=True))
    grd.fit(data.easting, data.northing, data.scalars)
    predicted = grd.predict(data.easting, data.northing)
    npt.assert_allclose(predicted, data.scalars)


def test_scipy_gridder_fails():
    "fit should fail for invalid method name"
    data = CheckerBoard().fit().scatter(random_state=0)
    grd = ScipyGridder(method='some invalid method name')
    with pytest.raises(ValueError):
        grd.fit(data.easting, data.northing, data.scalars)
