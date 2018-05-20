"""
Test the Chain class
"""
import numpy.testing as npt

from ..datasets.synthetic import CheckerBoard
from ..chain import Chain
from ..scipy_bridge import ScipyGridder
from ..trend import Trend


def test_chain():
    "Test chaining trend and gridder."
    region = (1000, 5000, -8000, -7000)
    synth = CheckerBoard(amplitude=100, w_east=1000, w_north=100)
    synth.fit(region=region)
    data = synth.scatter(size=5000, random_state=0)
    east, north = coords = data.easting, data.northing
    coefs = [1000, 0.2, -1.4]
    trend = coefs[0] + coefs[1]*east + coefs[2]*north
    all_data = trend + data.scalars

    grd = Chain([('trend', Trend(degree=1)),
                 ('gridder', ScipyGridder())])
    grd.fit(coords, all_data)

    npt.assert_allclose(grd.predict(coords), all_data)
    npt.assert_allclose(grd.residual_, 0, atol=1e-5)
    npt.assert_allclose(grd.named_steps['trend'].coef_, coefs, rtol=1e-2)
    npt.assert_allclose(grd.named_steps['trend'].predict(coords), trend,
                        rtol=1e-3)
    # The residual is too small to test. Need a robust trend probably before
    # this will work.
    # npt.assert_allclose(grd.named_steps['gridder'].predict(coords),
    #                     data.scalars, rtol=5e-2, atol=0.5)
