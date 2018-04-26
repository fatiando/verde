"""
Test the spline interpolation.
"""
import numpy as np
import numpy.testing as npt

from ..spline import spline1d_green


def test_spline1d_green():
    "Test the spline 1D Greens function against known values"
    npt.assert_allclose(spline1d_green(), np.ones(10))
