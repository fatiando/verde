"""
"""
import numpy as np


def biharmonic_spline2d(easting, northing, force_easting, force_northing,
                        fudge=1e-5):
    """
    """
    x = easting - force_easting
    y = northing - force_northing
    distance = np.hypot(x, y)
    return (distance**2)*(np.log(distance + fudge) - 1)
