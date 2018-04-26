"""
Spline interpolation using Green's functions.

Estimates the intensity of vertical forces acting on an elastic line or sheet.
The interpolated data are the deflections of this sheet by the forces.
"""
import numpy as np


def spline1d_green():
    """
    The Green's function for the 1D spline.
    """
    return np.ones(10)
