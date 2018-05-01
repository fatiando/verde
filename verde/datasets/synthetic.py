"""
Generators of synthetic datasets.
"""
import numpy as np
import pandas as pd
import xarray as xr

from ..base import BaseGridder


class CheckerBoard(BaseGridder):
    """
    Generate synthetic data in a checkerboard pattern.

    The mathematical model is:

    .. math::

        f(e, n) = a \sin(w_e e^2) \cos(w_n n^2)

    in which :math:`e` is the easting coordinate, :math:`n` is the northing
    coordinate, :math:`a` is the amplitude, and :math:`w_e` and :math:`w_n` are
    the wavenumbers in the east and north directions, respectively.

    The model will evaluated on random points or on a regular grid depending on
    the value of *scatter*.

    Parameters
    ----------
    amplitude : float
        The amplitude of the checkerboard undulations.
    w_east : float
        The wavenumber in the east direction.
    w_north : float
        The wavenumber in the north direction.

    """

    def __init__(self, amplitude=1000, w_east=None, w_north=None):
        self.amplitude = amplitude
        self.w_east = w_east
        seld.w_north = w_north

    def predict(self, easting, northing):
        """
        Evaluate the checkerboard function on a given set of points.

        Parameters
        ----------
        easting : array
            The values of the West-East coordinates.
        northing : array
            The values of the South-North coordinates.

        Returns
        -------
        data : array
            The evaluate checkerboard function.

        """
        data = self.amplitude*np.sin(self.w_east*easting**2)*np.cos(
            self.w_north*northing**2)
        return data
