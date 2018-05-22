# pylint: disable=abstract-method
"""
Generators of synthetic datasets.
"""
import numpy as np

from ..base import BaseGridder
from ..coordinates import check_region


class CheckerBoard(BaseGridder):
    r"""
    Generate synthetic data in a checkerboard pattern.

    The mathematical model is:

    .. math::

        f(e, n) = a
        \sin\left(\frac{2\pi}{w_e} e\right)
        \cos\left(\frac{2\pi}{w_n} n\right)

    in which :math:`e` is the easting coordinate, :math:`n` is the northing
    coordinate, :math:`a` is the amplitude, and :math:`w_e` and :math:`w_n` are
    the wavelengths in the east and north directions, respectively.

    Parameters
    ----------
    amplitude : float
        The amplitude of the checkerboard undulations.
    region : tuple
        The boundaries (``[W, E, S, N]``) of the region used to generate the
        synthetic data.
    w_east : float
        The wavelength in the east direction. Defaults to half of the West-East
        size of the evaluating region.
    w_north : float
        The wavelength in the north direction. Defaults to half of the
        South-North size of the evaluating region.

    Examples
    --------
    >>> synth = CheckerBoard()
    >>> # Default values for the wavelengths are selected automatically
    >>> print(synth.w_east_, synth.w_north_)
    2500.0 2500.0
    >>> # grid produces an xarray.Dataset with data on a regular grid
    >>> grid = synth.grid(shape=(11, 6))
    >>> grid.northing.values
    array([-5000., -4500., -4000., -3500., -3000., -2500., -2000., -1500.,
           -1000.,  -500.,     0.])
    >>> grid.easting.values
    array([   0., 1000., 2000., 3000., 4000., 5000.])
    >>> # Lower printing precision to shorten this example
    >>> import numpy as np; np.set_printoptions(precision=1, suppress=True)
    >>> print(grid.scalars.values)
    [[   0.   587.8 -951.1  951.1 -587.8   -0. ]
     [   0.   181.6 -293.9  293.9 -181.6   -0. ]
     [  -0.  -475.5  769.4 -769.4  475.5    0. ]
     [  -0.  -475.5  769.4 -769.4  475.5    0. ]
     [   0.   181.6 -293.9  293.9 -181.6   -0. ]
     [   0.   587.8 -951.1  951.1 -587.8   -0. ]
     [   0.   181.6 -293.9  293.9 -181.6   -0. ]
     [  -0.  -475.5  769.4 -769.4  475.5    0. ]
     [  -0.  -475.5  769.4 -769.4  475.5    0. ]
     [   0.   181.6 -293.9  293.9 -181.6   -0. ]
     [   0.   587.8 -951.1  951.1 -587.8   -0. ]]
    >>> # Use the random_state argument to seed the random number generator
    >>> table = synth.scatter()
    >>> # scatter and profile generate pandas.DataFrame objects
    >>> # Lower printing precision to shorten this example
    >>> import pandas as pd; pd.set_option('precision', 1)
    >>> print(table.head())
       northing  easting  scalars
    0    -467.2   2744.1    222.3
    1   -1129.8   3575.9   -404.4
    2   -3334.3   3013.8   -482.6
    3   -4594.5   2724.4    280.2
    4   -2963.8   2118.3   -322.8
    >>> print(table.tail())
         northing  easting  scalars
    295     -10.2   1121.6    317.1
    296   -3189.1    489.2   -151.0
    297   -2646.8   4311.0   -920.7
    298   -3108.8   4864.6    -13.6
    299    -102.4   4804.2   -457.0
    >>> profile = synth.profile(point1=(0, 0), point2=(2500, -2500), size=100)
    >>> print(profile.head())
       northing  easting  distance  scalars
    0       0.0      0.0       0.0      0.0
    1     -25.3     25.3      35.7     63.3
    2     -50.5     50.5      71.4    125.6
    3     -75.8     75.8     107.1    185.8
    4    -101.0    101.0     142.8    243.1
    >>> print(profile.tail())
        northing  easting  distance  scalars
    95   -2399.0   2399.0    3392.7 -2.4e+02
    96   -2424.2   2424.2    3428.4 -1.9e+02
    97   -2449.5   2449.5    3464.1 -1.3e+02
    98   -2474.7   2474.7    3499.8 -6.3e+01
    99   -2500.0   2500.0    3535.5 -2.4e-13

    """

    def __init__(self, amplitude=1000, region=(0, 5000, -5000, 0), w_east=None,
                 w_north=None):
        self.amplitude = amplitude
        self.w_east = w_east
        self.w_north = w_north
        self.region = region

    @property
    def w_east_(self):
        "Use half of the E-W extent"
        if self.w_east is None:
            return (self.region[1] - self.region[0])/2
        return self.w_east

    @property
    def w_north_(self):
        "Use half of the N-S extent"
        if self.w_north is None:
            return (self.region[3] - self.region[2])/2
        return self.w_north

    @property
    def region_(self):
        "Used to fool the BaseGridder methods"
        check_region(self.region)
        return self.region

    def predict(self, coordinates):
        """
        Evaluate the checkerboard function on a given set of points.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : array
            The evaluated checkerboard function.

        """
        easting, northing = coordinates[:2]
        data = (self.amplitude *
                np.sin((2*np.pi/self.w_east_)*easting) *
                np.cos((2*np.pi/self.w_north_)*northing))
        return data
