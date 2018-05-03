"""
Generators of synthetic datasets.
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..base import BaseGridder
from ..utils import check_region


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

    The model will evaluated on random points or on a regular grid depending on
    the value of *scatter*.

    Parameters
    ----------
    amplitude : float
        The amplitude of the checkerboard undulations.
    w_east : float
        The wavelength in the east direction. Defaults to half of the West-East
        size of the evaluating region.
    w_north : float
        The wavelength in the north direction. Defaults to half of the
        South-North size of the evaluating region.

    Attributes
    ----------
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the region used to generate the
        synthetic data.

    Examples
    --------
    >>> synth = CheckerBoard().fit()
    >>> # Default values for the region and wavelengths are selected by fit()
    >>> print(synth)
    CheckerBoard(amplitude=1000, w_east=2500.0, w_north=2500.0)
    >>> print(synth.region_)
    (0, 5000, -5000, 0)
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
    >>> table = synth.scatter(random_state=0)
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

    def __init__(self, amplitude=1000, w_east=None, w_north=None):
        self.amplitude = amplitude
        self.w_east = w_east
        self.w_north = w_north

    def fit(self, region=(0, 5000, -5000, 0)):
        """
        Set the region in which the checkerboard will be evaluated.

        If the wavelengths are not supplied, will set them to half of the
        region extent in each dimension.

        Parameters
        ----------
        region : list
            The boundaries (``[W, E, S, N]``) of a given region in Cartesian or
            geographic coordinates.

        Returns
        -------
        self : verde.datasets.CheckerBoard
            Returns this gridder instance for chaining operations.

        """
        check_region(region)
        w, e, s, n = region
        if self.w_east is None:
            self.w_east = (e - w)/2
        if self.w_north is None:
            self.w_north = (n - s)/2
        self.region_ = region
        return self

    def predict(self, easting, northing):
        """
        Evaluate the checkerboard function on a given set of points.

        Parameters
        ----------
        easting : array
            The values of the West-East coordinates of each data point.
        northing : array
            The values of the South-North coordinates of each data point.

        Returns
        -------
        data : array
            The evaluated checkerboard function.

        """
        check_is_fitted(self, ['region_'])
        data = (self.amplitude *
                np.sin((2*np.pi/self.w_east)*easting) *
                np.cos((2*np.pi/self.w_north)*northing))
        return data
