A taste of Verde
================

Verde offers a wealth of functionality for processing spatial and geophysical
data, like **bathymetry, GPS, temperature, gravity, or anything else that is
measured along a surface**.

While our main focus is on gridding (interpolating on a regular grid), you'll
also find other things like trend removal, data decimation, spatial
cross-validation, and blocked operations.

The library
-----------

Most classes and functions are available through the :mod:`verde` top level
package. Throughout the documentation we'll use ``vd`` as the alias for
:mod:`verde`.

.. jupyter-execute::

    import verde as vd

We'll also import other modules for this example:

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    # For projecting data
    import pyproj
    # For fetching sample datasets
    import ensaio




As an example, let's generate some synthetic data using
:class:`verde.synthetic.CheckerBoard`:

.. jupyter-execute::

    data = vd.synthetic.CheckerBoard().scatter(size=500, random_state=0)
    print(data.head())

The data are random points taken from a checkerboard function and returned to
us in a :class:`pandas.DataFrame`:

.. jupyter-execute::

    plt.figure()
    plt.scatter(data.easting, data.northing, c=data.scalars, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

Now we can use the bi-harmonic spline method [Sandwell1987]_ to fit this
data. First, we create a new :class:`verde.Spline`:

.. jupyter-execute::

    spline = vd.Spline()
    # Printing a gridder shows the class and all of it's configuration options.
    print(spline)

Before we can use the spline, we need to fit it to our synthetic data. After
that, we can use the spline to predict values anywhere:

.. jupyter-execute::

    spline.fit((data.easting, data.northing), data.scalars)

Generate coordinates for a regular grid with 100 m grid spacing (assuming
coordinates are in meters).

.. jupyter-execute::

    grid_coords = vd.grid_coordinates(region=(0, 5000, -5000, 0), spacing=100)
    gridded_scalars = spline.predict(grid_coords)

    plt.figure()
    plt.pcolormesh(grid_coords[0], grid_coords[1], gridded_scalars, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

We can compare our predictions with the true values for the checkerboard
function using the :meth:`~verde.Spline.score` method to calculate the
`R² coefficient of determination
<https://en.wikipedia.org/wiki/Coefficient_of_determination>`__.

.. jupyter-execute::

    true_values = vd.synthetic.CheckerBoard().predict(grid_coords)
    print(spline.score(grid_coords, true_values))

Generating grids and profiles
-----------------------------

A more convenient way of generating grids is through the
:meth:`~verde.base.BaseGridder.grid` method. It will automatically generate
coordinates and output an :class:`xarray.Dataset`.

.. jupyter-execute::

    grid = spline.grid(spacing=30)
    print(grid)

Method :meth:`~verde.base.BaseGridder.grid` uses default names for the
coordinates ("easting" and "northing") and data variables ("scalars"). You can
overwrite these names by setting the ``dims`` and ``data_names`` arguments.

.. jupyter-execute::

    grid = spline.grid(spacing=30, dims=["latitude", "longitude"], data_names="gravity")
    print(grid)

    plt.figure()
    grid.gravity.plot.pcolormesh()
    plt.show()

Gridders can also be used to interpolate data on a straight line between two
points using the :meth:`~verde.base.BaseGridder.profile` method. The profile
data are returned as a :class:`pandas.DataFrame`.

.. jupyter-execute::

    prof = spline.profile(point1=(0, 0), point2=(5000, -5000), size=200)
    print(prof.head())

.. jupyter-execute::

    plt.figure()
    plt.plot(prof.distance, prof.scalars, "-")
    plt.show()

Wrap up
-------

This covers the basics of using Verde. Most use cases and examples in the
documentation will involve some variation of the following workflow:

1. Load data (coordinates and data values)
2. Create a gridder
3. Fit the gridder to the data
4. Predict new values (using :meth:`~verde.base.BaseGridder.predict` or
   :meth:`~verde.base.BaseGridder.grid`)
