.. _overview:

A taste of Verde
================

Verde offers a wealth of functionality for processing spatial and geophysical
data, like **bathymetry, GPS, temperature, gravity, or anything else that is
measured along a surface**.
While our main focus is on gridding (interpolating on a regular grid), you'll
also find other things like trend removal, data decimation, spatial
cross-validation, and blocked operations.

This example will show you some of what Verde can do to process some data and
generate a grid.

.. seealso::

   Looking for a simpler and more detailed tutorial? Start with
   :ref:`tutorial-first-grid`.

The library
-----------

Most classes and functions are available through the :mod:`verde` top level
package. So we can import only that and we'll have everything Verde has to offer:


.. jupyter-execute::

    import verde as vd

.. note::

    Throughout the documentation we'll use ``vd`` as the alias for
    :mod:`verde`.

We'll also import other modules for this example:

.. jupyter-execute::

    # Standard Scipy stack
    import pandas as pd
    import matplotlib.pyplot as plt
    # For projecting data
    import pyproj
    # For plotting data on a map
    import pygmt
    # For fetching sample datasets
    import ensaio


.. jupyter-execute::
   :hide-code:

   # Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
   # the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
   # for sphinx-gallery to work means that fig.show won't display anything here
   # either.
   pygmt.set_display(method="notebook")

Loading some sample data
------------------------

For this example, we'll download some sample GPS/GNSS velocity data from across
the Alps using :mod:`ensaio`:

.. jupyter-execute::

    path_to_data_file = ensaio.fetch_alps_gps(version=1)
    print(path_to_data_file)

Since our data are in `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`__
format, the best way to load them is with :mod:`pandas`:

.. jupyter-execute::

    data = pd.read_csv(path_to_data_file)
    data

The data are the observed 3D velocity vectors of each GPS/GNSS station in
mm/year and show the deformation of the crust that is caused by the subduction
in the Alps.
For this example, **we'll only the vertical component** (but Verde can handle
all 3 components as well).

Before we do anything with this data, it would be useful to extract from it the
West, East, South, North bounding box (this is called a **region** in Verde).
This will help us make a map and can be useful in other operations as well.
Verde offers the function :func:`verde.get_region` function for doing just
that:

.. jupyter-execute::

   region = vd.get_region([data.longitude, data.latitude])
   print(region)

.. admonition:: Coordinate order
   :class: tip

   In Verde, coordinates are always given in the order:
   **West-East, South-North**. All functions and classes expect coordinates in
   this order. The **only exceptions** are the ``dims`` and ``shape`` arguments
   that some functions take.


Let's plot this on a :mod:`pygmt` map so we can see it more clearly:

.. jupyter-execute::

   # Start a figure
   fig = pygmt.Figure()
   # Add a basemap with the data region, Mercator projection, default frame
   # and ticks, color in the continents, and display national borders
   fig.coast(
       region=region, projection="M15c", frame="af",
       land="#eeeeee", borders="1/#666666", area_thresh=1e4,
   )
   # Create a colormap for the velocity
   pygmt.makecpt(
       cmap="polar+h",
       series=[data.velocity_up_mmyr.min(), data.velocity_up_mmyr.max()],
   )
   # Plot colored points for the velocities
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       fill=data.velocity_up_mmyr,
       style="c0.2c",
       cmap=True,
       pen="0.5p,black",
   )
   # Add a colorbar with automatic frame and ticks and a label
   fig.colorbar(frame='af+l"vertical velocity [mm/yr]"')
   fig.show()

Decimate the data
-----------------

You may have noticed that that the spacing between the points is highly
variable.
This can sometimes cause aliasing problems when gridding and also wastes
computation when multiple points would fall on the same grid cell.
To avoid all of the this, it's customary to **block average** the data first.

Block averaging means splitting the region into blocks (usually with the size
of the desired grid spacing) and then taking the average of all points inside
each block.
In Verde, this is done by :class:`verde.BlockMean`:

.. jupyter-execute::

   # Desired grid spacing in degrees
   spacing = 0.2
   blockmean = vd.BlockMean(spacing=spacing)

The :meth:`verde.BlockMean.filter` method applies the blocked average operation
with the given spacing to some data.
It returns for each block: the mean coordinates, the mean data value, and
a weight (we'll get to that soon).

.. jupyter-execute::

   block_coordinates, block_velocity, block_weights = blockmean.filter(
       coordinates=(data.longitude, data.latitude),
       data=data.velocity_up_mmyr,
   )
   block_coordinates

In this case, we have **uncertainty** data for each observation and so we can
pass that as **input weights** to the block averaging (and compute a
weighted average instead).
The weights should usually be **1 over the uncertainty squared** and we need to
let :class:`verde.BlockMean` know that our input weights are uncertainties.
**It's always recommended to use weights if you have them!**

.. jupyter-execute::

   blockmean = vd.BlockMean(spacing=spacing, uncertainty=True)
   block_coordinates, block_velocity, block_weights = blockmean.filter(
       coordinates=(data.longitude, data.latitude),
       data=data.velocity_up_mmyr,
       weights=1 / data.velocity_up_error_mmyr**2,
   )

.. admonition:: What if I don't have uncertainties?
   :class: attention

   Don't worry! **Input weights are optional** in Verde and can always be
   ommited.

.. admonition:: Block weights

   The weights that are returned by :meth:`verde.BlockMean.filter` can be
   different things. See :class:`verde.BlockMean` for a detailed explanation.
   In our case, they are 1 over the propagated uncertainty of the mean values
   for each block.
   These can be used in the gridding process to give less weight to the data
   that have higher uncertainties.

Now let's plot the block-averaged data:

.. jupyter-execute::

   fig = pygmt.Figure()
   fig.coast(
       region=region, projection="M15c", frame="af",
       land="#eeeeee", borders="1/#666666", area_thresh=1e4,
   )
   pygmt.makecpt(
       cmap="polar+h", series=[block_velocity.min(), block_velocity.max()],
   )
   fig.plot(
       x=block_coordinates[0],
       y=block_coordinates[1],
       fill=block_velocity,
       style="c0.2c",
       cmap=True,
       pen="0.5p,black",
   )
   fig.colorbar(frame='af+l"vertical velocity [mm/yr]"')
   fig.show()

It may not seem like much happened, but if you look closely you'll see that
data points that were closer than the spacing were combined into one.

Project the data
----------------

In this example, we'll use Verde's Cartesian interpolators.
So we need to project the geographic data we have to Cartesian coordinates
first.
We'll use :mod:`pyproj` to create a projection function and convert our
longitude and latitude to easting and northing Mercator projection coordinates.

.. jupyter-execute::

   # Create a Mercator projection with latitude of true scale as the data mean
   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

   easting, northing = projection(block_coordinates[0], block_coordinates[1])

Spline interpolation
--------------------

Since our data are relatively small (< 10k points), we can use the
:class:`verde.Spline` class for bi-harmonic spline interpolation
[Sandwell1987]_ to get a smooth surface that best fits the data:

.. jupyter-execute::

   # Generate a spline with the default arguments
   spline = vd.Spline()
   # Fit the spline to our decimated and projected data
   spline.fit(
       coordinates=(easting, northing),
       data=block_velocity,
       weights=block_weights,
   )

.. admonition:: Have more than 10k data?
   :class: seealso

   You may want to use some of our other interpolators instead, like
   :class:`~verde.KNeighbors` or :class:`~verde.Cubic`. The bi-harmonic spline
   is very memory intensive so avoid using it for >10k data unless you have a
   lot of RAM.

Now that we have a fitted spline, we can use it to **make predictions** at any
location we want using :meth:`verde.Spline.predict`.
For example, we can predict on the original data points to calculate the
**residuals** and evaluate how well the spline fits our data.
To do this, we'll have to project the original coordinates first:

.. jupyter-execute::

   prediction = spline.predict(projection(data.longitude, data.latitude))
   residuals = data.velocity_up_mmyr - prediction

   fig = pygmt.Figure()
   fig.coast(
       region=region, projection="M15c", frame="af",
       land="#eeeeee", borders="1/#666666", area_thresh=1e4,
   )
   pygmt.makecpt(
       cmap="polar+h", series=[residuals.min(), residuals.max()],
   )
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       fill=residuals,
       style="c0.2c",
       cmap=True,
       pen="0.5p,black",
   )
   fig.colorbar(frame='af+l"fit residuals [mm/yr]"')
   fig.show()

As you can see by the colorbar, the residuals are quite small meaning that the
spline fits the decimated data very well.

Generating a grid
-----------------

To make a grid using our spline interpolation, we can use
:meth:`verde.Spline.grid`:

.. jupyter-execute::

   # Set the spacing between grid nodes in km
   grid = spline.grid(spacing=20e3)
   grid

The generated grid is an :class:`xarray.Dataset` and is **Cartesian by
default**.
The grid contains some metadata and default names for the coordinates and data
variables.
Plotting the grid with matplotlib is as easy as:

.. jupyter-execute::

   # scalars is the default name Verde gives to data variables
   grid.scalars.plot()

But it's not that easy to draw borders and coastlines on top of this Cartesian
grid.

To generate a **geographic grid** with longitude and latitude, we can pass in
the geographic region and the projection used like so:

.. jupyter-execute::

   # Spacing in degrees and customize the names of the dimensions and variables
   grid = spline.grid(
       region=region,
       spacing=spacing,
       dims=("latitude", "longitude"),
       data_names="velocity_up",
       projection=projection,  # Our projection function from earlier
   )
   grid


Plotting a geographic grid is easier done with PyGMT so we can add coastlines
and country borders:

.. jupyter-execute::

   fig = pygmt.Figure()
   fig.grdimage(grid.velocity_up, cmap="polar+h", projection="M15c")
   fig.coast(
       frame="af", shorelines="#333333", borders="1/#666666", area_thresh=1e4,
   )
   fig.colorbar(frame='af+l"vertical velocity [mm/yr]"')
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       fill="#333333",
       style="c0.1c",
   )
   fig.show()

.. admonition:: Did you notice?
   :class: hint

   The :class:`verde.Spline` was fitted only once on the input that and we then
   used it to generate 3 separate interpolations. In general, fitting is the
   most time-consuming part for bi-harmonic splines.

Extracting a profile
--------------------

Once we have a fitted spline, we can also use it to predict data along a
profile with the :meth:`verde.Spline.profile` method. For example, let's
extract a profile that cuts across the Alps:

.. jupyter-execute::

   profile = spline.profile(
       point1=(4, 51),  # longitude, latitude of a point
       point2=(11, 42),
       size=200,  # number of points
       dims=("latitude", "longitude"),
       data_names="velocity_up_mmyr",
       projection=projection,
   )
   profile

.. note::

   We passed in a **projection** because our spline is Cartesian but we want to
   generate a profile between 2 points specified with geographic coordinates.
   The resulting points will be evenly spaced in the projected coordinates.

The result is a :class:`pandas.DataFrame` with the coordinates, distance along
the profile, and interpolated data values.
Let's plot the location of the profile on our map:

.. jupyter-execute::

   fig = pygmt.Figure()
   fig.grdimage(grid.velocity_up, cmap="polar+h", projection="M15c")
   fig.coast(
       frame="af", shorelines="#333333", borders="1/#666666", area_thresh=1e4,
   )
   fig.colorbar(frame='af+l"vertical velocity [mm/yr]"')
   fig.plot(
       x=profile.longitude,
       y=profile.latitude,
       pen="2p,#333333",
   )
   fig.show()

Finally, we can plot the profile data using :mod:`matplotlib`:

.. jupyter-execute::

   plt.figure(figsize=(12, 6))
   plt.plot(profile.distance, profile.velocity_up_mmyr, "-")
   plt.title("Vertical GPS/GNSS velocity across the Alps")
   plt.xlabel("Distance along North-South profile (m)")
   plt.ylabel("velocity (mm/yr)")
   plt.xlim(profile.distance.min(), profile.distance.max())
   plt.grid()
   plt.show()

Wrapping up
-----------

This covers the basics of using Verde. Most use cases will involve some
variation of the following workflow:

1. Load data (coordinates and data values)
2. Create a gridder
3. Fit the gridder to the data
4. Predict new values (using :meth:`~verde.base.BaseGridder.predict`,
   :meth:`~verde.base.BaseGridder.grid`, or
   :meth:`~verde.base.BaseGridder.profile`)

**Now go explore the rest of the documentation and try out Verde on your own
data!**

.. admonition:: Questions or comments?
   :class: seealso

   Reach out to us through one of our `communication channels
   <https://www.fatiando.org/contact/>`__! We love hearing from users and are
   always looking for more people to get involved with developing Verde.
