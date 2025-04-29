.. _tutorial-geographic:

Interpolation in geographic coordinates
=======================================

Most interpolators and processing methods in Verde operate under the assumption
that the data coordinates are Cartesian. To process data in geographic
(longitude and latitude) coordinates, we must first project them. There are
different ways of doing this in Python but most of them rely on the `PROJ
library <https://proj4.org/>`__ under the hood. We'll use `pyproj
<https://github.com/jswhit/pyproj>`__ to access PROJ directly and handle the
projection operations.
Verde then offers ways of passing the projection information to use the
Cartesian interpolators to generate geographic grids and profiles.

.. jupyter-execute::

   import pandas as pd
   import pygmt
   import pyproj
   import ensaio
   import verde as vd

Fetch some data
---------------

To demonstrate how to do this, we'll use Alps GPS velocity data from
:mod:`ensaio` once again:

.. jupyter-execute::

   path_to_data = ensaio.fetch_alps_gps(version=1)
   data = pd.read_csv(path_to_data)

We'll get the data region and add a little padding to it so that our
interpolation goes beyond the data bounding box.

.. jupyter-execute::

   region = vd.pad_region(
       vd.get_region((data.longitude, data.latitude)),
       1,  # degree
   )

Let's plot the data on a map with coastlines and country borders to see what it
looks like:

.. jupyter-execute::

   fig = pygmt.Figure()
   fig.coast(
       region=region, projection="M15c", frame="af",
       land="#eeeeee", borders="1/#666666", area_thresh=1e4,
   )
   pygmt.makecpt(
       cmap="polar+h",
       series=[data.velocity_up_mmyr.min(), data.velocity_up_mmyr.max()],
   )
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       fill=data.velocity_up_mmyr,
       style="c0.2c",
       cmap=True,
       pen="0.5p,black",
   )
   fig.colorbar(frame='af+l"vertical velocity [mm/yr]"')
   fig.show()

This is much better than just plotting the projected data with no context! Now
we can see that the Alps region is moving upward (red dots) and the surrounding
regions are moving downward (blue dots).

Projecting data
---------------

For this region, a Mercator projection will do fine since we're not too close
to the poles and the latitude range isn't very large. We'll create a projection
function and use to project the data coordinates.

.. jupyter-execute::

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
   coordinates = projection(data.longitude, data.latitude)

We've done this before in :ref:`tutorial-first-grid`. We'll still use the
projected coordinates to decimate the data and fit the interpolator. What's
different is how we'll use the interpolator to make a grid in geographic
coordinates.

Fit an interpolator to the Cartesian data
-----------------------------------------

Use a Cartesian :class:`~verde.Spline` to fit the data, like we did previously.

.. jupyter-execute::

   interpolator = vd.Spline().fit(coordinates, data.velocity_up_mmyr)

Now we can use this interpolator for gridding and predicting a profile.

Make a grid in geographic coordinates
-------------------------------------

The interpolator is inherently Cartesian. If we wanted to use to generate
a grid in geographic coordinates, we would have to:

1. Generate grid coordinates on a geographic system.
2. Project the grid coordinates to Cartesian.
3. Pass the projected coordinates to the ``predict`` method of the
   interpolator.
4. Generate an :class:`xarray.Dataset` with the grid values and the geographic
   coordinates.

To facilitate this, the ``grid`` and ``profile`` methods of Verde interpolators
take a ``projection`` argument. If this is passed, Verde will do the steps
above and generate a grid/profile in geographic coordinates.
In this case, the ``region`` and ``spacing`` arguments must be given in
**geographic** coordinates.

.. jupyter-execute::

   grid = interpolator.grid(
       spacing=10 / 60,  # 10 arc-minutes in decimal degrees
       region=region,
       projection=projection,
       dims=("latitude", "longitude"),
       data_names="velocity_up",
   )
   grid


.. hint::

   Notice that we set the ``dims`` and ``data_names`` arguments above. Those
   control the names of the coordinates and the data variables in the final
   grid. It's useful to set those to avoid Verde's default names, which for
   this case wouldn't be appropriate.

Notice that the grid has longitude and latitude coordinates and that they are
evenly spaced.
We can use this grid to plot a map of the vertical velocity with coastlines
and country borders added.

.. jupyter-execute::

   fig = pygmt.Figure()
   pygmt.makecpt(
       cmap="polar+h",
       series=[data.velocity_up_mmyr.min(), data.velocity_up_mmyr.max()],
   )
   fig.grdimage(
       grid.velocity_up,
       cmap=True,
       projection="M15c",
       frame="af",
   )
   fig.colorbar(frame='af+l"upward velocity (mm/yr)"')
   fig.coast(
       shorelines="#333333", borders="1/#666666", area_thresh=1e4,
   )
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       style="c0.1c",
       fill="#888888",
   )
   fig.show()

Make a profile in geographic coordinates
----------------------------------------

Profiles in geographic coordinates would require a similar workflow to grids:

1. Project the geophysics coordinates of the points to Cartesian.
2. Generate the profile coordinates using the Cartesian points.
3. Pass the Cartesian profile coordinates to the ``predict`` method of the
   interpolator.
4. Convert the projected profile coordinates to geographic with an inverse
   projection.

Once again, we pass the ``projection`` argument to the ``profile`` method of
the interpolator and let it do the work for us.

.. jupyter-execute::

   profile = interpolator.profile(
       point1=(4, 51),  # longitude, latitude
       point2=(11, 42),
       size=200,  # number of points
       dims=("latitude", "longitude"),
       data_names="velocity_up",
       projection=projection,
   )
   profile


The output comes as a :class:`pandas.DataFrame` with the longitude and latitude
coordinates of the points. The distance is calculated from the projected
coordinates and is **not a great circle distance**.

Lets plot the profile coordinates onto our map and the profile itself to see
what it looks like:

.. jupyter-execute::

   fig = pygmt.Figure()
   # Plot the grid
   pygmt.makecpt(
       cmap="polar+h",
       series=[data.velocity_up_mmyr.min(), data.velocity_up_mmyr.max()],
   )
   fig.grdimage(grid.velocity_up, cmap=True, projection="M15c", frame="af")
   fig.colorbar(frame='af+l"upward velocity (mm/yr)"')
   fig.coast(shorelines="#333333", borders="1/#666666", area_thresh=1e4)
   fig.plot(
       x=profile.longitude,
       y=profile.latitude,
       fill="#888888",
       style="c0.1c",
   )
   # Plot the profile above it
   fig.shift_origin(yshift="h+1.5c")
   fig.plot(
       x=profile.distance,
       y=profile.velocity_up,
       pen="1p",
       projection="X15c/5c",
       frame=["WSne", "xaf+lDistance along profile (m)", "yaf+lUpward velocity (mm/yr)"],
       region=vd.get_region((profile.distance, profile.velocity_up)),
   )
   fig.show()
