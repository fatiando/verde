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

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import pygmt
   import pyproj
   import ensaio
   import verde as vd

.. jupyter-execute::
   :hide-code:

   # Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
   # the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
   # for sphinx-gallery to work means that fig.show won't display anything here
   # either.
   pygmt.set_display(method="notebook")

To demonstrate how to do this, we'll use the Caribbean bathymetry data from
:mod:`ensaio`:

.. jupyter-execute::

   path_to_data = ensaio.fetch_caribbean_bathymetry(version=2)
   data = pd.read_csv(path_to_data)
   data

Let's plot the data on a map to see what it looks like:

.. jupyter-execute::

   fig = pygmt.Figure()
   pygmt.makecpt(
       cmap="cmocean/topo+h",
       series=[data.bathymetry_m.min(), data.bathymetry_m.max()],
   )
   fig.plot(
       x=data.longitude,
       y=data.latitude,
       fill=data.bathymetry_m,
       cmap=True,
       style="c0.02c",
       projection="M15c",
   )
   fig.coast(land="#666666", frame=True)
   fig.colorbar(frame='af+l"bathymetry [m]"')
   fig.show()

Projecting data
---------------

For this region, a Mercator projection will do fine since we're far away from
the poles and the latitude range isn't very large.
We'll create a projection function and use to project the data coordinates.

.. jupyter-execute::

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
   easting, northing = projection(data.longitude, data.latitude)

We've done this before in :ref:`tutorial-first-grid`. We'll still use the
projected coordinates to decimate the data and fit the interpolator. What's
different is how we'll use the interpolator to make a grid in geographic
coordinates.

Decimating the data
-------------------

The data are a bit too large for an example and oversampled along the ship
tracks. We'll decimate it before interpolation using a block-median operation.
It's easier to think of the block size in degrees so we'll do a rough
conversion to Cartesian for the block reduction.

.. jupyter-execute::

   block_size = 0.15  # degrees. 1 deg is roughly 111e3 meters
   median = vd.BlockReduce(np.median, spacing=block_size * 111e3)
   coordinates, bathymetry = median.filter(
      (easting, northing), data.bathymetry_m,
   )
   print(bathymetry.size)

.. tip::

   The block size used here is larger than we would use normally for this
   dataset. We use it here because this example needs to run relatively quickly
   and on very modest hardward. On your own maching, try using a smaller
   block size, like 0.05 degrees.

Fit an interpolator to the Cartesian data
-----------------------------------------

Use a Cartesian :class:`~verde.Spline` to fit the data and then interpolate.

.. jupyter-execute::

   interpolator = vd.Spline().fit(coordinates, bathymetry)

Generate a grid in geographic coordinates
-----------------------------------------

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

   region = vd.get_region((data.longitude, data.latitude))
   spacing = 2 / 60  # 2 arc-minutes in decimal degrees
   grid = interpolator.grid(
      region=region, spacing=spacing, projection=projection,
      dims=("latitude", "longitude"), data_names="bathymetry",
   )
   grid

Notice that the grid has longitude and latitude coordinates and that they are
evenly spaced.
We can use this grid to plot a map of the bathymetry with coastlines added.

.. jupyter-execute::

   fig = pygmt.Figure()
   pygmt.makecpt(
       cmap="cmocean/topo+h",
       series=[data.bathymetry_m.min(), data.bathymetry_m.max()],
   )
   fig.grdimage(
       grid.bathymetry,
       cmap=True,
       projection="M15c",
       shading=True,
   )
   fig.coast(land="#666666", frame=True)
   fig.colorbar(frame='af+l"bathymetry [m]"')
   fig.show()

Generating a profile in geographic coordinates
----------------------------------------------

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
       point1=(-67, 14),  # longitude, latitude
       point2=(-58.5, 14),
       size=200,  # number of points
       dims=("latitude", "longitude"),
       data_names="bathymetry_m",
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
       cmap="cmocean/topo+h",
       series=[data.bathymetry_m.min(), data.bathymetry_m.max()],
   )
   fig.grdimage(grid.bathymetry, cmap=True, projection="M15c", shading=True)
   fig.coast(land="#666666", frame=True)
   fig.colorbar(frame='af+l"bathymetry [m]"')
   fig.plot(
       x=profile.longitude,
       y=profile.latitude,
       fill="black",
       style="c0.1c",
   )
   # Plot the profile above it
   fig.shift_origin(yshift="h+1.5c")
   fig.plot(
       x=profile.distance,
       y=profile.bathymetry_m,
       pen="1p",
       projection="X15c/5c",
       frame=["WSne", "xaf+lDistance along profile (m)", "yaf+lBathymetry (m)"],
       region=vd.get_region((profile.distance, profile.bathymetry_m)),
   )
   fig.show()


