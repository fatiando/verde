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

The data are a bit too large and oversampled along the ship tracks. We'll
decimate it a bit before interpolation using a block-median. The grid spacing
we'll be using for interpolation will be defined in degrees and we'll do
a rough conversion to Cartesian for the block reduction.

.. jupyter-execute::

   spacing = 0.01  # degrees. 1 deg is roughly 111e3 meters
   median = vd.BlockReduce(np.median, spacing=spacing * 111e3)
   coordinates, bathymetry = median.filter(
      (easting, northing), data.bathymetry_m,
   )
   print(bathymetry.size)

Fit an interpolator to the Cartesian data
-----------------------------------------

Since the data are somewhat large, we'll use the :class:`verde.KNeighbors`
class for interpolation.
We'll use the 50 nearest neighbors to smooth the grid a bit.

.. jupyter-execute::

   interpolator = vd.KNeighbors(50).fit(coordinates, bathymetry)

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
   )
   fig.coast(land="#666666", frame=True)
   fig.colorbar(frame='af+l"bathymetry [m]"')
   fig.show()

Generating a profile in geographic coordinates
----------------------------------------------
