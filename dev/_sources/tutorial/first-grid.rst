.. _tutorial-first-grid:

Making your first grid
======================

This tutorial will take you through creating your first grid with Verde.
We'll use one of our sample datasets to demonstrate how to use
:class:`verde.Spline` to make a grid from relatively small datasets (fewer than
20,000 points).

Import what we need
-------------------

First thing to do is import all that we'll need. As usual, we'll import Verde
as ``vd``. We'll load the standard trio of data science in Python (numpy,
pandas, matplotlib) and some extra packages for geospatial data:
`Ensaio <https://www.fatiando.org/ensaio/>`__ which we use download sample
data, and `pyproj <https://pyproj4.github.io/pyproj/stable/>`__ to transform
our data from geographic to Cartesian coordinates.

.. jupyter-execute::

   import pandas as pd
   import matplotlib.pyplot as plt

   import ensaio
   import pyproj

   import verde as vd

Download and read in some data
------------------------------

Now we can use function :func:`ensaio.fetch_alps_gps` to download a sample
dataset for us to use.
This is a GPS dataset from stations along the Alps in Europe.
It contains the velocity with which each station was moving (in mm/year) and is
used for studies of plate tectonics.

.. jupyter-execute::

   path_to_data = ensaio.fetch_alps_gps(version=1)
   print(path_to_data)

Ensaio downloads the data and returns a path to the data file on your computer.
Since this is a CSV file, we can load it with :func:`pandas.read_csv`:

.. jupyter-execute::

   data = pd.read_csv(path_to_data)
   data

Convert from geographic to Cartesian
------------------------------------

Most interpolators and processing functions in Verde require Cartesian
coordinates.
So we can't just provide them with the longitude and latitude in our datasets,
which would cause distortions in our results.
Instead, we'll first **project** the data using :mod:`pyproj`.
We'll use a Mercator projection because our area is far enough away from the
poles to cause any issues:

.. jupyter-execute::

   projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
   easting, northing = projection(data.longitude, data.latitude)

Now we have arrays with easting and northing coordinates in meters. Let's plot
this data with matplotlib to see what we're dealing with:

.. jupyter-execute::

   fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
   # Set the aspect ratio to "equal" so that units in x and y match
   ax.set_aspect("equal")
   tmp = ax.scatter(easting, northing, c=data.velocity_up_mmyr, s=30)
   fig.colorbar(tmp, label="mm/yr")
   ax.set_title("Vertical velocity in the Alps measured by GPS")
   ax.set_xlabel("easting (m)")
   ax.set_ylabel("northing (m)")
   plt.show()

Our data has both positive (upward motion of the ground) and negative (downward
motion of the ground) values, which means that the default colormap used by
matplotlib isn't ideal for our use case.
We should instead use a diverging colormap and make sure the minimum and
maximum values are adjusted to have the middle color map to the zero data
value.
Verde offers function :func:`verde.maxabs` to help do this:

.. jupyter-execute::

   # Get the maximum absolute value
   scale = vd.maxabs(data.velocity_up_mmyr)

   fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
   ax.set_aspect("equal")
   # Use scale to set the vmin and vmax and center the colorbar
   tmp = ax.scatter(
       easting,
       northing,
       c=data.velocity_up_mmyr,
       s=30,
       cmap="RdBu_r",
       vmin=-scale,
       vmax=scale,
   )
   fig.colorbar(tmp, label="mm/yr")
   ax.set_title("Vertical velocity in the Alps measured by GPS")
   ax.set_xlabel("easting (m)")
   ax.set_ylabel("northing (m)")
   plt.show()

Now we can clearly see which points are going up and which ones are going down.
That big region of upward motion are the Alps which are being pushed up by
subduction.
The surrounding regions tend to move downward by flexure caused by the Alps
themselves and by the subduction as well.

Interpolation with bi-harmonic splines
--------------------------------------

The :class:`verde.Spline` class implements the bi-harmonic spline of
[Sandwell1987]_, which is a great method for interpolating smaller datasets
like ours (fewer than 20,000 data points).
It has a higher computation load than other methods but it allows use of data
weights and other neat features to control the smoothness of the solution.

To use it, we'll first create an instance of :class:`verde.Spline`:

.. jupyter-execute::

   spline = vd.Spline()

Now, we can fit it to our data. This will estimate a set of forces that push
on a thin elastic sheet to make it pass through our data.
The :meth:`verde.Spline.fit` method of all interpolators in Verde take the same
arguments: a tuple of coordinates and the corresponding data values (plus
optionally some weights).
The coordinates are **always** specified in **easting and northing order**
(think x and y on a plot).

.. jupyter-execute::

   spline.fit((easting, northing), data.velocity_up_mmyr)

Fitting the spline is the most time consuming part of the interpolation.
But once the spline is fitted, we can use it to make predictions of the data
values wherever we wish by using the :meth:`verde.Spline.predict` method:

.. jupyter-execute::

   coordinates = (0.6e6, 4e6)  # easting, northing in meters
   value = spline.predict(coordinates)
   print(f"Vertical velocity at {coordinates}: {value} mm/yr")

Likewise, we can predict values on a regular grid with the
:meth:`verde.Spline.grid` method.
All it requires is a grid spacing (but it can also take other arguments):

.. jupyter-execute::

   grid = spline.grid(spacing=10e3)
   grid

The generated grid is an :class:`xarray.Dataset` which contains the grid
coordinates, interpolated values, and some metadata.
We can plot this grid with xarray's plotting mechanics:

.. jupyter-execute::

   fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
   ax.set_aspect("equal")
   grid.scalars.plot(ax=ax)
   ax.set_title("Vertical velocity in the Alps measured by GPS")
   ax.set_xlabel("easting (m)")
   ax.set_ylabel("northing (m)")
   plt.show()

Notice that xarray handled choosing an appropriate colormap and centering it
for us.

The plot and grid can be even better if we add more metadata to it, like the
name of the data and its units.

.. jupyter-execute::

   # Rename the data variable and add some metadata
   grid = grid.rename(scalars="velocity_up")
   grid.velocity_up.attrs["long_name"] = "Vertical GPS velocity"
   grid.velocity_up.attrs["units"] = "mm/yr"

   # Make the plot again but plot the data locations on top
   fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
   ax.set_aspect("equal")
   grid.velocity_up.plot(ax=ax)
   ax.plot(easting, northing, ".k", markersize=1)
   ax.set_title("Vertical velocity in the Alps measured by GPS")
   ax.set_xlabel("easting (m)")
   ax.set_ylabel("northing (m)")
   plt.show()

Notice how xarray automatically adds the data name and units to the colorbar
for us!
Finally, you can save the grid to a file with :meth:`xarray.Dataset.to_netcdf`
or other similar methods if you want.

🎉 **Congratulations, you've made your first grid with Verde!** 🎉
