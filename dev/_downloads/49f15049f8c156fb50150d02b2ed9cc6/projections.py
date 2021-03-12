"""
Geographic Coordinates
======================

Most gridders and processing methods in Verde operate under the assumption that the data
coordinates are Cartesian. To process data in geographic (longitude and latitude)
coordinates, we must first project them. There are different ways of doing this in
Python but most of them rely on the `PROJ library <https://proj4.org/>`__. We'll use
`pyproj <https://github.com/jswhit/pyproj>`__ to access PROJ directly and handle the
projection operations.
"""
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd

########################################################################################
# With pyproj, we can create functions that will project our coordinates to and from
# different coordinate systems. For our Baja California bathymetry data, we'll use a
# Mercator projection.

data = vd.datasets.fetch_baja_bathymetry()
# We're choosing the latitude of true scale as the mean latitude of our dataset.
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())

########################################################################################
# The Proj object is a callable (meaning that it behaves like a function) that will take
# longitude and latitude and return easting and northing coordinates.

# pyproj doesn't play well with Pandas so we need to convert to numpy arrays
proj_coords = projection(data.longitude.values, data.latitude.values)
print(proj_coords)

########################################################################################
# We can plot our projected coordinates using matplotlib.

plt.figure(figsize=(7, 6))
plt.title("Projected coordinates of bathymetry measurements")
# Plot the bathymetry data locations as black dots
plt.plot(proj_coords[0], proj_coords[1], ".k", markersize=0.5)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

########################################################################################
# Cartesian grids
# ---------------
#
# Now we can use :class:`verde.BlockReduce` and :class:`verde.Spline` on our projected
# coordinates. We'll specify the desired grid spacing as degrees and convert it to
# Cartesian using the 1 degree approx. 111 km rule-of-thumb.
spacing = 10 / 60
reducer = vd.BlockReduce(np.median, spacing=spacing * 111e3)
filter_coords, filter_bathy = reducer.filter(proj_coords, data.bathymetry_m)
spline = vd.Spline().fit(filter_coords, filter_bathy)

########################################################################################
# If we now call :meth:`verde.Spline.grid` we'll get back a grid evenly spaced in
# projected Cartesian coordinates.
grid = spline.grid(spacing=spacing * 111e3, data_names="bathymetry")
print("Cartesian grid:")
print(grid)

########################################################################################
# We'll mask our grid using :func:`verde.distance_mask` to get rid of all the spurious
# solutions far away from the data points.
grid = vd.distance_mask(proj_coords, maxdist=30e3, grid=grid)

plt.figure(figsize=(7, 6))
plt.title("Gridded bathymetry in Cartesian coordinates")
pc = grid.bathymetry.plot.pcolormesh(cmap="viridis", vmax=0, add_colorbar=False)
plt.colorbar(pc).set_label("bathymetry (m)")
plt.plot(filter_coords[0], filter_coords[1], ".k", markersize=0.5)
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()


########################################################################################
# Geographic grids
# ----------------
#
# The Cartesian grid that we generated won't be evenly spaced if we convert the
# coordinates back to geographic latitude and longitude. Verde gridders allow you to
# generate an evenly spaced grid in geographic coordinates through the ``projection``
# argument of the :meth:`~verde.base.BaseGridder.grid` method.
#
# By providing a projection function (like our pyproj ``projection`` object), Verde will
# generate coordinates for a regular grid and then pass them through the projection
# function before predicting data values. This way, you can generate a grid in a
# coordinate system other than the one you used to fit the spline.

# Get the geographic bounding region of the data
region = vd.get_region((data.longitude, data.latitude))
print("Data region in degrees:", region)

# Specify the region and spacing in degrees and a projection function
grid_geo = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="bathymetry",
)
print("Geographic grid:")
print(grid_geo)

########################################################################################
# Notice that grid has longitude and latitude coordinates and slightly different number
# of points than the Cartesian grid.
#
# The :func:`verde.distance_mask` function also supports the ``projection`` argument and
# will project the coordinates before calculating distances.

grid_geo = vd.distance_mask(
    (data.longitude, data.latitude), maxdist=30e3, grid=grid_geo, projection=projection
)

########################################################################################
# Now we can use the Cartopy library to plot our geographic grid.

plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Geographic grid of bathymetry")
pc = grid_geo.bathymetry.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), vmax=0, zorder=-1, add_colorbar=False
)
plt.colorbar(pc).set_label("meters")
vd.datasets.setup_baja_bathymetry_map(ax, land=None)
plt.show()

########################################################################################
# Profiles
# --------
#
# For profiles, things are a bit different. The projection is applied to the
# input points before coordinates are generated. So the profile will be evenly
# spaced in *projected coordinates*, not geographic coordinates. This is to
# avoid issues with calculating distances on a sphere.
#
# The coordinates returned by the ``profile`` method will be in geographic
# coordinates, so projections given to ``profile`` must take an ``inverse``
# argument so we can undo the projection.
#
# The main utility of using a projection with ``profile`` is being able to pass
# in points in geographic coordinates and get coordinates back in that same
# system (making it easier to plot on a map).
#
# To generate a profile cutting across our bathymetry data, we can use
# longitude and latitude points taken from the map above).

start = (-114.5, 24.7)
end = (-110, 20.5)
profile = spline.profile(
    point1=start,
    point2=end,
    size=200,
    projection=projection,
    dims=("latitude", "longitude"),
    data_names=["bathymetry"],
)
print(profile)

########################################################################################
# Plot the profile location on our geographic grid from above.

plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Profile location")
pc = grid_geo.bathymetry.plot.pcolormesh(
    ax=ax, transform=ccrs.PlateCarree(), vmax=0, zorder=-1, add_colorbar=False
)
plt.colorbar(pc).set_label("meters")
ax.plot(profile.longitude, profile.latitude, "-k", transform=ccrs.PlateCarree())
ax.text(start[0], start[1], "A", transform=ccrs.PlateCarree())
ax.text(end[0], end[1], "B", transform=ccrs.PlateCarree())
vd.datasets.setup_baja_bathymetry_map(ax, land=None)
plt.show()

########################################################################################
# And finally plot the profile.

plt.figure(figsize=(8, 3))
ax = plt.axes()
ax.set_title("Profile of bathymetry (A-B)")
ax.plot(profile.distance, profile.bathymetry, "-k")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Bathymetry (m)")
ax.set_xlim(profile.distance.min(), profile.distance.max())
ax.grid()
plt.tight_layout()
plt.show()
