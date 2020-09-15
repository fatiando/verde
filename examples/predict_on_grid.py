"""
Interpolate on a preexisting regular grid
=========================================

The best way to interpolate scattered data into a regular grid using any Verde
gridder is through the ``grid`` method, which creates the grid coordinates,
makes the predictions and returns a :class:`xarray.Dataset` containing the
coordinates and the predicted values.

But in case we want to grid some scattered data onto a preexisting regular
grid, we can use the ``predict_onto_grid`` method of any gridder. It takes
a preexisting :class:`xarray.Dataset` and perform the predictions on its grid
coordinates. Then the predicted values are added as a new data array to the
given :class:`xarray.Dataset`.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import numpy as np
import xarray as xr
import verde as vd

# We'll use the Baja California shipborne bathymetry data as our scattered data
data = vd.datasets.fetch_baja_bathymetry()

# Before gridding, we need to decimate the data to avoid aliasing because of the
# oversampling along the ship tracks. We'll use a blocked median with 5 arc-minute
# blocks.
spacing = 5 / 60
reducer = vd.BlockReduce(reduction=np.median, spacing=spacing)
coordinates, bathymetry = reducer.filter(
    (data.longitude, data.latitude), data.bathymetry_m
)

# Supose we have another data set defined on a regular grid.
# Lets create this dataset with some dummy data
longitude, latitude = vd.grid_coordinates(
    region=vd.get_region((data.longitude, data.latitude)), spacing=spacing
)
dummy_data = np.ones_like(longitude)
grid = xr.Dataset(
    data_vars={"dummy": (("latitude", "longitude"), dummy_data)},
    coords={"longitude": longitude[0, :], "latitude": latitude[:, 0]},
)
print("Preexisting grid:", grid)

# Project the data so we can use it as input for the gridder.
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coordinates = projection(*coordinates)

# Initialize and fit the gridder
gridder = vd.ScipyGridder(method="cubic").fit(proj_coordinates, bathymetry)

# Predict on the grid
# The prediction will be added to the preexisting grid.
# We need to specify the name for the predicted data (data_Names) and the names of the
# coordiantes that should be used as northing and easting, in that order.
# Because the grid is in geodetic coordinates, we must also pass the
# projection, so the prediction is carried out in the projected coordiantes.
gridder.predict_onto_grid(
    grid,
    data_names=["bathymetry_m"],
    dims=("latitude", "longitude"),
    projection=projection,
)
print("Grid with predicted values:", grid)

# Plot the gridded bathymetry
crs = ccrs.PlateCarree()
plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Gridded Bathymetry Using Scipy")
pc = grid.bathymetry_m.plot.pcolormesh(
    ax=ax, transform=crs, vmax=0, zorder=-1, add_colorbar=False
)
plt.colorbar(pc).set_label("meters")
ax.coastlines()
plt.show()
