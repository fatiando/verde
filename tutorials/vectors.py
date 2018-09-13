"""
Vector Data
===========

Some datasets have multiple vector components measured for each location, like the East
and West components of wind speed or GPS velocities. For example, let's look at our
sample GPS velocity data from the U.S. West coast.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import verde as vd

data = vd.datasets.fetch_california_gps()

# We'll need to project the geographic coordinates to work with our Cartesian
# classes/functions
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coords = projection(data.longitude.values, data.latitude.values)
# This will be our desired grid spacing in degrees
spacing = 15 / 60

plt.figure(figsize=(6, 8))
ax = plt.axes(projection=ccrs.Mercator())
crs = ccrs.PlateCarree()
tmp = ax.quiver(
    data.longitude.values,
    data.latitude.values,
    data.velocity_east.values,
    data.velocity_north.values,
    scale=0.3,
    transform=crs,
    width=0.002,
)
ax.quiverkey(tmp, 0.2, 0.15, 0.05, label="0.05 m/yr", coordinates="figure")
ax.set_title("GPS horizontal velocities")
vd.datasets.setup_california_gps_map(ax)
plt.tight_layout()
plt.show()


########################################################################################
# Verde classes and functions are equipped to deal with vector data natively or through
# the use of the :class:`verde.Vector` class. Function and classes that can take vector
# data as input will accept tuples as the ``data`` and ``weights`` arguments. Each
# element of the tuple must be an array with the data values for a component of the
# vector data. As with ``coordinates``, **the order of components must be**
# ``(east_component, north_component, up_component)``.
#
#
# Blocked reductions
# ------------------
#
# Operations with :class:`verde.BlockReduce` and :class:`verde.BlockMean` can handle
# multi-component data automatically. The reduction operation is applied to each data
# component separately. The blocked data and weights will be returned in tuples as well
# following the same ordering as the inputs. This will work for an arbitrary number of
# components.

# Use a blocked mean with uncertainty type weights
reducer = vd.BlockMean(spacing=spacing * 111e3, uncertainty=True)
block_coords, block_data, block_weights = reducer.filter(
    coordinates=proj_coords,
    data=(data.velocity_east, data.velocity_north),
    weights=(1 / data.std_east ** 2, 1 / data.std_north ** 2),
)
print(len(block_data), len(block_weights))

########################################################################################
# We can convert the blocked coordinates back to longitude and latitude to plot with
# Cartopy.

block_lon, block_lat = projection(*block_coords, inverse=True)

plt.figure(figsize=(6, 8))
ax = plt.axes(projection=ccrs.Mercator())
crs = ccrs.PlateCarree()
tmp = ax.quiver(
    block_lon,
    block_lat,
    block_data[0],
    block_data[1],
    scale=0.3,
    transform=crs,
    width=0.002,
)
ax.quiverkey(tmp, 0.2, 0.15, 0.05, label="0.05 m/yr", coordinates="figure")
ax.set_title("Block mean velocities")
vd.datasets.setup_california_gps_map(ax)
plt.tight_layout()
plt.show()

########################################################################################
# Trends
# ------
#
# Trends can't handle vector data automatically, so you can't pass
# ``data=(data.velocity_east, data.velocity_north)`` to :meth:`verde.Trend.fit`. To get
# around that, you can use the :class:`verde.Vector` class to create multi-component
# estimators and gridders from single component ones.
#
# :class:`~verde.Vector` takes an estimator/gridder for each data component and
# implements the :ref:`gridder interface <gridder_interface>` for vector data, fitting
# each estimator/gridder given to a different component of the data.
#
# For example, to fit a trend to our GPS velocities, we need to make a 2-component
# vector trend:

trend = vd.Vector([vd.Trend(4), vd.Trend(1)])
print(trend)

########################################################################################
# We can use the ``trend`` as if it were a regular :class:`verde.Trend` but passing in
# 2-component data to fit. This will fit each data component to a different
# :class:`verde.Trend`.

trend.fit(
    coordinates=proj_coords,
    data=(data.velocity_east, data.velocity_north),
    weights=(1 / data.std_east ** 2, 1 / data.std_north ** 2),
)

########################################################################################
# Each estimator can be accessed through the ``components`` attribute:

print(trend.components)
print("East trend coefficients:", trend.components[0].coef_)
print("North trend coefficients:", trend.components[1].coef_)

########################################################################################
# When we call :meth:`verde.Vector.predict` or :meth:`verde.Vector.grid`, we'll get back
# predictions for two components instead of just one. Each prediction comes from a
# different :class:`verde.Trend`.

pred_east, pred_north = trend.predict(proj_coords)

# Make histograms of the residuals
plt.figure(figsize=(6, 5))
ax = plt.axes()
ax.set_title("Trend residuals")
ax.hist(data.velocity_north - pred_north, bins="auto", label="North", alpha=0.7)
ax.hist(data.velocity_east - pred_east, bins="auto", label="East", alpha=0.7)
ax.legend()
ax.set_xlabel("Velocity (m/yr)")
plt.tight_layout()
plt.show()

########################################################################################
# As expected, the residuals are higher for the North component because of the lower
# degree polynomial.
#
# Let's make geographic grids of these trends.

grid = trend.grid(
    region=vd.get_region((data.longitude, data.latitude)),
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
)
print(grid)

########################################################################################
# By default, the names of the data components in the :class:`xarray.Dataset` are
# ``east_component`` and ``north_component``. This can be customized using the
# ``data_names`` argument.
#
# Now we can map the trends.

fig, axes = plt.subplots(
    1, 2, figsize=(9, 7), subplot_kw=dict(projection=ccrs.Mercator())
)
crs = ccrs.PlateCarree()
titles = ["East component trend", "North component trend"]
components = [grid.east_component, grid.north_component]
for ax, component, title in zip(axes, components, titles):
    ax.set_title(title)
    maxabs = vd.maxabs(component)
    tmp = ax.pcolormesh(
        component.longitude,
        component.latitude,
        component.values,
        vmin=-maxabs,
        vmax=maxabs,
        cmap="bwr",
        transform=crs,
    )
    cb = plt.colorbar(tmp, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_label("meters/year")
    vd.datasets.setup_california_gps_map(ax, land=None, ocean=None)
    ax.coastlines(color="white")
plt.tight_layout()
plt.show()

########################################################################################
# Gridding
# --------
#
#
