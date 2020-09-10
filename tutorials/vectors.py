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
spacing = 12 / 60

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
plt.show()

########################################################################################
# As expected, the residuals are higher for the North component because of the lower
# degree polynomial.
#
# Let's make geographic grids of these trends.

region = vd.get_region((data.longitude, data.latitude))

grid = trend.grid(
    region=region,
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
    tmp = component.plot.pcolormesh(
        ax=ax,
        vmin=-maxabs,
        vmax=maxabs,
        cmap="bwr",
        transform=crs,
        add_colorbar=False,
        add_labels=False,
    )
    cb = plt.colorbar(tmp, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_label("meters/year")
    vd.datasets.setup_california_gps_map(ax, land=None, ocean=None)
    ax.coastlines(color="white")
plt.show()

########################################################################################
# Gridding
# --------
#
# You can use :class:`verde.Vector` to create multi-component gridders out of
# :class:`verde.Spline` the same way as we did for trends. In this case, each component
# is treated separately.
#
# We can start by splitting the data into training and testing sets (see
# :ref:`model_selection`). Notice that :func:`verde.train_test_split` work for
# multicomponent data automatically.

train, test = vd.train_test_split(
    coordinates=proj_coords,
    data=(data.velocity_east, data.velocity_north),
    weights=(1 / data.std_east ** 2, 1 / data.std_north ** 2),
    random_state=1,
)

########################################################################################
# Now we can make a 2-component spline. Since :class:`verde.Vector` implements
# ``fit``, ``predict``, and ``filter``, we can use it in a :class:`verde.Chain` to build
# a pipeline.
#
# We need to use a bit of damping so that the weights can be taken into account. Splines
# without damping provide a perfect fit to the data and ignore the weights as a
# consequence.

chain = vd.Chain(
    [
        ("mean", vd.BlockMean(spacing=spacing * 111e3, uncertainty=True)),
        ("trend", vd.Vector([vd.Trend(1), vd.Trend(1)])),
        ("spline", vd.Vector([vd.Spline(damping=1e-10), vd.Spline(damping=1e-10)])),
    ]
)
print(chain)

########################################################################################
#
# .. warning::
#
#     Never generate the component gridders with ``[vd.Spline()]*2``. This will result
#     in each component being a represented by **the same Spline object**, causing
#     problems when trying to fit it to different components.
#
# Fitting the spline and gridding is exactly the same as what we've done before.

chain.fit(*train)

# Score on the test data
print(chain.score(*test))

grid = chain.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
)
print(grid)

########################################################################################
# Mask out the points too far from data and plot the gridded vectors.

grid = vd.distance_mask(
    (data.longitude, data.latitude),
    maxdist=spacing * 2 * 111e3,
    grid=grid,
    projection=projection,
)

plt.figure(figsize=(6, 8))
ax = plt.axes(projection=ccrs.Mercator())
tmp = ax.quiver(
    grid.longitude.values,
    grid.latitude.values,
    grid.east_component.values,
    grid.north_component.values,
    scale=0.3,
    transform=crs,
    width=0.002,
)
ax.quiverkey(tmp, 0.2, 0.15, 0.05, label="0.05 m/yr", coordinates="figure")
ax.set_title("Gridded velocities")
vd.datasets.setup_california_gps_map(ax)
plt.show()

########################################################################################
# GPS/GNSS data
# +++++++++++++
#
# For some types of vector data, like GPS/GNSS displacements, the vector
# components are coupled through elasticity. In these cases, elastic Green's
# functions can be used to achieve better interpolation results. The `Erizo
# package <https://github.com/fatiando/erizo>`__ implements some of these
# Green's functions.
#
# .. warning::
#
#     The :class:`verde.VectorSpline2D` class implemented an elastically
#     coupled Green's function but it is deprecated and will be removed in
#     Verde v2.0.0. Please use the implementation in the `Erizo
#     <https://github.com/fatiando/erizo>`__ package instead.
