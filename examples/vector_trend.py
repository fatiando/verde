"""
Trends in vector data
=====================

Verde provides the :class:`verde.Trend` class to estimate a polynomial trend and the
:class:`verde.Vector` class to apply any combination of estimators to each component
of vector data, like GPS velocities. You can access each component as a separate
(fitted) :class:`verde.Trend` instance or operate on all vector components directly
using using :meth:`verde.Vector.predict`, :meth:`verde.Vector.grid`, etc, or
chaining it with a vector interpolator using :class:`verde.Chain`.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import verde as vd


# Fetch the GPS data from the U.S. West coast. The data has a strong trend toward the
# North-West because of the relative movement along the San Andreas Fault System.
data = vd.datasets.fetch_california_gps()

# We'll fit a degree 2 trend on both the East and North components and weight the data
# using the inverse of the variance of each component.
# Note: Never use [Trend(...)]*2 as an argument to Vector. This creates references
# to the same Trend instance and will mess up the fitting.
trend = vd.Vector([vd.Trend(degree=2) for i in range(2)])
weights = vd.variance_to_weights((data.std_east ** 2, data.std_north ** 2))
trend.fit(
    coordinates=(data.longitude, data.latitude),
    data=(data.velocity_east, data.velocity_north),
    weights=weights,
)
print("Vector trend estimator:", trend)

# The separate Trend objects for each component can be accessed through the 'components'
# attribute. You could grid them individually if you wanted.
print("East component trend:", trend.components[0])
print("East trend coefficients:", trend.components[0].coef_)
print("North component trend:", trend.components[1])
print("North trend coefficients:", trend.components[1].coef_)

# We can make a grid with both trend components as data variables
grid = trend.grid(spacing=0.1, dims=["latitude", "longitude"])
print("\nGridded 2-component trend:")
print(grid)


# Now we can map both trends along with the original data for comparison
fig, axes = plt.subplots(
    1, 2, figsize=(9, 7), subplot_kw=dict(projection=ccrs.Mercator())
)
crs = ccrs.PlateCarree()
# Plot the two trend components
titles = ["East component trend", "North component trend"]
components = [grid.east_component, grid.north_component]
for ax, component, title in zip(axes, components, titles):
    ax.set_title(title)
    # Plot the trend in pseudo color
    maxabs = vd.maxabs(component)
    tmp = component.plot.pcolormesh(
        ax=ax,
        vmin=-maxabs,
        vmax=maxabs,
        cmap="seismic",
        transform=crs,
        add_colorbar=False,
        add_labels=False,
    )
    cb = plt.colorbar(tmp, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_label("meters/year")
    # Plot the original data
    ax.quiver(
        data.longitude.values,
        data.latitude.values,
        data.velocity_east.values,
        data.velocity_north.values,
        scale=0.2,
        transform=crs,
        color="k",
        width=0.001,
        label="Original data",
    )
    # Setup the map ticks
    vd.datasets.setup_california_gps_map(
        ax, land=None, ocean=None, region=vd.get_region((data.longitude, data.latitude))
    )
    ax.coastlines(color="white")
axes[0].legend(loc="lower left")
plt.show()
