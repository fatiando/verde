"""
Trends in vector data
=====================

Verde provides the :class:`verde.VectorTrend` class to estimate a polynomial trend on
each component of vector data, like GPS velocities. You can access each trend as a
separate :class:`verde.Trend` or operate on all vector components directly using using
:meth:`verde.VectorTrend.predict`, :meth:`verde.VectorTrend.grid`, etc, or chaining it
with a vector interpolator using :class:`verde.Chain`.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import verde as vd


# Fetch the GPS data from the U.S. West coast. The data has a strong trend toward the
# North-West because of the relative movement along the San Andreas Fault System.
data = vd.datasets.fetch_california_gps()

# We'll fit a degree 4 trend on both the East and North components and weight the data
# using the inverse of the variance of each component.
trend = vd.VectorTrend(degree=4)
weights = vd.variance_to_weights((data.std_east ** 2, data.std_north ** 2))
trend.fit(
    coordinates=(data.longitude, data.latitude),
    data=(data.velocity_east, data.velocity_north),
    weights=weights,
)
print("Vector trend estimator:", trend)

# The separate Trend objects for each component can be accessed through the 'component_'
# attribute. You could grid them individually if you wanted.
print("East component trend:", trend.component_[0])
print("East trend coefficients:", trend.component_[0].coef_)
print("North component trend:", trend.component_[1])
print("North trend coefficients:", trend.component_[1].coef_)

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
    tmp = ax.pcolormesh(
        component.longitude,
        component.latitude,
        component.values,
        vmin=-maxabs,
        vmax=maxabs,
        cmap="seismic",
        transform=crs,
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
        color="y",
        label="Original data",
    )
    # and the residuals
    ax.quiver(
        data.longitude.values,
        data.latitude.values,
        trend.residual_[0].values,
        trend.residual_[1].values,
        scale=0.2,
        transform=crs,
        color="k",
        label="Residuals",
    )
    # Setup the map ticks
    vd.datasets.setup_california_gps_map(ax, land=None, ocean=None)
    ax.coastlines(color="white")
axes[0].legend(loc="lower left")
plt.tight_layout()
plt.show()
