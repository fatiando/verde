"""
Trends in vector data
=====================

Verde provides the :class:`verde.VectorTrend` class to estimate a polynomial
trend on each component of vector data, like GPS velocities. You can access
each trend as a separate :class:`verde.Trend` or operate on all vector
components directly (for gridding, etc).
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import verde as vd


# Fetch the GPS data from the U.S. West coast. The data has a strong trend
# toward the North-West because of the relative movement along the San Andreas
# Fault System.
data = vd.datasets.fetch_california_gps()

# We'll fit a degree 4 trend on both the East and North components and weight
# the data using the variance of each component.
trend = vd.VectorTrend(degree=4)
trend.fit(coordinates=(data.longitude, data.latitude),
          data=(data.velocity_east, data.velocity_north),
          weights=((data.std_east.min()/data.std_east)**2,
                   (data.std_north.min()/data.std_north)**2))
print("Vector trend estimator:", trend)

# The separate Trend objects for each component can be accessed through the
# 'component_' attribute. You could grid them individually if you wanted.
print("East component trend:", trend.component_[0])
print("East trend coefficients:", trend.component_[0].coef_)
print("North component trend:", trend.component_[1])
print("North trend coefficients:", trend.component_[1].coef_)

# We can make a grid with both trend components as data variables
grid = trend.grid(spacing=0.1, dims=['latitude', 'longitude'])
print("\nGridded 2-component trend:")
print(grid)


# Now we can map both trends along with the original data for comparison
def plot_trend(ax, component, title):
    "Make a map of the given trend component on the given axes"
    crs = ccrs.PlateCarree()
    ax.set_title(title)
    # Plot the trend in pseudo color
    maxabs = np.abs([component.min(), component.max()]).max()
    tmp = ax.pcolormesh(component.longitude, component.latitude,
                        component.values, vmin=-maxabs, vmax=maxabs,
                        cmap='seismic', transform=crs)
    cb = plt.colorbar(tmp, ax=ax, orientation='horizontal', pad=0.05)
    cb.set_label('meters/year')
    # Plot the original data
    ax.quiver(data.longitude.values, data.latitude.values,
              data.velocity_east.values, data.velocity_north.values,
              scale=0.2, transform=crs, color='y', label="Original data")
    # and the residuals
    ax.quiver(data.longitude.values, data.latitude.values,
              trend.residual_[0].values, trend.residual_[1].values,
              scale=0.2, transform=crs, color='k', label='Residuals')
    # Setup the map ticks
    ax.set_xticks(np.arange(-124, -115, 4), crs=crs)
    ax.set_yticks(np.arange(33, 42, 2), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.coastlines(color='white')
    ax.set_extent(vd.get_region((data.longitude, data.latitude)), crs=crs)


# Make a plot of the data using Cartopy to handle projections and coastlines
fig, axes = plt.subplots(1, 2, figsize=(9, 7),
                         subplot_kw=dict(projection=ccrs.Mercator()))
# Plot the two trend components
plot_trend(axes[0], grid.east_component, "East component trend")
plot_trend(axes[1], grid.north_component, "North component trend")
axes[0].legend(loc="lower left")
plt.tight_layout()
plt.show()
