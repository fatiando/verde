"""
Chaining operations
===================

The :class:`verde.Chain` class allows us to created gridders that perform
multiple estimator operations on data. It will fit each estimator on the data
residuals of the previous one. For example, say we want to calculate a
polynomial trend for our data, fit a gridder on the residual values (not the
trend), but then make a grid of the original data values. This is useful
because many gridders can't handle trends in data.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pyproj
import verde as vd

# Load the Rio de Janeiro total field magnetic anomaly data
data = vd.datasets.fetch_rio_magnetic_anomaly()
region = vd.get_region(data.longitude, data.latitude)

# Before gridding, we need to decimate the data using a blocked mean to avoid
# aliasing
spacing = 0.5/60
lon, lat, total_field = vd.block_reduce(data.longitude, data.latitude,
                                        data.total_field_anomaly_nt,
                                        reduction=np.median, spacing=spacing)

# Project the data using pyproj so that we can use it as input for the gridder.
# We'll set the latitude of true scale to the mean latitude of the data.
projection = pyproj.Proj(proj='merc', lat_ts=data.latitude.mean())
coordinates = projection(lon, lat)

# Create a chain that fits a 2nd degree trend to the anomaly data and then a
# standard gridder to the residuals
chain = vd.Chain([('trend', vd.Trend(degree=2)),
                  ('gridder', vd.ScipyGridder())])
print("Chained estimator:", chain)
# Calling 'fit' will automatically run the data through the chain
chain.fit(coordinates, total_field)

# Each component of the chain can be accessed separately using the
# 'named_steps' attribute
grid_trend = chain.named_steps['trend'].grid()
print("\nTrend grid:")
print(grid_trend)

grid_residual = chain.named_steps['gridder'].grid()
print("\nResidual grid:")
print(grid_residual)

# Chain.grid will use both the trend and the gridder to predict values.
# We'll use the 'projection' keyword to generate a geographic grid instead of
# Cartesian
grid = chain.grid(region=region, spacing=spacing, projection=projection,
                  dims=['latitude', 'longitude'],
                  data_names=['total_field_anomaly'])
print("\nChained geographic grid:")
print(grid)

# We'll plot only the chained grid
crs = ccrs.PlateCarree()

plt.figure(figsize=(7, 5))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Chained trend and gridder")
maxabs = np.max(np.abs([grid.total_field_anomaly.min(),
                        grid.total_field_anomaly.max()]))
pc = ax.pcolormesh(grid.longitude, grid.latitude, grid.total_field_anomaly,
                   transform=crs, cmap='seismic', vmin=-maxabs, vmax=maxabs)
cb = plt.colorbar(pc, pad=0.01)
cb.set_label('total field anomaly [nT]')
# Plot the locations of the decimated data
ax.plot(lon, lat, '.k', markersize=0.5, transform=crs)
# Set the proper ticks for a Cartopy map
ax.set_xticks(np.arange(-42.5, -42, 0.1), crs=crs)
ax.set_yticks(np.arange(-22.4, -22, 0.1), crs=crs)
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent(region, crs=crs)
plt.tight_layout()
plt.show()
