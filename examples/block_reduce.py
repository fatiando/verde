"""
Data decimation in blocks
=========================

When gridding data that has been highly oversampled in a direction (shipborne
and airborne data, for example), it is important to decimate the data before
interpolation to avoid aliasing. Function :func:`verde.block_reduce` decimates
data by applying a reduction operation (mean, median, mode, max, etc) to the
data in blocks. For non-smooth data, like bathymetry, a blocked median filter
is a good choice.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import verde as vd

# We'll test this on the Baja California shipborne bathymetry data
data = vd.datasets.fetch_baja_bathymetry()

# Decimate the data using a blocked median with 10 arc-minute blocks
coordinates, bathymetry = vd.block_reduce(
    coordinates=(data.longitude, data.latitude), data=data.bathymetry_m,
    reduction=np.median, spacing=10/60)
lon, lat = coordinates

print("Original data size:", data.bathymetry_m.size)
print("Decimated data size:", bathymetry.size)

# Make a plot of the decimated data using Cartopy
plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("10' Block Median Bathymetry", pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry as colored circles.
# Cartopy requires setting the projection of the original data through the
# transform argument. Use PlateCarree for geographic data.
plt.scatter(lon, lat, c=bathymetry, s=5, transform=ccrs.PlateCarree())
cb = plt.colorbar(pad=0.08)
cb.set_label('meters')
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
