"""
Bathymetry data from Baja California
====================================

We provide sample bathymetry data from Baja California to test the gridding
methods. This is the ``@tut_ship.xyz`` sample data from the `GMT
<http://gmt.soest.hawaii.edu/>`__ tutorial. The data is downloaded to a local
directory if it's not there already.
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from verde.datasets import fetch_baja_bathymetry


# The data are in a pandas.DataFrame
data = fetch_baja_bathymetry()
print(data.head())

# Make a plot of the data using Cartopy to handle projections and coastlines
plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title('Bathymetry from Baja California', pad=25)
# Plot the land as a solid color
ax.add_feature(cfeature.LAND, edgecolor='black')
# Plot the bathymetry as colored circles.
# Cartopy requires setting the projection of the original data through the
# transform argument. Use PlateCarree for geographic data.
plt.scatter(data.longitude, data.latitude, c=data.bathymetry_m, s=0.1,
            transform=ccrs.PlateCarree())
cb = plt.colorbar(pad=0.08)
cb.set_label('meters')
ax.gridlines(draw_labels=True)
plt.tight_layout()
plt.show()
