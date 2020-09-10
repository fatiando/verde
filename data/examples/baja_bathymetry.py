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
import verde as vd


# The data are in a pandas.DataFrame
data = vd.datasets.fetch_baja_bathymetry()
print(data.head())

# Make a Mercator map of the data using Cartopy
plt.figure(figsize=(7, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Bathymetry from Baja California")
# Plot the bathymetry as colored circles. Cartopy requires setting the projection of the
# original data through the transform argument. Use PlateCarree for geographic data.
plt.scatter(
    data.longitude,
    data.latitude,
    c=data.bathymetry_m,
    s=0.1,
    transform=ccrs.PlateCarree(),
)
plt.colorbar().set_label("meters")
# Use an utility function to add tick labels and land and ocean features to the map.
vd.datasets.setup_baja_bathymetry_map(ax)
plt.show()
