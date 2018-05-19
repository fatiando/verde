"""
Polynomial trend
================

Verde offers the :class:`verde.Trend` class to fit a 2D polynomial trend to
your data. This can be useful for isolating a regional component of your data,
for example, which is a common operation for gravity and magnetic data. Let's
look at how we can use Verde to remove the clear positive trend from the Rio de
Janeiro magnetic anomaly data.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import verde as vd

# Load the Rio de Janeiro total field magnetic anomaly data as a
# pandas.DataFrame
data = vd.datasets.fetch_rio_magnetic_anomaly()
print("Original data:")
print(data.head())

# Fit a 2nd degree 2D polynomial to the anomaly data
trend = vd.Trend(degree=2).fit(data.longitude, data.latitude,
                               data.total_field_anomaly_nt)
print("\nTrend estimator:", trend)

# Add the estimated trend and the residual data to the DataFrame
data['trend'] = trend.predict(data.longitude, data.latitude)
data['residual'] = trend.residual_
print("\nUpdated DataFrame:")
print(data.head())


# Make a function to plot the data using the same colorbar
def plot_data(column, i, title):
    "Plot the column from the DataFrame in the ith subplot"
    crs = ccrs.PlateCarree()
    ax = plt.subplot(2, 2, i, projection=ccrs.Mercator())
    ax.set_title(title)
    # Set vmin and vmax to the extremes of the original data
    maxabs = np.max(np.abs([data.total_field_anomaly_nt.min(),
                            data.total_field_anomaly_nt.max()]))
    mappable = ax.scatter(data.longitude, data.latitude, c=data[column], s=1,
                          cmap='seismic', vmin=-maxabs, vmax=maxabs,
                          transform=crs)
    # Set the proper ticks for a Cartopy map
    ax.set_xticks(np.arange(-42.5, -42, 0.1), crs=crs)
    ax.set_yticks(np.arange(-22.4, -22, 0.1), crs=crs)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # Set the plot region to be tight around the data
    ax.set_extent(vd.get_region(data.longitude, data.latitude))
    return mappable


plt.figure(figsize=(9, 8))

# Plot the data fields and capture the mappable returned by scatter to use for
# the colorbar
mappable = plot_data('total_field_anomaly_nt', 1, 'Original magnetic anomaly')
plot_data('trend', 2, 'Regional trend')
plot_data('residual', 3, 'Residual')

# Make histograms of the data and the residuals to show that the trend was
# removed
ax = plt.subplot(2, 2, 4)
ax.set_title('Distribution of data')
ax.hist(data.total_field_anomaly_nt, bins='auto', alpha=0.7,
        label='Original data')
ax.hist(data.residual, bins='auto', alpha=0.7, label='Residuals')
ax.legend()
ax.set_xlabel('Total field anomaly (nT)')

# Add a single colorbar on top of the histogram plot where there is some space
cax = plt.axes((0.58, 0.44, 0.18, 0.015))
cb = plt.colorbar(mappable, cax=cax, orientation='horizontal',
                  ticks=np.arange(-800, 801, 400))
cb.set_label('nT')

plt.tight_layout()
plt.show()
