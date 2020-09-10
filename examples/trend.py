"""
Polynomial trend
================

Verde offers the :class:`verde.Trend` class to fit a 2D polynomial trend to your data.
This can be useful for isolating a regional component of your data, for example, which
is a common operation for gravity and magnetic data. Let's look at how we can use Verde
to remove the clear trend from our Texas temperature dataset
(:func:`verde.datasets.fetch_texas_wind`).
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import verde as vd

# Load the Texas wind and temperature data as a pandas.DataFrame
data = vd.datasets.fetch_texas_wind()
print("Original data:")
print(data.head())

# Fit a 1st degree 2D polynomial to the data
coordinates = (data.longitude, data.latitude)
trend = vd.Trend(degree=1).fit(coordinates, data.air_temperature_c)
print("\nTrend estimator:", trend)

# Add the estimated trend and the residual data to the DataFrame
data["trend"] = trend.predict(coordinates)
data["residual"] = data.air_temperature_c - data.trend
print("\nUpdated DataFrame:")
print(data.head())


# Make a function to plot the data using the same colorbar
def plot_data(column, i, title):
    "Plot the column from the DataFrame in the ith subplot"
    crs = ccrs.PlateCarree()
    ax = plt.subplot(2, 2, i, projection=ccrs.Mercator())
    ax.set_title(title)
    # Set vmin and vmax to the extremes of the original data
    maxabs = vd.maxabs(data.air_temperature_c)
    mappable = ax.scatter(
        data.longitude,
        data.latitude,
        c=data[column],
        s=50,
        cmap="seismic",
        vmin=-maxabs,
        vmax=maxabs,
        transform=crs,
    )
    # Set the proper ticks for a Cartopy map
    vd.datasets.setup_texas_wind_map(ax)
    return mappable


plt.figure(figsize=(10, 9.5))

# Plot the data fields and capture the mappable returned by scatter to use for
# the colorbar
mappable = plot_data("air_temperature_c", 1, "Original data")
plot_data("trend", 2, "Regional trend")
plot_data("residual", 3, "Residual")

# Make histograms of the data and the residuals to show that the trend was
# removed
ax = plt.subplot(2, 2, 4)
ax.set_title("Distribution of data")
ax.hist(data.air_temperature_c, bins="auto", alpha=0.7, label="Original data")
ax.hist(data.residual, bins="auto", alpha=0.7, label="Residuals")
ax.legend()
ax.set_xlabel("Air temperature (C)")

# Add a single colorbar on top of the histogram plot where there is some space
cax = plt.axes((0.35, 0.44, 0.10, 0.01))
cb = plt.colorbar(
    mappable,
    cax=cax,
    orientation="horizontal",
)
cb.set_label("C")

plt.show()
