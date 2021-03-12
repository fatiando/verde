"""
.. _model_evaluation:

Evaluating Performance
======================

The Green's functions based interpolations in Verde are all linear regressions under the
hood. This means that we can use some of the same tactics from
:mod:`sklearn.model_selection` to evaluate our interpolator's performance. Once we have
a quantified measure of the quality of a given fitted gridder, we can use it to tune the
gridder's parameters, like ``damping`` for a :class:`~verde.Spline` (see
:ref:`model_selection`).

Verde provides adaptations of common scikit-learn tools to work better with spatial
data. Let's use these tools to evaluate the performance of a :class:`~verde.Spline` on
our sample air temperature data.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyproj
import verde as vd

data = vd.datasets.fetch_texas_wind()

# Use Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coords = projection(data.longitude.values, data.latitude.values)

region = vd.get_region((data.longitude, data.latitude))
# For this data, we'll generate a grid with 15 arc-minute spacing
spacing = 15 / 60

########################################################################################
# Splitting the data
# ------------------
#
# We can't evaluate a gridder on the data that went into fitting it. The true test of a
# model is if it can correctly predict data that it hasn't seen before. scikit-learn has
# the :func:`sklearn.model_selection.train_test_split` function to separate a dataset
# into two parts: one for fitting the model (called *training* data) and a separate one
# for evaluating the model (called *testing* data). Using it with spatial data would
# involve some tedious array conversions so Verde implements
# :func:`verde.train_test_split` which does the same thing but takes coordinates and
# data arrays instead.
#
# The split is done randomly so we specify a seed for the random number generator to
# guarantee that we'll get the same result every time we run this example. You probably
# don't want to do that for real data. We'll keep 30% of the data to use for testing
# (``test_size=0.3``).

train, test = vd.train_test_split(
    proj_coords, data.air_temperature_c, test_size=0.3, random_state=0
)

########################################################################################
# The returned ``train`` and ``test`` variables are tuples containing coordinates, data,
# and (optionally) weights arrays. Since we're not using weights, the third element of
# the tuple will be ``None``:
print(train)


########################################################################################
#
print(test)

########################################################################################
# Let's plot these two datasets with different colors:

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.set_title("Air temperature measurements for Texas")
ax.plot(train[0][0], train[0][1], ".r", label="train")
ax.plot(test[0][0], test[0][1], ".b", label="test")
ax.legend()
ax.set_aspect("equal")
plt.show()

########################################################################################
# We can pass the training dataset to the :meth:`~verde.base.BaseGridder.fit` method of
# most gridders using Python's argument expansion using the ``*`` symbol.

spline = vd.Spline()
spline.fit(*train)

########################################################################################
# Let's plot the gridded result to see what it looks like. First, we'll create a
# geographic grid:
grid = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="temperature",
)
print(grid)

########################################################################################
# Then, we'll mask out grid points that are too far from any given data point and plot
# the grid:
mask = vd.distance_mask(
    (data.longitude, data.latitude),
    maxdist=3 * spacing * 111e3,
    coordinates=vd.grid_coordinates(region, spacing=spacing),
    projection=projection,
)
grid = grid.where(mask)

plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_title("Gridded temperature")
pc = grid.temperature.plot.pcolormesh(
    ax=ax,
    cmap="plasma",
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    add_labels=False,
)
plt.colorbar(pc).set_label("C")
ax.plot(data.longitude, data.latitude, ".k", markersize=1, transform=ccrs.PlateCarree())
vd.datasets.setup_texas_wind_map(ax)
plt.show()

########################################################################################
# Scoring
# --------
#
# Gridders in Verde implement the :meth:`~verde.base.BaseGridder.score` method that
# calculates the `R² coefficient of determination
# <https://en.wikipedia.org/wiki/Coefficient_of_determination>`__
# for a given comparison dataset (``test`` in our case). The R² score is at most 1,
# meaning a perfect prediction, but has no lower bound.

score = spline.score(*test)
print("R² score:", score)

########################################################################################
# That's a good score meaning that our gridder is able to accurately predict data that
# wasn't used in the gridding algorithm.
#
# .. caution::
#
#     Once caveat for this score is that it is highly dependent on the particular split
#     that we made. Changing the random number generator seed in
#     :func:`verde.train_test_split` will result in a different score.

# Use 1 as a seed instead of 0
train_other, test_other = vd.train_test_split(
    proj_coords, data.air_temperature_c, test_size=0.3, random_state=1
)

print("R² score with seed 1:", vd.Spline().fit(*train_other).score(*test_other))

########################################################################################
# Cross-validation
# ----------------
#
# A more robust way of scoring the gridders is to use function
# :func:`verde.cross_val_score`, which (by default) uses a `k-fold cross-validation
# <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`__
# by default. It will split the data *k* times and return the score on each *fold*. We
# can then take a mean of these scores.

scores = vd.cross_val_score(vd.Spline(), proj_coords, data.air_temperature_c)
print("k-fold scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# You can also use most cross-validation splitter classes from
# :mod:`sklearn.model_selection` by specifying the ``cv`` argument. For example, if we
# want to shuffle then split the data *n* times
# (:class:`sklearn.model_selection.ShuffleSplit`):

from sklearn.model_selection import ShuffleSplit

shuffle = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

scores = vd.cross_val_score(
    vd.Spline(), proj_coords, data.air_temperature_c, cv=shuffle
)
print("shuffle scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# Parallel cross-validation
# -------------------------
#
# Cross-validation involves running several model fit and score operations
# which are independent of each other. Because of this, they are prime targets
# for parallelization. Verde uses the excellent `Dask <https://dask.org/>`__
# library for parallel execution.
#
# To run :func:`verde.cross_val_score` with Dask, use the ``delayed`` argument:

scores = vd.cross_val_score(
    vd.Spline(), proj_coords, data.air_temperature_c, delayed=True
)
print("Delayed k-fold scores:", scores)

########################################################################################
# In this case, the scores haven't actually been computed yet (hence the
# "delayed" term). Instead, Verde scheduled the operations with Dask. Since we
# are interested only in the mean score, we can schedule the mean as well using
# :func:`dask.delayed`:

import dask

mean_score = dask.delayed(np.mean)(scores)
print("Delayed mean:", mean_score)

########################################################################################
# To run the scheduled computations and get the mean score, use
# :func:`dask.compute` or ``.compute()``. Dask will automatically execute
# things in parallel.

mean_score = mean_score.compute()
print("Mean score:", mean_score)

########################################################################################
# .. note::
#
#     Dask will run many ``fit`` operations in parallel, which can be memory
#     intensive. Make sure you have enough RAM to run multiple fits.
#

########################################################################################
# Improving the score
# -------------------
#
# That score is not bad but it could be better. The default arguments for
# :class:`~verde.Spline` aren't optimal for this dataset. We could try
# different combinations manually until we get a good score. A better way is to
# do this automatically. In :ref:`model_selection` we'll go over how to do just
# that.
