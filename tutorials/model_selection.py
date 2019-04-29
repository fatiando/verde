"""
.. _model_selection:

Model Selection
===============

The Green's functions based interpolations in Verde are all linear regressions under the
hood. This means that we can use some of the same tactics from
:mod:`sklearn.model_selection` to evaluate our interpolator's performance. Once we have
a quantified measure of the quality of a given fitted gridder, we can use it to tune the
gridder's parameters, like ``damping`` for a :class:`~verde.Spline`.

Verde provides adaptations of common scikit-learn tools to work better with spatial
data. Let's use these tools to evaluate and tune a :class:`~verde.Spline` to grid our
sample air temperature data.
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import itertools
import pyproj
import verde as vd

data = vd.datasets.fetch_texas_wind()

# Use Mercator projection because Spline is a Cartesian gridder
projection = pyproj.Proj(proj="merc", lat_ts=data.latitude.mean())
proj_coords = projection(data.longitude.values, data.latitude.values)

region = vd.get_region((data.longitude, data.latitude))
# The desired grid spacing in degrees (converted to meters using 1 degree approx. 111km)
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
# don't want to do that for real data. We'll keep 30% of the data to use for testing.

train, test = vd.train_test_split(
    proj_coords, data.air_temperature_c, test_size=0.3, random_state=0
)
print(train)
print(test)

plt.figure(figsize=(8, 6))
ax = plt.axes()
ax.set_title("Air temperature measurements for Texas")
ax.plot(train[0][0], train[0][1], ".r", label="train")
ax.plot(test[0][0], test[0][1], ".b", label="test")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

########################################################################################
# The returned ``train`` and ``test`` arguments are each tuples with the coordinates (in
# a tuple) and a data array. They are in a format that can be easily passed to the
# :meth:`~verde.base.BaseGridder.fit` method of most gridders using Python's argument
# expansion using the ``*`` symbol.

spline = vd.Spline()
spline.fit(*train)

########################################################################################
# Let's plot the gridded result to see what it looks like. We'll mask out grid points
# that are too far from any given data point.
mask = vd.distance_mask(
    (data.longitude, data.latitude),
    maxdist=3 * spacing * 111e3,
    coordinates=vd.grid_coordinates(region, spacing=spacing),
    projection=projection,
)
grid = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names=["temperature"],
).where(mask)

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
plt.tight_layout()
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
# Once caveat for this score is that it is highly dependent on the particular split that
# we made. Changing the random number generator seed in :func:`verde.train_test_split`
# will result in a different score.

# Use 1 as a seed instead of 0
train_other, test_other = vd.train_test_split(
    proj_coords, data.air_temperature_c, test_size=0.3, random_state=1
)
print("R² score with seed 1:", spline.fit(*train_other).score(*test_other))

########################################################################################
# A more robust way of scoring the gridders is to use function
# :func:`verde.cross_val_score`, which (by default) uses a `k-fold cross-validation
# <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation>`__.
# It will split the data *k* times and return the score on each *fold*. We can then take
# a mean of these scores.

scores = vd.cross_val_score(spline, proj_coords, data.air_temperature_c)
print("k-fold scores:", scores)
print("Mean score:", np.mean(scores))

########################################################################################
# That is not a very good score so clearly the default arguments for
# :class:`~verde.Spline` aren't suitable for this dataset. We could try different
# combinations manually until we get a good score. A better way is to do this
# automatically.

########################################################################################
# Tuning
# ------
#
# :class:`~verde.Spline` has many parameters that can be set to modify the final result.
# Mainly the ``damping`` regularization parameter and the ``mindist`` "fudge factor"
# which smooths the solution. Would changing the default values give us a better score?
#
# We can answer these questions by changing the values in our ``spline`` and
# re-evaluating the model score repeatedly for different values of these parameters.
# Let's test the following combinations:

dampings = [None, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
mindists = [5e3, 10e3, 25e3, 50e3, 75e3, 100e3]

# Use itertools to create a list with all combinations of parameters to test
parameter_sets = [
    dict(damping=combo[0], mindist=combo[1])
    for combo in itertools.product(dampings, mindists)
]
print("Number of combinations:", len(parameter_sets))
print("Combinations:", parameter_sets)

########################################################################################
# Now we can loop over the combinations and collect the scores for each parameter set.

scores = []
for params in parameter_sets:
    spline.set_params(**params)
    score = np.mean(vd.cross_val_score(spline, proj_coords, data.air_temperature_c))
    scores.append(score)
print(scores)

########################################################################################
# The largest score will yield the best parameter combination.

best = np.argmax(scores)
print("Best score:", scores[best])
print("Best parameters:", parameter_sets[best])

########################################################################################
# That is a big improvement over our previous score!
#
# We can now configure our spline with the best configuration and re-fit.

spline.set_params(**parameter_sets[best])
spline.fit(proj_coords, data.air_temperature_c)

########################################################################################
# Finally, we can make a grid with the best configuration to see how it compares to our
# previous result.

grid_best = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names=["temperature"],
).where(mask)

plt.figure(figsize=(14, 8))
for i, title, grd in zip(range(2), ["Defaults", "Tuned"], [grid, grid_best]):
    ax = plt.subplot(1, 2, i + 1, projection=ccrs.Mercator())
    ax.set_title(title)
    pc = grd.temperature.plot.pcolormesh(
        ax=ax,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        vmin=data.air_temperature_c.min(),
        vmax=data.air_temperature_c.max(),
        add_colorbar=False,
        add_labels=False,
    )
    plt.colorbar(pc, orientation="horizontal", aspect=50, pad=0.05).set_label("C")
    ax.plot(
        data.longitude, data.latitude, ".k", markersize=1, transform=ccrs.PlateCarree()
    )
    vd.datasets.setup_texas_wind_map(ax)
plt.tight_layout()
plt.show()

########################################################################################
# Notice that, for sparse data like these, smoother models tend to be better predictors.
# This is a sign that you should probably not trust many of the short wavelength
# features that we get from the defaults.
