"""
.. _model_selection:

Model Selection
===============

In :ref:`model_evaluation`, we saw how to check the performance of an interpolator using
cross-validation. We found that the default parameters for :class:`verde.Spline` are not
good for predicting our sample air temperature data. Now, let's see how we can tune the
:class:`~verde.Spline` to improve the cross-validation performance.

Once again, we'll start by importing the required packages and loading our sample data.
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
# Before we begin tuning, let's reiterate what the results were with the default
# parameters.

spline_default = vd.Spline()
score_default = np.mean(
    vd.cross_val_score(spline_default, proj_coords, data.air_temperature_c)
)
spline_default.fit(proj_coords, data.air_temperature_c)
print("R² with defaults:", score_default)


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

dampings = [None, 1e-4, 1e-3, 1e-2]
mindists = [5e3, 10e3, 50e3, 100e3]

# Use itertools to create a list with all combinations of parameters to test
parameter_sets = [
    dict(damping=combo[0], mindist=combo[1])
    for combo in itertools.product(dampings, mindists)
]
print("Number of combinations:", len(parameter_sets))
print("Combinations:", parameter_sets)

########################################################################################
# Now we can loop over the combinations and collect the scores for each parameter set.

spline = vd.Spline()
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
print("Score with defaults:", score_default)
print("Best parameters:", parameter_sets[best])

########################################################################################
# **That is a nice improvement over our previous score!**
#
# This type of tuning is important and should always be performed when using a new
# gridder or a new dataset. However, the above implementation requires a lot of
# coding. Fortunately, Verde provides convenience classes that perform the
# cross-validation and tuning automatically when fitting a dataset.


########################################################################################
# Cross-validated gridders
# ------------------------
#
# The :class:`verde.SplineCV` class provides a cross-validated version of
# :class:`verde.Spline`. It has almost the same interface but does all of the above
# automatically when fitting a dataset. The only difference is that you must provide a
# list of ``damping`` and ``mindist`` parameters to try instead of only a single value:

spline = vd.SplineCV(
    dampings=dampings,
    mindists=mindists,
)

########################################################################################
# Calling :meth:`~verde.SplineCV.fit` will run a grid search over all parameter
# combinations to find the one that maximizes the cross-validation score.

spline.fit(proj_coords, data.air_temperature_c)

########################################################################################
# The estimated best damping and mindist, as well as the cross-validation
# scores, are stored in class attributes:

print("Highest score:", spline.scores_.max())
print("Best damping:", spline.damping_)
print("Best mindist:", spline.mindist_)

########################################################################################
# The cross-validated gridder can be used like any other gridder (including in
# :class:`verde.Chain` and :class:`verde.Vector`):

grid = spline.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="temperature",
)
print(grid)

########################################################################################
# Like :func:`verde.cross_val_score`, :class:`~verde.SplineCV` can also run the
# grid search in parallel using `Dask <https://dask.org/>`__ by specifying the
# ``delayed`` attribute:

spline = vd.SplineCV(dampings=dampings, mindists=mindists, delayed=True)

########################################################################################
# Unlike :func:`verde.cross_val_score`, calling :meth:`~verde.SplineCV.fit`
# does **not** result in :func:`dask.delayed` objects. The full grid search is
# executed and the optimal parameters are found immediately.

spline.fit(proj_coords, data.air_temperature_c)

print("Best damping:", spline.damping_)
print("Best mindist:", spline.mindist_)

########################################################################################
# The one caveat is the that the ``scores_`` attribute will be a list of
# :func:`dask.delayed` objects instead because the scores are only computed as
# intermediate values in the scheduled computations.

print("Delayed scores:", spline.scores_)

########################################################################################
# Calling :func:`dask.compute` on the scores will calculate their values but
# will unfortunately run the entire grid search again. So using
# ``delayed=True`` is not recommended if you need the scores of each parameter
# combination.

########################################################################################
# The importance of tuning
# ------------------------
#
# To see the difference that tuning has on the results, we can make a grid
# with the best configuration and see how it compares to the default result.

grid_default = spline_default.grid(
    region=region,
    spacing=spacing,
    projection=projection,
    dims=["latitude", "longitude"],
    data_names="temperature",
)

########################################################################################
# Let's plot our grids side-by-side:

mask = vd.distance_mask(
    (data.longitude, data.latitude),
    maxdist=3 * spacing * 111e3,
    coordinates=vd.grid_coordinates(region, spacing=spacing),
    projection=projection,
)

grid = grid.where(mask)
grid_default = grid_default.where(mask)

plt.figure(figsize=(14, 8))
for i, title, grd in zip(range(2), ["Defaults", "Tuned"], [grid_default, grid]):
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
plt.show()

########################################################################################
# Notice that, for sparse data like these, **smoother models tend to be better
# predictors**. This is a sign that you should probably not trust many of the short
# wavelength features that we get from the defaults.
