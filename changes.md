**DEPRECATIONS** (the following features are deprecated and will be removed in Verde v2.0.0):

* Functions and the associated sample dataset `verde.datasets.fetch_rio_magnetic` and `verde.datasets.setup_rio_magnetic_map` are deprecated. Please use another dataset instead. (#213)
* Class `verde.VectorSpline2D` is deprecated. The class is specific for GPS/GNSS data and doesn't fit the general-purpose nature of Verde. The implementation will be moved to the [Erizo](https://github.com/fatiando/erizo) package instead. (#214)
* The `client` keyword argument for `verde.cross_val_score` and `verde.SplineCV` is deprecated in favor of the new `delayed` argument (see below). (#222)

New features:

* Use the `dask.delayed` interface for parallelism in cross-validation instead of the futures interface (`dask.distributed.Client`). It's easier and allows building the entire graph lazily before executing. To use the new feature, pass `delayed=True` to `verde.cross_val_score` and `verde.SplineCV`. The argument `client` in both of these is deprecated (see above). (#222)
* Expose the optimal spline in `verde.SplineCV.spline_`. This is the fitted `verde.Spline` object using the optimal parameters. (#219)
* New option `drop_coords` to allow `verde.BlockReduce` and `verde.BlockMean` to reduce extra elements in `coordinates` (basically, treat them as data). Default to `True` to maintain backwards compatibility. If `False`, will no longer drop coordinates after the second one but will apply the reduction in blocks to them as well. The reduced coordinates are returned in the same order in the `coordinates`. (#198)

Improvements:

* Use the default system cache location to store the sample data instead of `~/.verde/data`. This is so users can more easily clean up unused files. Because this is system specific, function `verde.datasets.locate` was added to return the cache folder location. (#220)

Bug fixes:

* Correctly use `parallel=True` and `numba.prange` in the numba compiled functions. Using it on the Green's function was raising a warning because there is nothing to parallelize. (#221)

Maintenance:

* Add testing and support for Python 3.8. (#211)

Documentation:

* Fix a typo in the JOSS paper Bibtex entry. (#215)
* Wrap docstrings to 79 characters for better integration with Jupyter and IPython. These systems display docstrings using 80 character windows, causing our larger lines to wrap around and become almost illegible. (#212)
* Use napoleon instead of numpydoc to format docstrings. Results is slightly different layout in the website documentation. (#209)
* Update contact information to point to the Slack chat instead of Gitter. (#204)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda

