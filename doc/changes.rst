.. _changes:

Changelog
=========

Version 1.8.1
-------------

Released on: 2023/06/12

DOI: https://doi.org/10.5281/zenodo.10964877

Breaking changes:

* Drop support for Python 3.7 and 3.8 (`#418 <https://github.com/fatiando/verde/pull/418>`__).

Bug fixes:

* Fix pandas warnings about ``.ravel()`` (`#449 <https://github.com/fatiando/verde/pull/449>`__)
* Fix dropping non-dimensional coordinates in ``grid_to_table`` (`#441 <https://github.com/fatiando/verde/pull/441>`__)

Maintenance:

* Remove dependence on ``pkg_resources`` (`#448 <https://github.com/fatiando/verde/pull/448>`__)
* Extend support for Python 3.12 (`#442 <https://github.com/fatiando/verde/pull/442>`__)
* Use Trusted Publishers for PyPI deployment (`#436 <https://github.com/fatiando/verde/pull/436>`__)
* Use Burocrata to check/add license notices (`#435 <https://github.com/fatiando/verde/pull/435>`__)
* Add Dependabot config to update GitHub Actions (`#421 <https://github.com/fatiando/verde/pull/421>`__)

Documentation:

* Replace Sphinx napoleon for numpydoc (`#450 <https://github.com/fatiando/verde/pull/450>`__)
* Refactor the Overview introductory tutorial (`#429 <https://github.com/fatiando/verde/pull/429>`__)
* Update the versions of documentation tools (`#419 <https://github.com/fatiando/verde/pull/419>`__)

This release contains contributions from:

* Leonardo Uieda
* Santiago Soler
* Gelson Ferreira Souza Junior


Version 1.8.0
-------------

Released on: 2023/05/08

DOI: https://doi.org/10.5281/zenodo.7907182

Deprecations:

* Deprecate the ``engine`` argument of ``Spline/SplineCV`` (`#373 <https://github.com/fatiando/verde/pull/373>`__)
* Deprecate the ``engine`` argument in ``VectorSpline2D`` (`#410 <https://github.com/fatiando/verde/pull/410>`__)
* Deprecate ``verde.ScipyGridder`` in favor of the new ``Linear/Cubic/KNeighbors`` (`#393 <https://github.com/fatiando/verde/pull/393>`__)
* Deprecate the ``scatter`` method of all interpolators (`#357 <https://github.com/fatiando/verde/pull/357>`__)
* Undo deprecation of region/spacing/shape in the ``grid`` method (`#394 <https://github.com/fatiando/verde/pull/394>`__)
* Undo deprecation of ``verde.VectorSpline2D`` (`#385 <https://github.com/fatiando/verde/pull/385>`__)

New features:

* New interpolator ``verde.KNeighbors`` class for nearest neighbor interpolation (`#378 <https://github.com/fatiando/verde/pull/378>`__)
* New interpolator ``verde.Cubic`` gridder class based on SciPy (`#374 <https://github.com/fatiando/verde/pull/374>`__)
* New interpolator ``verde.Linear`` gridder class based on SciPy (`#372 <https://github.com/fatiando/verde/pull/372>`__)
* New function ``verde.line_coordinates``, a 1D version of ``verde.grid_coordinates`` (`#390 <https://github.com/fatiando/verde/pull/390>`__)
* New ``scoring`` parameter for ``verde.SplineCV`` to specify the scoring function (`#380 <https://github.com/fatiando/verde/pull/380>`__)

Improvements:

* Remove the need for ``mindist`` in ``verde.Spline`` by using a better Green's function for small distances (`#401 <https://github.com/fatiando/verde/pull/401>`__)
* Fix behavior of coordinate generation in ``verde.line_coordinates`` if ``spacing >= 2 * interval`` (`#406 <https://github.com/fatiando/verde/pull/406>`__)
* Use the new classes ``Linear/Cubic/KNeighbors`` in ``verde.project_grid`` (`#395 <https://github.com/fatiando/verde/pull/395>`__)
* Default to not rescaling coordinates in ``Linear/Cubic`` (`#391 <https://github.com/fatiando/verde/pull/391>`__)
* Add option to return 1D arrays in ``grid_coordinates`` (`#388 <https://github.com/fatiando/verde/pull/388>`__)

Documentation:

* New logo and use sphinx-design in the docs (`#367 <https://github.com/fatiando/verde/pull/367>`__)
* Move deprecated APIs to their own docs section (`#400 <https://github.com/fatiando/verde/pull/400>`__)
* Improve docstring of ``verde.make_xarray_grid`` (`#399 <https://github.com/fatiando/verde/pull/399>`__)
* Fix typo in the ``verde.base.BaseGridder.fit`` docstring (`#397 <https://github.com/fatiando/verde/pull/397>`__)
* Add missing matplotlib scraper for sphinx-gallery (`#389 <https://github.com/fatiando/verde/pull/389>`__)
* Use PyGMT instead of Cartopy in the ``verde.Chain`` tutorial (`#386 <https://github.com/fatiando/verde/pull/386>`__)
* Use Markdown for the README instead of RST (`#366 <https://github.com/fatiando/verde/pull/366>`__)

Maintenance:

* Drop support for Python 3.6 (`#364 <https://github.com/fatiando/verde/pull/364>`__)
* Fail CI if codecov upload fails (`#409 <https://github.com/fatiando/verde/pull/409>`__)
* More informative warning messages by setting ``stacklevel=2`` (`#407 <https://github.com/fatiando/verde/pull/407>`__)
* Remove the deprecated sample data gallery (`#387 <https://github.com/fatiando/verde/pull/387>`__)
* Set lower bounds for dependencies based on NEP29 (`#384 <https://github.com/fatiando/verde/pull/384>`__)
* Replace ``setup.py`` with PyPA "build" (`#371 <https://github.com/fatiando/verde/pull/371>`__)
* Replace deprecated numpy dtypes ``np.bool`` and ``np.int`` (`#362 <https://github.com/fatiando/verde/pull/362>`__)

This release contains contributions from:

* Sarah Margrethe Askevold
* James Sample
* Santiago Soler
* Matt Tankersley
* Leonardo Uieda

Version 1.7.0
-------------

Released on: 2022/03/25

DOI: https://doi.org/10.5281/zenodo.6384887

.. warning::

    **Verde v1.7.0 is the last release that is compatible with Python 3.6.**

Deprecation:

* Move the ``CheckerBoard`` class to ``verde.synthetic`` (`#353 <https://github.com/fatiando/verde/pull/353>`__)
* Deprecate the ``verde.test`` function which will be removed in v2.0.0 (`#344 <https://github.com/fatiando/verde/pull/344>`__)
* Deprecate the ``datasets`` module, which will be replaced by `Ensaio <https://github.com/fatiando/ensaio>`__ in the future (`#277 <https://github.com/fatiando/verde/pull/277>`__)
* Warn that the default score will change from R² to negative RMSE in v2.0 (`#352 <https://github.com/fatiando/verde/pull/352>`__)

New features:

* Add option to pass coordinates to the ``grid`` method instead of just ``region`` and ``spacing`` (`#326 <https://github.com/fatiando/verde/pull/326>`__)
* Add support for Python 3.9 (`#323 <https://github.com/fatiando/verde/pull/323>`__) and 3.10 (`#346 <https://github.com/fatiando/verde/pull/346>`__)

Documentation:

* Modernize the front page of the docs (`#356 <https://github.com/fatiando/verde/pull/356>`__)
* Modernize the Installing page (`#355 <https://github.com/fatiando/verde/pull/355>`__)
* Update the contact link in the docs (`#347 <https://github.com/fatiando/verde/pull/347>`__)
* Switch the docs theme to the sphinx-book-theme (`#343 <https://github.com/fatiando/verde/pull/343>`__)
* Update ``dims`` in example of ``make_xarray_grid`` (`#329 <https://github.com/fatiando/verde/pull/329>`__)
* Explicitly pass default arguments with their corresponding keywords on tests and examples (`#327 <https://github.com/fatiando/verde/pull/327>`__)

Maintenance:

* Replace Google Analytics for Plausible one to make our docs more privacy-friendly (`#358 <https://github.com/fatiando/verde/pull/358>`__)
* Move configuration from ``setup.py`` to ``setup.cfg`` (`#348 <https://github.com/fatiando/verde/pull/348>`__)
* Link CoC, Authorship, Contributing, and Maintainers guides back to the Fatiando-wide pages (`#338 <https://github.com/fatiando/verde/pull/338>`__)
* Replace pylint with more flake8 plugins (`#337 <https://github.com/fatiando/verde/pull/337>`__)
* Rename the main branch from "master" to "main" (`#335 <https://github.com/fatiando/verde/pull/335>`__)
* Remove ``normalize`` argument when creating scikit-learn solvers (`#333 <https://github.com/fatiando/verde/pull/333>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda

Version 1.6.1
-------------

Released on: 2021/03/22

DOI: https://doi.org/10.5281/zenodo.4626786

Minor changes:

* Allow ``make_xarray_grid`` to receive ``data=None`` instead of raising an error. This is used to create an empty ``xarray.Dataset`` (`#318 <https://github.com/fatiando/verde/pull/318>`__)

Maintenance:

* Fix use of wrong version numbers for PyPI releases (`#317 <https://github.com/fatiando/verde/pull/317>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda

Version 1.6.0
-------------

Released on: 2021/03/18

DOI: https://doi.org/10.5281/zenodo.4617252

New features:

* Allow specifing the scoring function in ``cross_val_score`` instead of always using the ``.score`` method of the gridder (`#273 <https://github.com/fatiando/verde/pull/273>`__)
* New function ``verde.make_xarray_grid`` to simplify the creation of ``xarray.Dataset`` from individual numpy arrays that represent a 2D grid (`#282 <https://github.com/fatiando/verde/pull/282>`__ and `#300 <https://github.com/fatiando/verde/pull/300>`__)

Enhancements:

* Raise informative errors for invalid ``verde.rolling_window`` arguments, like missing ``spacing`` or ``shape`` and invalid window sizes (`#280 <https://github.com/fatiando/verde/pull/280>`__)
* Replace ``DeprecationWarning`` with ``FutureWarning`` since these are intended for end-users, which allows us to avoid having to set ``warning.simplefilter`` (`#305 <https://github.com/fatiando/verde/pull/305>`__ and `#293 <https://github.com/fatiando/verde/pull/293>`__)

Documentation:

* Several typo fixes (`#306 <https://github.com/fatiando/verde/pull/306>`__ `#303 <https://github.com/fatiando/verde/pull/303>`__ `#281 <https://github.com/fatiando/verde/pull/281>`__)
* Update link to the GMT website in the Baja bathymetry example (`#298 <https://github.com/fatiando/verde/pull/298>`__)
* Fix issue with Cartopy 0.17 and require versions >= 0.18 for building the docs (`#283 <https://github.com/fatiando/verde/pull/283>`__)

Maintenance:

* Refactor internal function ``get_data_names`` and related check functions to simplify their logic and make them more useful (`#295 <https://github.com/fatiando/verde/pull/295>`__)
* Require Black >=20.8b1 (`#284 <https://github.com/fatiando/verde/pull/284>`__)
* Format the ``doc/conf.py`` sphinx configuration file with Black (`#275 <https://github.com/fatiando/verde/pull/275>`__)
* Add a license and copyright notice to every source file (`#308 <https://github.com/fatiando/verde/pull/308>`__)
* Replace versioneer for setuptools-scm (`#307 <https://github.com/fatiando/verde/pull/307>`__)
* Replace Travis and Azure with GitHub Actions (`#309 <https://github.com/fatiando/verde/pull/309>`__)
* Exclude Dask 2021.03.0 as a dependency. This release was causing the tests to fail under Python 3.8 on every OS. The problem seems to be originated in ``dask.distributed`` (`#311 <https://github.com/fatiando/verde/pull/311>`__)
* Use the OSI version of item 3 in the license (`#299 <https://github.com/fatiando/verde/pull/299>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda
* Federico Esteban
* DC Slagel

Version 1.5.0
-------------

Released on: 2020/06/04

DOI: https://doi.org/10.5281/zenodo.3877060

Bug fixes:

* Apply projections using only the first two coordinates instead all given
  coordinates. Projections only really involve the first two (horizontal)
  coordinates. Only affects users passing ``extra_coords`` to gridder methods.
  (`#264 <https://github.com/fatiando/verde/pull/264>`__)

New features:

* **New** blocked cross-validation classes ``BlockShuffleSplit`` and
  ``BlockKFold``. These are scikit-learn compatible cross-validators that split
  the data into spatial blocks before assigning them to folds. Blocked
  cross-validation can help avoid overestimation of prediction accuracy for
  spatial data (see [Roberts_etal2017]_). The classes work with
  ``verde.cross_val_score`` and any other function/method/class that accepts a
  scikit-learn cross-validator.
  (`#251 <https://github.com/fatiando/verde/pull/251>`__ and
  `#254 <https://github.com/fatiando/verde/pull/254>`__)
* Add the option for block-wise splitting in ``verde.train_test_split`` by
  passing in a ``spacing`` or ``shape`` parameters.
  (`#253 <https://github.com/fatiando/verde/pull/253>`__ and
  `#257 <https://github.com/fatiando/verde/pull/257>`__)

Base classes:

* Add optional argument to ``verde.base.least_squares`` to copy Jacobian
  matrix.
  (`#255 <https://github.com/fatiando/verde/pull/255>`__)
* Add extra coordinates (specified by the ``extra_coords`` keyword argument
  to outputs of ``BaseGridder`` methods.
  (`#265 <https://github.com/fatiando/verde/pull/265>`__)

Maintenance:

* Update tests to ``repr`` changes in scikit-learn 0.23.0.
  (`#267 <https://github.com/fatiando/verde/pull/267>`__)

Documentation:

* Fix typo in README contributing section.
  (`#258 <https://github.com/fatiando/verde/pull/258>`__)

This release contains contributions from:

* Leonardo Uieda
* Santiago Soler
* Rowan Cockett

Version 1.4.0
-------------

Released on: 2020/04/06

DOI: https://doi.org/10.5281/zenodo.3739449

Bug fixes:

* **Profile distances are now returned in projected (Cartesian) coordinates by
  the** ``profile`` **method of gridders if a projection is given.** The method
  has the option to apply a projection to the coordinates before predicting so
  we can pass geographic coordinates to Cartesian gridders. In these cases, the
  distance along the profile is calculated by the ``profile_coordinates``
  function with the unprojected coordinates (in the geographic case it would be
  degrees). The profile point calculation is also done assuming that
  coordinates are Cartesian, which is clearly wrong if inputs are longitude and
  latitude. To fix this, we now project the input points prior to passing them
  to ``profile_coordinates``. This means that the distances are Cartesian and
  generation of profile points is also Cartesian (as is assumed by the
  function). The generated coordinates are projected back so that the user gets
  longitude and latitude but distances are still projected Cartesian meters.
  (`#231 <https://github.com/fatiando/verde/pull/231>`__)
* **Function** ``verde.grid_to_table`` **now sets the correct order for
  coordinates.** We were relying on the order of the ``coords`` attribute of
  the ``xarray.Dataset`` for the order of the coordinates. This is wrong
  because xarray takes the coordinate order from the ``dims`` attribute
  instead, which is what we should also have been doing.
  (`#229 <https://github.com/fatiando/verde/pull/229>`__)

Documentation:

* Generalize coordinate system specifications in ``verde.base.BaseGridder``
  docstrings. Most methods don't really depend on the coordinate system so use
  a more generic language to allow derived classes to specify their coordinate
  systems without having to overload the base methods just to rewrite the
  docstrings.
  (`#240 <https://github.com/fatiando/verde/pull/240>`__)

New features:

* New function ``verde.convexhull_mask`` to mask points in a grid that fall
  outside the convex hull defined by data points.
  (`#237 <https://github.com/fatiando/verde/pull/237>`__)
* New function ``verde.project_grid`` that transforms 2D gridded data using a
  given projection. It re-samples the data using ``ScipyGridder`` (by default)
  and runs a blocked mean (optional) to avoid aliasing when the points aren't
  evenly distributed in the projected coordinates (like in polar projections).
  Finally, it applies a ``convexhull_mask`` to the grid to avoid extrapolation
  to points that had no original data.
  (`#246 <https://github.com/fatiando/verde/pull/246>`__)
* New function ``verde.expanding_window`` for selecting data that falls inside
  of an expanding window around a central point.
  (`#238 <https://github.com/fatiando/verde/pull/238>`__)
* New function ``verde.rolling_window`` for rolling window selections of
  irregularly sampled data.
  (`#236 <https://github.com/fatiando/verde/pull/236>`__)

Improvements:

* Allow ``verde.grid_to_table`` to take ``xarray.DataArray`` as input.
  (`#235 <https://github.com/fatiando/verde/pull/235>`__)

Maintenance:

* Use newer MacOS images on Azure Pipelines.
  (`#234 <https://github.com/fatiando/verde/pull/234>`__)

This release contains contributions from:

* Leonardo Uieda
* Santiago Soler
* Jesse Pisel

Version 1.3.0
-------------

Released on: 2020/01/22

DOI: https://doi.org/10.5281/zenodo.3620851

**DEPRECATIONS** (the following features are deprecated and will be removed in
Verde v2.0.0):

* Functions and the associated sample dataset
  ``verde.datasets.fetch_rio_magnetic`` and
  ``verde.datasets.setup_rio_magnetic_map`` are deprecated. Please use another
  dataset instead.
  (`#213 <https://github.com/fatiando/verde/pull/213>`__)
* Class ``verde.VectorSpline2D`` is deprecated. The class is specific for
  GPS/GNSS data and doesn't fit the general-purpose nature of Verde. The
  implementation will be moved to the `Erizo
  <https://github.com/fatiando/erizo>`__ package instead.
  (`#214 <https://github.com/fatiando/verde/pull/214>`__)
* The ``client`` keyword argument for ``verde.cross_val_score`` and
  ``verde.SplineCV`` is deprecated in favor of the new ``delayed`` argument
  (see below).
  (`#222 <https://github.com/fatiando/verde/pull/222>`__)

New features:

* Use the ``dask.delayed`` interface for parallelism in cross-validation
  instead of the futures interface (``dask.distributed.Client``). It's easier
  and allows building the entire graph lazily before executing. To use the new
  feature, pass ``delayed=True`` to ``verde.cross_val_score`` and
  ``verde.SplineCV``. The argument ``client`` in both of these is deprecated
  (see above).
  (`#222 <https://github.com/fatiando/verde/pull/222>`__)
* Expose the optimal spline in ``verde.SplineCV.spline_``. This is the fitted
  ``verde.Spline`` object using the optimal parameters.
  (`#219 <https://github.com/fatiando/verde/pull/219>`__)
* New option ``drop_coords`` to allow ``verde.BlockReduce`` and
  ``verde.BlockMean`` to reduce extra elements in ``coordinates`` (basically,
  treat them as data). Default to ``True`` to maintain backwards compatibility.
  If ``False``, will no longer drop coordinates after the second one but will
  apply the reduction in blocks to them as well. The reduced coordinates are
  returned in the same order in the ``coordinates``.
  (`#198 <https://github.com/fatiando/verde/pull/198>`__)

Improvements:

* Use the default system cache location to store the sample data instead of
  ``~/.verde/data``. This is so users can more easily clean up unused files.
  Because this is system specific, function ``verde.datasets.locate`` was added
  to return the cache folder location.
  (`#220 <https://github.com/fatiando/verde/pull/220>`__)

Bug fixes:

* Correctly use ``parallel=True`` and ``numba.prange`` in the numba compiled
  functions. Using it on the Green's function was raising a warning because
  there is nothing to parallelize.
  (`#221 <https://github.com/fatiando/verde/pull/221>`__)

Maintenance:

* Add testing and support for Python 3.8.
  (`#211 <https://github.com/fatiando/verde/pull/211>`__)

Documentation:

* Fix a typo in the JOSS paper Bibtex entry.
  (`#215 <https://github.com/fatiando/verde/pull/215>`__)
* Wrap docstrings to 79 characters for better integration with Jupyter and
  IPython. These systems display docstrings using 80 character windows, causing
  our larger lines to wrap around and become almost illegible.
  (`#212 <https://github.com/fatiando/verde/pull/212>`__)
* Use napoleon instead of numpydoc to format docstrings. Results is slightly
  different layout in the website documentation.
  (`#209 <https://github.com/fatiando/verde/pull/209>`__)
* Update contact information to point to the Slack chat instead of Gitter.
  (`#204 <https://github.com/fatiando/verde/pull/204>`__)

This release contains contributions from:

* Santiago Soler
* Leonardo Uieda


Version 1.2.0
-------------

Released on: 2019/07/23

DOI: https://doi.org/10.5281/zenodo.3347076

Bug fixes:

* Return the correct coordinates when passing ``pixel_register=True`` and ``shape`` to
  ``verde.grid_coordinates``. The returned coordinates had 1 too few elements in each
  dimension (and the wrong values). This is because we generate grid-line registered
  points first and then shift them to the center of the pixels and drop the last point.
  This only works when specifying ``spacing`` because it will generate the right amount
  of points. When ``shape`` is given, we need to first convert it to "grid-line" shape
  (with 1 extra point per dimension) before generating coordinates.
  (`#183 <https://github.com/fatiando/verde/pull/183>`__)
* Reset force coordinates when refitting splines. Previously, the splines set the force
  coordinates from the data coordinates only the first time ``fit`` was called. This
  means that when fitting on different data, the spline would still use the old
  coordinates leading to a poor prediction score. Now, the spline will use the
  coordinates of the current data passed to ``fit``. This only affects cases where
  ``force_coords=None``. It's a slight change and only affects some of the scores for
  cross-validation. (`#191 <https://github.com/fatiando/verde/pull/191>`__)

New functions/classes:

* New class ``verde.SplineCV``: a cross-validated version of ``Spline`` . that performs
  grid search cross-validation to automatically tune the parameters of a ``Spline``.
  (`#185 <https://github.com/fatiando/verde/pull/185>`__)
* New function ``verde.longitude_continuity`` to format longitudes to a continuous
  range so that they can be indexed with ``verde.inside``
  (`#181 <https://github.com/fatiando/verde/pull/181>`__)
* New function ``verde.load_surfer`` to load grid data from a Surfer ASCII file (a
  contouring, griding and surface mapping software from GoldenSoftware).
  (`#169 <https://github.com/fatiando/verde/pull/169>`__)
* New function ``verde.median_distance`` that calculates the median near neighbor
  distance between each point in the given dataset.
  (`#163 <https://github.com/fatiando/verde/pull/163>`__)

Improvements:

* Allow ``verde.block_split`` and ``verde.BlockReduce`` to take a ``shape`` argument
  instead of ``spacing``. Useful when the size of the block is less meaningful than the
  number of blocks.
  (`#184 <https://github.com/fatiando/verde/pull/184>`__)
* Allow zero degree polynomials in ``verde.Trend``, which represents a mean value.
  (`#162 <https://github.com/fatiando/verde/pull/162>`__)
* Function ``verde.cross_val_score`` returns a numpy array instead of a list for easier
  computations on the results. (`#160 <https://github.com/fatiando/verde/pull/160>`__)
* Function ``verde.maxabs`` now handles inputs with NaNs automatically.
  (`#158 <https://github.com/fatiando/verde/pull/158>`__)

Documentation:

* New tutorial to explain the intricacies of grid coordinates generation, adjusting
  spacing vs region, pixel registration, etc.
  (`#192 <https://github.com/fatiando/verde/pull/192>`__)

Maintenance:

* Drop support for Python 3.5. (`#178 <https://github.com/fatiando/verde/pull/178>`__)
* Add support for Python 3.7. (`#150 <https://github.com/fatiando/verde/pull/150>`__)
* More functions are now part of the base API: ``n_1d_arrays``, ``check_fit_input`` and
  ``least_squares`` are now included in ``verde.base``.
  (`#156 <https://github.com/fatiando/verde/pull/156>`__)

This release contains contributions from:

* Goto15
* Lindsey Heagy
* Jesse Pisel
* Santiago Soler
* Leonardo Uieda


Version 1.1.0
-------------

Released on: 2018/11/06

DOI: https://doi.org/10.5281/zenodo.1478245

New features:

* **New** ``verde.grid_to_table`` function that converts grids to xyz tables with the
  coordinate and data values for each grid point
  (`#148 <https://github.com/fatiando/verde/pull/148>`__)
* Add an ``extra_coords`` option to coordinate generators (``grid_coordinates``,
  ``scatter_points``, and ``profile_coordinates``) to specify a constant value to be
  used as an extra coordinate (`#145 <https://github.com/fatiando/verde/pull/145>`__)
* Allow gridders to pass extra keyword arguments (``**kwargs``) for the coordinate
  generator functions (`#144 <https://github.com/fatiando/verde/pull/144>`__)

Improvements:

* Don't use the Jacobian matrix for predictions to avoid memory overloads. Use dedicated
  and numba wrapped functions instead. As a consequence, predictions are also a bit
  faster when numba is installed (`#149 <https://github.com/fatiando/verde/pull/149>`__)
* Set the default ``n_splits=5`` when using ``KFold`` from scikit-learn
  (`#143 <https://github.com/fatiando/verde/pull/143>`__)

Bug fixes:

* Use the xarray grid's pcolormesh method instead of matplotlib to plot grids in the
  examples. The xarray method takes care of shifting the pixels by half a spacing when
  grids are not pixel registered (`#151 <https://github.com/fatiando/verde/pull/151>`__)

New contributors to the project:

* Jesse Pisel


Version 1.0.1
-------------

Released on: 2018/10/10

DOI: https://doi.org/10.5281/zenodo.1421979

* Paper submission to JOSS (`#134 <https://github.com/fatiando/verde/pull/134>`__). This
  is the new default citation for Verde.
* Remove default ``shape`` for the ``grid`` method (`#140 <https://github.com/fatiando/verde/pull/140>`__).
  There is no reason to have one and it wasn't even implemented in ``grid_coordinates``.
* Fix typo in the weights tutorial (`#136 <https://github.com/fatiando/verde/pull/136>`__).


Version 1.0.0
-------------

Released on: 2018/09/13

DOI: https://doi.org/10.5281/zenodo.1415281

* First release of Verde. Establishes the gridder API and includes blocked reductions,
  bi-harmonic splines [Sandwell1987]_, coupled 2D interpolation [SandwellWessel2016]_,
  chaining operations to form a pipeline, and more.
