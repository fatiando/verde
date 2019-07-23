.. _changes:

Changelog
=========


Version 1.2.0
-------------

*Released on: 2019/07/23*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3347076.svg
   :target: https://doi.org/10.5281/zenodo.3347076

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

*Released on: 2018/11/06*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1478245.svg
   :target: https://doi.org/10.5281/zenodo.1478245

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

*Released on: 2018/10/10*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1421979.svg
   :target: https://doi.org/10.5281/zenodo.1421979

* Paper submission to JOSS (`#134 <https://github.com/fatiando/verde/pull/134>`__). This
  is the new default citation for Verde.
* Remove default ``shape`` for the ``grid`` method (`#140 <https://github.com/fatiando/verde/pull/140>`__).
  There is no reason to have one and it wasn't even implemented in ``grid_coordinates``.
* Fix typo in the weights tutorial (`#136 <https://github.com/fatiando/verde/pull/136>`__).


Version 1.0.0
-------------

*Released on: 2018/09/13*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1415281.svg
   :target: https://doi.org/10.5281/zenodo.1415281

* First release of Verde. Establishes the gridder API and includes blocked reductions,
  bi-harmonic splines [Sandwell1987]_, coupled 2D interpolation [SandwellWessel2016]_,
  chaining operations to form a pipeline, and more.
