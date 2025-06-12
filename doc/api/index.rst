.. _api:

List of functions and classes (API)
===================================

.. automodule:: verde

.. currentmodule:: verde

Interpolators
-------------

.. autosummary::
   :toctree: generated/

    Spline
    SplineCV
    KNeighbors
    Linear
    Cubic
    VectorSpline2D

Data Processing
---------------

.. autosummary::
   :toctree: generated/

    BlockReduce
    BlockMean
    Trend

Composite Estimators
--------------------

.. autosummary::
   :toctree: generated/

    Vector

Model Selection
---------------

.. autosummary::
   :toctree: generated/

    train_test_split
    cross_val_score
    BlockShuffleSplit
    BlockKFold

Coordinate Manipulation
-----------------------

.. autosummary::
   :toctree: generated/

    grid_coordinates
    scatter_points
    profile_coordinates
    get_region
    pad_region
    inside
    rolling_window
    expanding_window

Projection
----------

.. autosummary::
   :toctree: generated/

    project_region
    project_grid

Masking
-------

.. autosummary::
   :toctree: generated/

    distance_mask
    convexhull_mask

Utilities
---------

.. autosummary::
   :toctree: generated/

    maxabs
    variance_to_weights
    grid_to_table
    make_xarray_grid
    median_distance

Input/Output
------------

.. autosummary::
   :toctree: generated/

    load_surfer

.. automodule:: verde.synthetic

.. currentmodule:: verde

Synthetic data
--------------

.. autosummary::
    :toctree: generated/

    synthetic.CheckerBoard

Base Classes and Functions
--------------------------

.. autosummary::
   :toctree: generated/

    base.BaseGridder
    base.BaseBlockCrossValidator
    base.n_1d_arrays
    base.least_squares
