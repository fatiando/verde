.. _sample_data:

Sample Data
===========

Verde provides some sample data and ways of generating synthetic data through the
``verde.datasets`` module. The sample data are automatically downloaded from the `Github
repository <https://github.com/fatiando/verde>`__ to a folder on your computer the first
time you use them. After that, the data are loaded from this folder. The download is
managed by the :mod:`pooch` package. See :func:`pooch.os_cache` for an explanation of
where the data is stored depending on your system.

.. currentmodule:: verde

Loading Data
------------

.. autosummary::
   :toctree: ../api/generated/

    datasets.fetch_baja_bathymetry
    datasets.fetch_california_gps
    datasets.fetch_rio_magnetic

Utiltiy Functions
-----------------

Setting up `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`__ maps can be a
bit repetitive so we provide some utility functions to automate this in the examples and
tutorials.

.. autosummary::
   :toctree: ../api/generated/

    datasets.setup_baja_bathymetry_map
    datasets.setup_california_gps_map
    datasets.setup_rio_magnetic_map

Synthetic Data
--------------

.. autosummary::
   :toctree: ../api/generated/

    datasets.CheckerBoard

Examples
--------
