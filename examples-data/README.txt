.. _sample_data:

Sample Data
===========

Verde provides some sample data and ways of generating synthetic data through the
``verde.datasets`` module.
The sample data are automatically downloaded
`from Github <https://github.com/fatiando/verde/tree/master/data>`__ to a
``$HOME/.verde/data`` directory the first time you use them.
After that, the data are loaded from this directory.

If you have any issues with the data, try deleting the ``$HOME/.verde/data`` directory
to force Verde to re-download the data files. If the problem persists, please
`open an issue <https://github.com/fatiando/verde/issues>`__ to let us know.

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
