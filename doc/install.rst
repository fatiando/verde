.. _install:

Installing
==========

There are different ways to install Verde:

.. tab-set::

    .. tab-item:: pip

        Using the `pip <https://pypi.org/project/pip/>`__ package manager:

        .. code:: bash

            python -m pip install verde

    .. tab-item:: conda/mamba

        Using the `conda package manager <https://conda.io/>`__ (or ``mamba``)
        that comes with the Anaconda/Miniconda distribution:

        .. code:: bash

            conda install verde --channel conda-forge

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/verde

.. note::

    The commands above should be executed in a terminal. On Windows, use the
    ``cmd.exe`` or the "Anaconda Prompt" app if you're using Anaconda.

Which Python?
-------------

You'll need **Python >= 3.6**.

We recommend using the
`Anaconda <https://www.anaconda.com/download>`__
or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
Python distributions to ensure you have all dependencies installed and the
``conda`` package manager available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.

.. _dependencies:

Dependencies
------------

The required dependencies should be installed automatically when you install
Verde using ``conda`` or ``pip``.

Required:

* `numpy <http://www.numpy.org/>`__
* `scipy <https://docs.scipy.org/doc/scipy/reference/>`__
* `pandas <http://pandas.pydata.org/>`__
* `xarray <http://xarray.pydata.org/>`__
* `scikit-learn <http://scikit-learn.org/>`__
* `pooch <http://www.fatiando.org/pooch/>`__
* `dask <https://dask.org/>`__

The following are optional dependencies that can make some parts of the code
more efficient if they are installed:

* `numba <https://numba.pydata.org/>`__: replaces numpy calculations of
  predictions and Jacobian matrices in splines with faster and more memory
  efficient multi-threaded versions.
* `pykdtree <https://github.com/storpipfugl/pykdtree>`__: replaces
  :class:`scipy.spatial.cKDTree` for better performance in near neighbor
  calculations used in blocked operations, distance masking, etc.

Our examples use other packages as well which are not used within Verde itself.
If you wish to **run the examples in the documentation**, you will also have to
install:

* `matplotlib <https://matplotlib.org/>`__
* `cartopy <https://scitools.org.uk/cartopy/>`__ for plotting maps
* `pyproj <https://jswhit.github.io/pyproj/>`__ for cartographic projections
