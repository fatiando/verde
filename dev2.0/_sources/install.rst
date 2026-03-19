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

You'll need **Python >= 3.7**.
See :ref:`python-versions` if you require support for older versions.

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
* `dask <https://dask.org/>`__
* `numba <https://numba.pydata.org/>`__
* `bordado <https://www.fatiando.org/bordado>`__

Our examples use other packages as well which are not used within Verde itself.
If you wish to **run the examples in the documentation**, you will also have to
install:

* `matplotlib <https://matplotlib.org/>`__
* `pygmt <https://www.pygmt.org>`__ for plotting maps
* `pyproj <https://jswhit.github.io/pyproj/>`__ for cartographic projections
