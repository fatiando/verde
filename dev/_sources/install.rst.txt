.. _install:

Installing
==========

Which Python?
-------------

You'll need **Python 3.6 or greater**.

We recommend using the
`Anaconda Python distribution <https://www.anaconda.com/download>`__
to ensure you have all dependencies installed and the ``conda`` package manager
available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.


Dependencies
------------

* `numpy <http://www.numpy.org/>`__
* `scipy <https://docs.scipy.org/doc/scipy/reference/>`__
* `pandas <http://pandas.pydata.org/>`__
* `xarray <http://xarray.pydata.org/>`__
* `scikit-learn <http://scikit-learn.org/>`__
* `pooch <http://www.fatiando.org/pooch/>`__
* `dask <https://dask.org/>`__

The following are optional dependencies that can make some parts of the code faster if
they are installed:

* `numba <https://numba.pydata.org/>`__: replaces numpy calculations of predictions and
  Jacobian matrices in splines with faster and more memory efficient multi-threaded
  versions.
* `pykdtree <https://github.com/storpipfugl/pykdtree>`__: replaces
  :class:`scipy.spatial.cKDTree` for better performance in near neighbor calculations
  used in blocked operations, distance masking, etc.

Most of the examples in the :ref:`gallery` and :ref:`tutorials` also use:

* `matplotlib <https://matplotlib.org/>`__
* `cartopy <https://scitools.org.uk/cartopy/>`__ for plotting maps
* `pyproj <https://jswhit.github.io/pyproj/>`__ for cartographic projections


Installing with conda
---------------------

You can install Verde using the `conda package manager <https://conda.io/>`__ that comes
with the Anaconda distribution::

    conda install verde --channel conda-forge


Installing with pip
-------------------

Alternatively, you can also use the `pip package manager
<https://pypi.org/project/pip/>`__::

    pip install verde


Installing the latest development version
-----------------------------------------

You can use ``pip`` to install the latest source from Github::

    pip install https://github.com/fatiando/verde/archive/master.zip

Alternatively, you can clone the git repository locally and install from there::

    git clone https://github.com/fatiando/verde.git
    cd verde
    pip install .


Testing your install
--------------------

We ship a full test suite with the package.
To run the tests, you'll need to install some extra dependencies first:

* `pytest <https://docs.pytest.org/>`__
* `pytest-mpl <https://github.com/matplotlib/pytest-mpl>`__
* `matplotlib <https://matplotlib.org/>`__
* `cartopy <https://scitools.org.uk/cartopy/>`__
* `dask <https://dask.pydata.org/>`__
* `pyproj <https://jswhit.github.io/pyproj/>`__

After that, you can test your installation by running the following inside a Python
interpreter::

    import verde
    verde.test()
