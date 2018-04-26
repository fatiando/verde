.. _install:

Installing
==========

Which Python?
-------------

You'll need **Python 3.5 or greater**.

We recommend using the `Anaconda <http://continuum.io/downloads#all>`__ Python
distribution to ensure you have all dependencies installed and the ``conda``
package manager available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.


Dependencies
------------

* `numpy <http://www.numpy.org/>`__


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
You can run our tests after you install it but you will need to install `pytest
<https://docs.pytest.org/>`__ first.
After that, you can test your installation by running the following inside a
Python interpreter::

    import verde
    verde.test()
