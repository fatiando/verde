.. image:: https://github.com/fatiando/verde/raw/master/doc/_static/readme-banner.png
    :alt: Verde

`Documentation <http://www.fatiando.org/verde>`__ |
`Documentation (dev version) <http://www.fatiando.org/verde/dev>`__ |
`Contact <https://gitter.im/fatiando/fatiando>`__ |
Part of the `Fatiando a Terra <https://www.fatiando.org>`__ project


.. image:: http://img.shields.io/pypi/v/verde.svg?style=flat-square
    :alt: Latest version on PyPI
    :target: https://pypi.python.org/pypi/verde
.. image:: http://img.shields.io/travis/fatiando/verde/master.svg?style=flat-square&label=Linux|Mac
    :alt: TravisCI build status
    :target: https://travis-ci.org/fatiando/verde
.. image:: http://img.shields.io/appveyor/ci/fatiando/verde/master.svg?style=flat-square&label=Windows
    :alt: AppVeyor build status
    :target: https://ci.appveyor.com/project/fatiando/verde
.. image:: https://img.shields.io/codecov/c/github/fatiando/verde/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://codecov.io/gh/fatiando/verde
.. image:: https://img.shields.io/codeclimate/maintainability/fatiando/verde.svg?style=flat-square
    :alt: Code quality status
    :target: https://codeclimate.com/github/fatiando/verde
.. image:: https://img.shields.io/codacy/grade/6b698defc0df47288a634930d41a9d65.svg?style=flat-square&label=codacy
    :alt: Code quality grade on codacy
    :target: https://www.codacy.com/app/leouieda/verde
.. image:: https://img.shields.io/pypi/pyversions/verde.svg?style=flat-square
    :alt: Compatible Python versions.
    :target: https://pypi.python.org/pypi/verde
.. image:: https://img.shields.io/gitter/room/fatiando/fatiando.svg?style=flat-square
    :alt: Chat room on Gitter
    :target: https://gitter.im/fatiando/fatiando


 .. placeholder-for-doc-index


Disclaimer
----------

🚨 **This package is in early stages of design and implementation.** 🚨

We welcome any feedback and ideas!
Let us know by submitting
`issues on Github <https://github.com/fatiando/verde/issues>`__
or send us a message on our
`Gitter chatroom <https://gitter.im/fatiando/fatiando>`__.


About
-----

Verde is a Python library for processing spatial data (bathymetry, geophysics
surveys, etc) and interpolating it on regular grids (i.e., *gridding*).

Most gridding methods in Verde use a Green's functions approach.
A linear model is estimated based on the input data and then used to predict
data on a regular grid (or in a scatter, a profile, as derivatives).
The models are Green's functions from (mostly) elastic deformation theory.
This approach is very similar to *machine learning* so we implement gridder
classes that are similar to `scikit-learn <http://scikit-learn.org/>`__
regression classes.
The API is not 100% compatible but it should look familiar to those with some
scikit-learn experience.

Advantages of using Green's functions include:

* Easily apply **weights** to data points. This is a linear least-squares
  problem.
* Perform **model selection** using established machine learning techniques,
  like k-fold or holdout cross-validation.
* The estimated model can be **easily stored** for later use, like
  spherical-harmonic coefficients are used in gravimetry.


Project goals
-------------

* Provide a machine-learning inspired interface for Green's functions gridding
  of spatial data
* Integration with the Scipy stack: numpy, pandas (for xyz data), and xarray
  (for grids)
* Include functions for common processing and data preparation tasks, like
  blocked means and medians
* Support for gridding scalar and vector data (like wind speed or GPS
  velocities)
* Support for both Cartesian and geographic coordinates


Contacting Us
-------------

* Most discussion happens `on Github <https://github.com/fatiando/verde>`__.
  Feel free to `open an issue
  <https://github.com/fatiando/verde/issues/new>`__ or comment
  on any open issue or pull request.
* We have `chat room on Gitter <https://gitter.im/fatiando/fatiando>`__
  where you can ask questions and leave comments.


Contributing
------------

Code of conduct
+++++++++++++++

Please note that this project is released with a
`Contributor Code of Conduct <https://github.com/fatiando/verde/blob/master/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

Contributing Guidelines
+++++++++++++++++++++++

Please read our
`Contributing Guide <https://github.com/fatiando/verde/blob/master/CONTRIBUTING.md>`__
to see how you can help and give feedback.

Imposter syndrome disclaimer
++++++++++++++++++++++++++++

**We want your help.** No, really.

There may be a little voice inside your head that is telling you that you're
not ready to be an open source contributor; that your skills aren't nearly good
enough to contribute.
What could you possibly offer?

We assure you that the little voice in your head is wrong.

**Being a contributor doesn't just mean writing code**.
Equality important contributions include:
writing or proof-reading documentation, suggesting or implementing tests, or
even giving feedback about the project (including giving feedback about the
contribution process).
If you're coming to the project with fresh eyes, you might see the errors and
assumptions that seasoned contributors have glossed over.
If you can write any code at all, you can contribute code to open source.
We are constantly trying out new skills, making mistakes, and learning from
those mistakes.
That's how we all improve and we are happy to help others learn.

*This disclaimer was adapted from the*
`MetPy project <https://github.com/Unidata/MetPy>`__.


License
-------

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
`LICENSE.txt <https://github.com/fatiando/verde/blob/master/LICENSE.txt>`__.


Documentation for other versions
--------------------------------

* `Development <http://www.fatiando.org/verde/dev>`__ (reflects the *master* branch on
  Github)
* `Latest release <http://www.fatiando.org/verde/latest>`__
* `v0.1.0 <http://www.fatiando.org/verde/v0.1a0>`__
