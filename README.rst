.. image:: https://github.com/fatiando/verde/raw/main/doc/_static/readme-banner.png
    :alt: Verde

`Documentation <http://www.fatiando.org/verde>`__ |
`Documentation (dev version) <http://www.fatiando.org/verde/dev>`__ |
Part of the `Fatiando a Terra <https://www.fatiando.org>`__ project


.. image:: http://img.shields.io/pypi/v/verde.svg?style=flat-square&label=version
    :alt: Latest version on PyPI
    :target: https://pypi.python.org/pypi/verde
.. image:: https://img.shields.io/conda/vn/conda-forge/verde.svg?style=flat-square
    :alt: Latest version on conda-forge
    :target: https://github.com/conda-forge/verde-feedstock
.. image:: https://img.shields.io/codecov/c/github/fatiando/verde/main.svg?style=flat-square
    :alt: Test coverage status
    :target: https://codecov.io/gh/fatiando/verde
.. image:: https://img.shields.io/pypi/pyversions/verde.svg?style=flat-square
    :alt: Compatible Python versions.
    :target: https://pypi.python.org/pypi/verde
.. image:: https://img.shields.io/badge/doi-10.21105%2Fjoss.00957-blue.svg?style=flat-square
    :alt: Digital Object Identifier for the JOSS paper
    :target: https://doi.org/10.21105/joss.00957


.. placeholder-for-doc-index


About
-----

**Verde** is a Python library for processing spatial data (bathymetry,
geophysics surveys, etc) and interpolating it on regular grids (i.e.,
*gridding*).

Our core interpolation methods are inspired by machine-learning.
As such, Verde implements an interface that is similar to the popular
`scikit-learn <https://scikit-learn.org/>`__ library.
We also provide other analysis methods that are often used in combination with
gridding, like trend removal, blocked/windowed operations, cross-validation,
and more!


Project goals
-------------

* Provide a machine-learning inspired interface for gridding spatial data
* Integration with the Scipy stack: numpy, pandas, scikit-learn, and xarray
* Include common processing and data preparation tasks, like blocked means and 2D trends
* Support for gridding scalar and vector data (like wind speed or GPS velocities)
* Support for both Cartesian and geographic coordinates

The first release of Verde was focused on meeting most of these initial goals
and establishing the look and feel of the library.
Later releases will focus on expanding the range of gridders available,
optimizing the code, and improving algorithms so that larger-than-memory
datasets can also be supported.


Contacting us
-------------

Find out more about how to reach us at
`fatiando.org/contact <https://www.fatiando.org/contact/>`__

Citing Verde
------------

This is research software **made by scientists** (see
`AUTHORS.md <https://github.com/fatiando/verde/blob/main/AUTHORS.md>`__). Citations
help us justify the effort that goes into building and maintaining this project. If you
used Verde for your research, please consider citing us.

See our `CITATION.rst file <https://github.com/fatiando/verde/blob/main/CITATION.rst>`__
to find out more.


Contributing
------------

Code of conduct
+++++++++++++++

Please note that this project is released with a
`Contributor Code of Conduct <https://github.com/fatiando/verde/blob/main/CODE_OF_CONDUCT.md>`__.
By participating in this project you agree to abide by its terms.

Contributing Guidelines
+++++++++++++++++++++++

Please read our
`Contributing Guide <https://github.com/fatiando/verde/blob/main/CONTRIBUTING.md>`__
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
Equally important contributions include:
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
`LICENSE.txt <https://github.com/fatiando/verde/blob/main/LICENSE.txt>`__.
