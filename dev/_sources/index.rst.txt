.. title:: Home

========
|banner|
========

.. |banner| image:: _static/readme-banner.png
    :alt: Verde Documentation
    :align: middle

**Verde** is a Python library for processing spatial data (bathymetry,
geophysics surveys, etc) and interpolating it on regular grids (i.e.,
*gridding*).

Our core interpolation methods are inspired by machine-learning. As such, Verde
implements an interface that is similar to the popular
`scikit-learn <https://scikit-learn.org/>`__ library.
We also provide other analysis methods that are often used in combination with
gridding, like trend removal, blocked/windowed operations, cross-validation,
and more!


.. panels::
    :header: text-center text-large
    :card: border-1 m-1 text-center

    **Getting started**
    ^^^^^^^^^^^^^^^^^^^

    New to Verde? Start here!

    .. link-button:: overview
        :type: ref
        :text: Overview
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Need help?**
    ^^^^^^^^^^^^^^

    Ask on our community channels

    .. link-button:: https://www.fatiando.org/contact
        :type: url
        :text: Join the conversation
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Reference documentation**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    A list of modules and functions

    .. link-button:: api
        :type: ref
        :text: API reference
        :classes: btn-outline-primary btn-block stretched-link

    ---

    **Using Verde for research?**
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Citations help support our work

    .. link-button:: citing
        :type: ref
        :text: Cite Verde
        :classes: btn-outline-primary btn-block stretched-link


.. seealso::

    Verde is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`__ project.


----


Table of contents
-----------------

.. toctree::
    :maxdepth: 1
    :caption: Getting Started

    tutorials/overview.rst
    install.rst
    gallery/index.rst

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    sample_data/index.rst
    tutorials/grid_coordinates.rst
    tutorials/trends.rst
    tutorials/decimation.rst
    tutorials/projections.rst
    tutorials/chain.rst
    tutorials/model_evaluation.rst
    tutorials/model_selection.rst
    tutorials/weights.rst
    tutorials/vectors.rst

.. toctree::
    :maxdepth: 1
    :caption: Reference documentation

    api/index.rst
    citing.rst
    changes.rst
    references.rst
    versions.rst

.. toctree::
    :maxdepth: 1
    :caption: Community

    Join the community <https://www.fatiando.org/contact/>
    How to contribute <https://github.com/fatiando/verde/blob/master/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/verde/blob/master/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/verde>
    The Fatiando a Terra project <https://www.fatiando.org>
