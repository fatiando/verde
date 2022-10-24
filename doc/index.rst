.. title:: Home

.. grid::
    :gutter: 4 4 4 4
    :margin: 5 5 0 0
    :padding: 0 0 0 0

    .. grid-item::
        :columns: 12 8 12 8

        .. grid::
            :padding: 0 0 0 0

            .. grid-item::
                :columns: 12 12 12 9

                .. raw:: html

                    <h1 class="display-1"><img src="_static/verde-title.svg" alt="Verde"></h1>

        .. div:: sd-fs-3

            Processing and gridding spatial data, machine-learning style

    .. grid-item::
        :columns: 12 4 12 4

        .. image:: ./_static/verde-logo.svg
            :width: 200px
            :class: sd-m-auto

**Verde** is a Python library for processing spatial data (bathymetry,
geophysics surveys, etc) and interpolating it on regular grids (i.e.,
*gridding*).

Our core interpolation methods are inspired by machine-learning. As such, Verde
implements an interface that is similar to the popular
`scikit-learn <https://scikit-learn.org/>`__ library.
We also provide other analysis methods that are often used in combination with
gridding, like trend removal, blocked/windowed operations, cross-validation,
and more!

----

.. grid:: 1 2 1 2
    :margin: 5 5 0 0
    :padding: 0 0 0 0
    :gutter: 4

    .. grid-item-card:: :octicon:`info` Getting started
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        New to Verde? Start here!

        .. button-ref:: overview
            :ref-type: ref
            :click-parent:
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`comment-discussion` Need help?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Ask on our community channels.

        .. button-link:: https://www.fatiando.org/contact
            :click-parent:
            :color: primary
            :outline:
            :expand:

            Join the conversation :octicon:`link-external`

    .. grid-item-card:: :octicon:`file-badge` Reference documentation
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        A list of modules and functions.

        .. button-ref:: api
            :ref-type: ref
            :color: primary
            :outline:
            :expand:

    .. grid-item-card:: :octicon:`bookmark` Using Verde for research?
        :text-align: center
        :class-title: sd-fs-5
        :class-card: sd-p-3

        Citations help support our work!

        .. button-ref:: citing
            :ref-type: ref
            :color: primary
            :outline:
            :expand:

----

.. seealso::

    Verde is a part of the
    `Fatiando a Terra <https://www.fatiando.org/>`__ project.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    tutorials/overview.rst
    install.rst
    gallery/index.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User Guide

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
    :hidden:
    :caption: Reference documentation

    api/index.rst
    citing.rst
    references.rst
    changes.rst
    compatibility.rst
    versions.rst

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Community

    Join the community <https://www.fatiando.org/contact/>
    How to contribute <https://github.com/fatiando/verde/blob/master/CONTRIBUTING.md>
    Code of Conduct <https://github.com/fatiando/verde/blob/master/CODE_OF_CONDUCT.md>
    Source code on GitHub <https://github.com/fatiando/verde>
    The Fatiando a Terra project <https://www.fatiando.org>
