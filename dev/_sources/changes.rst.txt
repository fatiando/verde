.. _changes:

Changelog
=========

Version 1.0.1 (2018-10-10)
--------------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1421979.svg
   :target: https://doi.org/10.5281/zenodo.1421979

* Paper submission to JOSS (`#134 <https://github.com/fatiando/verde/pull/134>`__). This
  is the new default citation for Verde.
* Remove default ``shape`` for the ``grid`` method (`#140 <https://github.com/fatiando/verde/pull/140>`__).
  There is no reason to have one and it wasn't even implemented in ``grid_coordinates``.
* Fix typo in the weights tutorial (`#136 <https://github.com/fatiando/verde/pull/136>`__).


Version 1.0.0 (2018-09-13)
--------------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1415281.svg
   :target: https://doi.org/10.5281/zenodo.1415281

* First release of Verde. Establishes the gridder API and includes blocked reductions,
  bi-harmonic splines [Sandwell1987]_, coupled 2D interpolation [SandwellWessel2016]_,
  chaining operations to form a pipeline, and more.
