.. _compatibility:

Version compatibility
=====================

Verde version compatibility
---------------------------

Verde uses `semantic versioning <https://semver.org/>`__ (i.e.,
``MAJOR.MINOR.BUGFIX`` format).

* Major releases mean that backwards incompatible changes were made.
  Upgrading will require users to change their code.
* Minor releases add new features/data without changing existing functionality.
  Users can upgrade minor versions without changing their code.
* Bug fix releases fix errors in a previous release without adding new
  functionality. Users can upgrade minor versions without changing their code.

We will add ``FutureWarning`` messages about deprecations ahead of making any
breaking changes to give users a chance to upgrade.

.. _dependency-versions:

Supported dependency versions
-----------------------------

Verde follows the recommendations in
`NEP29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`__ for setting
the minimum required version of our dependencies.
In short, we support **all minor releases of our dependencies from the previous
24 months** before a Verde release with a minimum of 2 minor releases.

We follow this guidance conservatively and won't require newer versions if the
older ones are still working without causing problems.
Whenever support for a version is dropped, we will include a note in the
:ref:`changes`.

.. note::

    This was introduced in Verde v1.8.0.


.. _python-versions:

Supported Python versions
-------------------------

If you require support for older Python versions, please pin Verde to the
following releases to ensure compatibility:

.. list-table::
    :widths: 40 60

    * - **Python version**
      - **Last compatible release**
    * - 3.6
      - 1.7.0
