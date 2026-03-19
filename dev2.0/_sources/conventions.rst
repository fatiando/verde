.. _conventions:

Definitions and conventions
===========================

Here are a few of the conventions and definitions we use across Verde:

.. glossary::

    Coordinate types
        Coordinates can be **Cartesian or Geographic**. We generally make **no
        assumptions** about which one you're using.

    Order of coordinates
        Coordinates are usually given as West-East and South-North. For example,
        ``longitude, latitude`` or ``easting, northing``. All functions and
        classes expect coordinates **in this order**. This applies to the
        actual coordinate values, bounding regions, grid spacing, etc.
        **Exceptions** to this rule are the ``dims`` and ``shape`` arguments.

    Coordinate names
        We **don't use names like "x" and "y"** to avoid ambiguity. Cartesian
        coordinates are "easting" and "northing" and Geographic coordinates are
        "longitude" and "latitude". Sometimes this doesn't make sense, like
        when using a polar projection, but we keep the convention for the sake
        of consistency.

    Region
        The term "region" means **the bounding box of the data**. It is ordered
        west, east, south, north.
