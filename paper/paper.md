---
title: "Verde: Processing and gridding spatial data using Green’s functions"
tags:
  - python
  - geophysics
  - geospatial
authors:
  - name: Leonardo Uieda
    orcid: 0000-0001-6123-9515
    affiliation: 1
affiliations:
 - name: Department of Earth Sciences, SOEST, University of Hawai'i at Mānoa, Honolulu, Hawaii, USA
   index: 1
date: 13 September 2018
bibliography: paper.bib
---

# Summary

Measurements made on the surface of the Earth are often sparse and unevenly distributed.
For example, GPS displacement measurements are limited by the availability of ground
stations and airborne geophysical measurements are highly sampled along flight lines but
there is often a large gap between lines. Many data processing methods require data
distributed on a uniform regular grid, particularly methods involving the Fourier
transform or the computation of directional derivatives. Hence, the interpolation of
sparse measurements onto a regular grid (known as *gridding*) is a prominent problem in
the Earth Sciences.

Popular gridding methods include kriging, minimum curvature with tension [@smith1990],
and bi-harmonic splines [@sandwell1987]. The latter belongs to a group of methods often
called *radial basis functions* and is similar to the *thin-plate spline* [@franke1982].
In these methods, the data are assumed to be represented by a linear combination of
Green's functions,

$$ d_i = \sum\limits_{j=1}^M p_j G(\mathbf{x}_i, \mathbf{x}_j) , $$

in which $d_i$ is the $i$th datum, $p_j$ is a scalar coefficient, $G$ is a Green's
function, and $\mathbf{x}_i$ and $\mathbf{x}_j$ are the position vectors for the datum
and the point defining the Green's function, respectively. Interpolation is done by
estimating the $M$ $p_j$ coefficients through linear least-squares and using them to
predict data values at new locations on a grid. Essentially, these methods are linear
models used for prediction. As such, many of the model selection and evaluation
techniques used in machine learning can be applied to griding problems as well.

*Verde* is a Python library for gridding spatial data using different Green's functions.
It differs from the radial basis functions in `scipy.interpolate` by providing an API
inspired by scikit-learn [@pedregosa2011]. The *Verde* API should be familiar to
scikit-learn users but is tweaked to work with spatial data, which has Cartesian or
geographic coordinates and multiple data components instead of an `X` feature matrix and
`y` label vector. The library also includes more specialized Green's functions
[@sandwell2016], utilities for trend estimation and data decimation (which are often
required prior to gridding [@smith1990]), and more. Some of these interpolation and data
processing methods already exist in the Generic Mapping Tools (GMT) [@wessel2013a], a
command-line program popular in the Earth Sciences. However, there are no model
selection tools in GMT and it can be difficult to separate parts of the processing that
are done internally by its modules. *Verde* is designed to be modular, easily extended,
and integrated into the scientific Python ecosystem. It can be used to implement new
interpolation methods by subclassing the `verde.base.BaseGridder` class, requiring only
the implementation of the new Green's function. For example, it is currently being used
to develop a method for interpolation of 3-component GPS data [@uieda2018].

# Acknowledgements

I would like to thank collaborators Paul Wessel and David Hoese, reviewers Dom Fournier 
and Philippe Rivière, and editor Lindsey Heagy for helpful discussions and contributions
to this project. This is SOEST contribution 10467.

# References
