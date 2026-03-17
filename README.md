<img src="https://github.com/fatiando/verde/raw/main/doc/_static/readme-banner.png" alt="Verde">

<h2 align="center">Processing and gridding spatial data, machine-learning style</h2>

<p align="center">
<a href="https://www.fatiando.org/verde"><strong>Documentation</strong> (latest)</a> •
<a href="https://www.fatiando.org/verde/dev"><strong>Documentation</strong> (main branch)</a> •
<a href="https://github.com/fatiando/verde/blob/main/CONTRIBUTING.md"><strong>Contributing</strong></a> •
<a href="https://www.fatiando.org/contact/"><strong>Contact</strong></a> •
<a href="https://github.com/orgs/fatiando/discussions"><strong>Ask a question</strong></a>
</p>

<p align="center">
Part of the <a href="https://www.fatiando.org"><strong>Fatiando a Terra</strong></a> project
</p>

<p align="center">
<a href="https://pypi.python.org/pypi/verde"><img src="http://img.shields.io/pypi/v/verde.svg?style=flat-square" alt="Latest version on PyPI"></a>
<a href="https://github.com/conda-forge/verde-feedstock"><img src="https://img.shields.io/conda/vn/conda-forge/verde.svg?style=flat-square" alt="Latest version on conda-forge"></a>
<a href="https://codecov.io/gh/fatiando/verde"><img src="https://img.shields.io/codecov/c/github/fatiando/verde/main.svg?style=flat-square" alt="Test coverage status"></a>
<a href="https://pypi.python.org/pypi/verde"><img src="https://img.shields.io/pypi/pyversions/verde.svg?style=flat-square" alt="Compatible Python versions."></a>
<a href="https://doi.org/10.21105/joss.00957"><img src="https://img.shields.io/badge/doi-10.21105%2Fjoss.00957-blue?style=flat-square" alt="DOI used to cite this software"></a>
</p>

## About

**Verde** is a Python library for processing spatial data (topography, point
clouds, bathymetry, geophysics surveys, etc) and interpolating them on a 2D
surface (i.e., gridding) with a hint of machine learning.

Our core interpolation methods are inspired by machine-learning.
As such, Verde implements an interface that is similar to the popular
[scikit-learn](https://scikit-learn.org/) library.
We also provide other analysis methods that are often used in combination with
gridding, like trend removal, blocked/windowed operations, cross-validation,
and more!

## Project goals

* Provide a machine-learning inspired interface for gridding spatial data
* Integration with the Scipy stack: numpy, pandas, scikit-learn, and xarray
* Include common processing and data preparation tasks, like blocked means and 2D trends
* Support for gridding scalar and vector data (like wind speed or GPS velocities)
* Support for both Cartesian and geographic coordinates

## Project status

**Verde is stable and ready for use!**
This means that we are careful about introducing backwards incompatible changes
and will provide ample warning when doing so. Upgrading minor versions of Verde
should not require making changes to your code.

The first major release of Verde was focused on meeting most of these initial
goals and establishing the look and feel of the library.
Later releases will focus on expanding the range of gridders available,
optimizing the code, and improving algorithms so that larger-than-memory
datasets can also be supported.

## Getting involved

🗨️ **Contact us:**
Find out more about how to reach us at
[fatiando.org/contact](https://www.fatiando.org/contact/).

👩🏾‍💻 **Contributing to project development:**
Please read our
[Contributing Guide](https://github.com/fatiando/verde/blob/main/CONTRIBUTING.md)
to see how you can help and give feedback.

🧑🏾‍🤝‍🧑🏼 **Code of conduct:**
This project is released with a
[Code of Conduct](https://github.com/fatiando/community/blob/main/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.

> **Imposter syndrome disclaimer:**
> We want your help. **No, really.** There may be a little voice inside your
> head that is telling you that you're not ready, that you aren't skilled
> enough to contribute. We assure you that the little voice in your head is
> wrong. Most importantly, **there are many valuable ways to contribute besides
> writing code**.
>
> *This disclaimer was adapted from the*
> [MetPy project](https://github.com/Unidata/MetPy).

## License

This is free software: you can redistribute it and/or modify it under the terms
of the **BSD 3-clause License**. A copy of this license is provided in
[`LICENSE.txt`](https://github.com/fatiando/verde/blob/main/LICENSE.txt).
