# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# pylint: disable=invalid-name
"""
Get the version number and commit hash from setuptools-scm.
"""
from pkg_resources import get_distribution


# Get semantic version through setuptools-scm
version = get_distribution("verde").version
# Append a "v" before the semver version so it looks like: v0.1.0
full_version = "v" + version
