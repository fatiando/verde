# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Testing utilities.
"""
import pytest

try:
    import numba
except ImportError:
    numba = None


def requires_numba(function):
    """
    Skip the decorated test if numba is not installed.
    """
    mark = pytest.mark.skipif(numba is None, reason="requires numba")
    return mark(function)
