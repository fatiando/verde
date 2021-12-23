# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Generators of synthetic datasets.
"""
import warnings

from ..synthetic import CheckerBoard as _CheckerBoard


class CheckerBoard(_CheckerBoard):
    """
    .. warning::

        Using ``CheckerBoard`` from ``verde.datasets`` is deprecated and will
        be removed in Verde 2.0.0. Use ``verde.synthetic.CheckerBoard``
        instead.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        warnings.warn(
            "Using CheckerBoard from verde.datasets is deprecated and will be "
            "removed in Verde 2.0.0. "
            "Use verde.synthetic.CheckerBoard instead.",
            FutureWarning,
        )
