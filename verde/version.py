# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=invalid-name
"""
Get the version number and commit hash from versioneer.
"""
from ._version import get_versions


full_version = get_versions()["version"]
git_revision = get_versions()["full-revisionid"]
