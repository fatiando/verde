"""
Get the version number and commit hash from versioneer.
"""
from ._version import get_versions


version = get_versions()["version"]
git_revision = get_versions()["full-revisionid"]
