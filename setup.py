"""
Build and install the project.

Uses versioneer to manage version numbers using git tags.
"""
from setuptools import setup, find_packages

import versioneer


NAME = "verde"
FULLNAME = "Verde"
AUTHOR = "The Verde Developers"
AUTHOR_EMAIL = "leouieda@gmail.com"
MAINTAINER = "Leonardo Uieda"
MAINTAINER_EMAIL = AUTHOR_EMAIL
LICENSE = "BSD License"
URL = "https://github.com/fatiando/verde"
DESCRIPTION = "Processing and gridding spatial data"
KEYWORDS = ""
with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())
VERSION = versioneer.get_version()
CMDCLASS = versioneer.get_cmdclass()
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: {}".format(LICENSE),
]
PLATFORMS = "Any"
PACKAGES = find_packages(exclude=["doc"])
SCRIPTS = []
PACKAGE_DATA = {
    "verde.datasets": ["registry.txt"],
    "verde.tests": ["data/*", "baseline/*"],
}
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "pandas",
    "xarray",
    "scikit-learn",
    "pooch>=0.7.0",
    "dask",
]
PYTHON_REQUIRES = ">=3.6"

if __name__ == "__main__":
    setup(
        name=NAME,
        fullname=FULLNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        url=URL,
        platforms=PLATFORMS,
        scripts=SCRIPTS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        python_requires=PYTHON_REQUIRES,
        cmdclass=CMDCLASS,
    )
