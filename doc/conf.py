# Copyright (c) 2017 The Verde Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import datetime

import pygmt.sphinx_gallery
from sphinx_gallery.sorting import FileNameSortKey

import verde

# Project information
# -----------------------------------------------------------------------------
project = "Verde"
copyright = f"2017-{datetime.date.today().year}, The {project} Developers"  # noqa: A001
if len(verde.__version__.split("+")) > 1 or verde.__version__ == "unknown":
    version = "dev"
else:
    version = verde.__version__

# General configuration
# -----------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
    "jupyter_sphinx",
]

# Configuration to include links to other project docs when referencing
# functions/classes
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "pygmt": ("https://www.pygmt.org/latest/", None),
}

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = []

# Create cross-references for the parameter types in the Parameters, Other
# Returns and Yields sections of the docstring
numpydoc_xref_param_type = True

# Format the Attributes like the Parameters section.
numpydoc_attributes_as_param_list = True

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ["png"]

# Sphinx project configuration
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
source_suffix = ".rst"
source_encoding = "utf-8"
master_doc = "index"
pygments_style = "default"
add_function_parentheses = False

# Sphinx-Gallery configuration
# -----------------------------------------------------------------------------
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["gallery_src", "tutorials_src"],
    # path where to save gallery generated examples
    "gallery_dirs": ["gallery", "tutorials"],
    "filename_pattern": r"\.py",
    # Remove the "Download all examples" button from the top level gallery
    "download_all_examples": False,
    # Sort gallery example by file name instead of number of lines (default)
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "api/generated/backreferences",
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": "verde",
    # Insert links to documentation of objects in the examples
    "reference_url": {"verde": None},
    # Use the PyGMT image scraper
    "image_scrapers": ("matplotlib", pygmt.sphinx_gallery.PyGMTScraper()),
}

# HTML output configuration
# -----------------------------------------------------------------------------
html_title = f'{project} <span class="project-version">{version}</span>'
# Don't use the logo since it gets in the way of the project name and is
# repeated in the front page.
# html_logo = "_static/verde-logo.png"
html_favicon = "_static/favicon.png"
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = True
html_static_path = ["_static"]
html_extra_path = []
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True
# CSS files are relative to the static path
html_css_files = ["custom.css"]

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/fatiando/verde",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "home_page_in_toc": False,
}
