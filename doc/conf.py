# -*- coding: utf-8 -*-
import sys
import os
import datetime
import sphinx_rtd_theme
import sphinx_gallery

# Sphinx needs to be able to import the package to use autodoc and get the
# version number
sys.path.append(os.path.pardir)

from verde import __version__, __commit__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = False

numpydoc_class_members_toctree = False

plot_gallery = True
sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': '../examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'gallery',
    'filename_pattern': '\.py',
    # directory where function granular galleries are stored
    'backreferences_dir': 'api/backreferences',
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    'doc_module': 'verde',
    # Insert links to documentation of objects in the examples
    'reference_url': {'verde': None},
}

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']

# Sphinx project configuration
templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']
source_suffix = '.rst'
# The encoding of source files.
source_encoding = 'utf-8-sig'
master_doc = 'index'

# General information about the project
year = datetime.date.today().year
project = 'Verde'
copyright = '2018-{}, Leonardo Uieda'.format(year)
if len(__version__.split('+')) > 1 or __version__ == 'unknown':
    version = 'dev'
else:
    version = __version__

# These enable substitutions using |variable| in the rst files
rst_epilog = """
.. |year| replace:: {year}
""".format(year=year)

html_last_updated_fmt = '%b %d, %Y'
html_title = 'verde'
html_short_title = 'verde'
html_logo = ''
html_favicon = '_static/favicon.png'
html_static_path = ['_static']
html_extra_path = []
pygments_style = 'default'
add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# Theme config
html_theme = "sphinx_rtd_theme"
html_theme_options = {
}
html_context = {
    'menu_links': [
        ('<i class="fa fa-users fa-fw"></i> Contributing', 'https://github.com/fatiando/verde/blob/master/CONTRIBUTING.md'),
        ('<i class="fa fa-gavel fa-fw"></i> Code of Conduct', 'https://github.com/fatiando/verde/blob/master/CODE_OF_CONDUCT.md'),
        ('<i class="fa fa-comment fa-fw"></i> Contact', 'https://gitter.im/fatiando/fatiando'),
        ('<i class="fa fa-github fa-fw"></i> Source Code', 'https://github.com/fatiando/verde'),
    ],
    # Custom variables to enable "Improve this page"" and "Download notebook"
    # links
    'doc_path': 'doc',
    'github_repo': 'fatiando/verde',
    'github_version': 'master',
}

# Load the custom CSS files (needs sphinx >= 1.6 for this to work)
def setup(app):
    app.add_stylesheet("style.css")
