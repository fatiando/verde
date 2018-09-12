.. _sample_data:

Sample Data
===========

Verde provides some sample data and ways of generating synthetic data through the
:mod:`verde.datasets` module. The sample data are automatically downloaded from the `Github
repository <https://github.com/fatiando/verde>`__ to a folder on your computer the first
time you use them. After that, the data are loaded from this folder. The download is
managed by the :mod:`pooch` package.


Where is my data?
-----------------

The data files are downloaded to a folder ``~/.verde/data/`` by default. This is the
*base data directory*. :mod:`pooch` will create a separate folder in the base directory
for each version of Verde. So for Verde 0.1, the base data dir is ``~/.verde/data/0.1``.
If you're using the latest development version from Github, the version is ``master``.

You can change the base data directory by setting the ``VERDE_DATA_DIR`` environment
variable to a different path.


Available datasets
------------------

These are the datasets currently available. Most also come with a function for setting
up a Cartopy map to display the data.
