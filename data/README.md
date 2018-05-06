# Sample data sets

These files are used as sample data in Verde and are downloaded by
`verde.datasets` functions.

The corresponding `*.sha256` files contain the hash of the data file. It's used
to check if the downloaded data needs to be updated. Generate a hash for a
new/updated file using OpenSSL:

    openssl sha256 FILE_NAME | awk '{print $2}' > FILE_NAME.sha256

The `awk` command isolates the hash only from the full output of `openssl`.

Contents:

* `baja-california-bathymetry.csv.xz`: Bathymetry data from Baja California.
  This is the GMT tutorial data `@tut_ship.xyz` stored in CSV format and
  `xz` compressed.
