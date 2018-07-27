# Sample data sets

These files are used as sample data in Verde and are downloaded by `verde.datasets`
functions:

* `baja-bathymetry.csv.xz`: Bathymetry data from Baja California. This is the GMT
  tutorial data `@tut_ship.xyz` stored in CSV format and `xz` compressed.
* `rio-magnetic.csv.xz`: Total-field magnetic anomaly data from the northwestern part of
  an airborne survey of Rio de Janeiro, Brazil, conducted in 1978. Columns are
  longitude, latitude, total field anomaly (nanoTesla), and observation height above the
  ellipsoid (meters). The anomaly is calculated with respect to the IGRF field at the
  center of the survey area at 500 m altitude for the year 1978.3. This dataset was
  cropped from the original survey, made available by the Geological Survey of Brazil
  (CPRM) through their [GEOSGB portal](http://geosgb.cprm.gov.br/). See the original
  data for more processing information.
* `california-gps.csv.xz`: GPS velocities (east, north, vertical) from the United States
  West coast cut from EarthScope Plate Boundary Observatory data provided by UNAVCO
  through the GAGE Facility with support from the National Science Foundation (NSF) and
  National Aeronautics and Space Administration (NASA) under NSF Cooperative Agreement
  No. EAR-1261833. Velocities are referenced to the North American tectonic plate
  (NAM08).
