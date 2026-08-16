"""core — I/O, validation, zonal statistics, ARF, and statistical analysis.

Re-exports the public API from sub-modules so that consumers can write::

    from watershed_analyzer.core import (
        load_shapefile,
        load_raster_bytes,
        load_netcdf,
        compute_basin_area_km2,
    )
"""

from watershed_analyzer.core.io import (
    compute_basin_area_km2,
    load_netcdf,
    load_raster_bytes,
    load_shapefile,
)

__all__ = [
    "load_shapefile",
    "load_raster_bytes",
    "load_netcdf",
    "compute_basin_area_km2",
]
