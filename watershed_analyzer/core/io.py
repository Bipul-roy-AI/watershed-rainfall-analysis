"""File I/O and basin geometry utilities for the Watershed Rainfall Analyzer.

This module provides four public helpers:

* ``load_shapefile``  — extract a ``.shp`` from a zipped BytesIO and return
  a :class:`geopandas.GeoDataFrame`.
* ``load_raster_bytes`` — wrap raw raster bytes in a ``rasterio.MemoryFile``
  and return a lightweight metadata dictionary.
* ``load_netcdf`` — load NetCDF / Zarr bytes into an :class:`xarray.DataArray`
  with automatic variable-name detection.
* ``compute_basin_area_km2`` — reproject a polygon to an equal-area CRS and
  compute the basin area in km².

All functions use Python 3.10+ union syntax (``X | None``) for type hints
and Google-style docstrings.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import zipfile
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import MemoryFile
from rasterio.crs import CRS as RasterCRS
from shapely.geometry import Polygon

from watershed_analyzer.config import EQUAL_AREA_CRS

logger = logging.getLogger(__name__)

# Attempt to import streamlit for optional caching; fail gracefully if absent.
try:
    import streamlit as st

    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover — streamlit is an optional dep
    _HAS_STREAMLIT = False
    st = None  # type: ignore[assignment]

# Common NetCDF precipitation variable names, in search order.
_PRECIP_VARIABLES: list[str] = [
    "precip",
    "precipitation",
    "pr",
    "rainfall",
    "tp",
]


# ---------------------------------------------------------------------------
# Optional streamlit caching decorator
# ---------------------------------------------------------------------------
def _cache_data(func):
    """Return ``st.cache_data(func)`` when Streamlit is available, else *func*.

    This thin wrapper keeps the rest of the module free of runtime
    ``if st is not None`` branches.
    """
    if _HAS_STREAMLIT:
        return st.cache_data(func)  # type: ignore[return-value]
    return func


# ---------------------------------------------------------------------------
# load_shapefile
# ---------------------------------------------------------------------------
@_cache_data
def load_shapefile(uploaded_zip: io.BytesIO) -> gpd.GeoDataFrame | None:
    """Extract a shapefile from a ZIP archive and load it as a GeoDataFrame.

    The function scans the ZIP for the first ``.shp`` file, extracts every
    related component (``.shx``, ``.dbf``, ``.prj``, …) into a temporary
    directory, and then reads the shapefile with :func:`geopandas.read_file`.

    When :mod:`streamlit` is installed the result is cached via
    ``@st.cache_data`` so that re-runs do not re-read the same upload.

    Parameters
    ----------
    uploaded_zip : io.BytesIO
        A bytes buffer containing a valid ZIP archive that holds at least
        one ``.shp`` file and its sibling component files.

    Returns
    -------
    gpd.GeoDataFrame | None
        The loaded GeoDataFrame, or ``None`` if no ``.shp`` was found or
        reading failed.
    """
    try:
        with zipfile.ZipFile(uploaded_zip, mode="r") as zf:
            shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]

            if not shp_names:
                logger.error("No .shp file found inside the uploaded ZIP.")
                return None

            target_shp = shp_names[0]
            logger.info("Found shapefile in ZIP: %s", target_shp)

            # Extract all entries that share the same base name (e.g. .shx, .dbf).
            base = target_shp.rsplit(".", 1)[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_real = os.path.realpath(tmpdir)
                for name in zf.namelist():
                    if name.rsplit(".", 1)[0] != base:
                        continue
                    # Reject absolute paths and any entry that would resolve
                    # outside tmpdir (zip-slip protection) before extracting.
                    dest_path = os.path.realpath(os.path.join(tmpdir, name))
                    if os.path.isabs(name) or not dest_path.startswith(tmpdir_real + os.sep):
                        logger.error(
                            "Refusing to extract unsafe zip entry: %r", name
                        )
                        continue
                    zf.extract(name, tmpdir)

                shp_path = str(tmpdir) + "/" + target_shp
                gdf = gpd.read_file(shp_path)

            logger.info(
                "Loaded GeoDataFrame: %d features, CRS=%s",
                len(gdf),
                gdf.crs,
            )
            return gdf

    except zipfile.BadZipFile:
        logger.error("Uploaded file is not a valid ZIP archive.")
        return None
    except Exception as exc:
        logger.error("Failed to load shapefile: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# load_raster_bytes
# ---------------------------------------------------------------------------
def load_raster_bytes(
    tif_bytes: bytes,
    file_name: str,
) -> tuple[MemoryFile, dict[str, Any]]:
    """Load a GeoTIFF from raw bytes into a :class:`rasterio.MemoryFile`.

    Parameters
    ----------
    tif_bytes : bytes
        Raw bytes of a GeoTIFF (.tif / .tiff) file.
    file_name : str
        Original file name (used only for logging).

    Returns
    -------
    tuple[MemoryFile, dict[str, Any]]
        A 2-tuple of ``(memfile, metadata)`` where *memfile* is an open
        :class:`rasterio.MemoryFile` and *metadata* is a dictionary with
        the following keys:

        * ``bands`` (*int*) — number of raster bands.
        * ``width`` (*int*) — pixel width.
        * ``height`` (*int*) — pixel height.
        * ``crs`` (*str | None*) — CRS as a PROJ string, or ``None``.
        * ``nodata`` (*float | None*) — nodata value, if defined.
        * ``dtype`` (*str*) — data-type string (e.g. ``"float32"``).
        * ``resolution`` (*tuple[float, float]*) — (x, y) pixel resolution.
        * ``transform`` (*Affine*) — the affine geotransform.
    """
    memfile = MemoryFile(tif_bytes)

    try:
        with memfile.open() as dst:
            resolution = (dst.res[0], dst.res[1])
            crs_str: str | None = dst.crs.to_string() if dst.crs else None

            metadata: dict[str, Any] = {
                "bands": dst.count,
                "width": dst.width,
                "height": dst.height,
                "crs": crs_str,
                "nodata": dst.nodata,
                "dtype": str(dst.dtypes[0]),
                "resolution": resolution,
                "transform": dst.transform,
            }

            logger.info(
                "Loaded raster '%s': %d band(s), %dx%d, CRS=%s, dtype=%s",
                file_name,
                dst.count,
                dst.width,
                dst.height,
                crs_str,
                dst.dtypes[0],
            )
    except Exception:
        # If we cannot open the dataset, close the memfile and re-raise.
        memfile.close()
        raise

    return memfile, metadata


# ---------------------------------------------------------------------------
# load_netcdf
# ---------------------------------------------------------------------------
def load_netcdf(
    nc_bytes: bytes,
    file_name: str,
    variable: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a NetCDF or Zarr file from raw bytes into an xarray DataArray.

    The function writes the bytes to a temporary file on disk and opens it
    with :func:`xarray.open_dataset` (which delegates to the netCDF4 or
    Zarr engine as appropriate).  If *variable* is ``None`` the module
    searches for a common precipitation variable name among the dataset's
    data variables.

    Parameters
    ----------
    nc_bytes : bytes
        Raw bytes of a NetCDF (``.nc``) or Zarr (``.zarr``) file.
    file_name : str
        Original file name, used for logging and engine-selection heuristics.
    variable : str | None, optional
        Name of the data variable to extract.  If ``None`` (default), the
        function tries common names: ``precip``, ``precipitation``, ``pr``,
        ``rainfall``, ``tp``.

    Returns
    -------
    tuple[xr.DataArray, dict[str, Any]]
        A 2-tuple of ``(data_array, metadata)``.  *metadata* contains:

        * ``variable`` (*str*) — the variable name that was selected.
        * ``dims`` (*list[str]*) — dimension names.
        * ``shape`` (*tuple[int, ...]*) — shape of the DataArray.
        * ``spatial_dims`` (*list[str]*) — identified spatial dims (``lat``/``y``,
          ``lon``/``x``).
        * ``time_dims`` (*list[str]*) — identified time dimension names.
        * ``crs`` (*str | None*) — CRS, if discoverable via ``rioxarray``.
        * ``size_mb`` (*float*) — approximate in-memory size in MiB.
    """
    import xarray as xr

    # Determine engine from extension.
    suffix = file_name.lower().split(".")[-1] if "." in file_name else ""
    engine: str | None = None
    if suffix == "zarr":
        engine = "zarr"
    elif suffix in ("nc", "nc4"):
        engine = "netcdf4"

    # Write bytes to a temp file so xarray can open it.
    tmp_suffix = f".{suffix}" if suffix else ".nc"
    tmp = tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False)
    try:
        tmp.write(nc_bytes)
        tmp.close()

        open_kwargs: dict[str, Any] = {}
        if engine is not None:
            open_kwargs["engine"] = engine

        ds = xr.open_dataset(tmp.name, **open_kwargs)

        # --- Variable selection ------------------------------------------------
        if variable is None:
            available = list(ds.data_vars)
            for candidate in _PRECIP_VARIABLES:
                if candidate in available:
                    variable = candidate
                    break
            if variable is None:
                # Fall back to the first data variable.
                variable = available[0] if available else ""
                logger.warning(
                    "No known precipitation variable found in '%s'; "
                    "falling back to '%s'. Available: %s",
                    file_name,
                    variable,
                    available,
                )
        if not variable or variable not in ds.data_vars:
            raise ValueError(
                f"Variable '{variable}' not found in dataset. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[variable]

        # --- Identify spatial and time dimensions -----------------------------
        all_dims = list(da.dims)
        spatial_dim_names: list[str] = []
        time_dim_names: list[str] = []

        lat_candidates = {"lat", "latitude", "y"}
        lon_candidates = {"lon", "longitude", "x"}
        time_candidates = {"time", "t", "valid_time", "forecast_time"}

        for dim in all_dims:
            if dim.lower() in lat_candidates or dim.lower() in lon_candidates:
                spatial_dim_names.append(dim)
            elif dim.lower() in time_candidates:
                time_dim_names.append(dim)

        # --- Attempt to extract CRS via rioxarray -----------------------------
        crs_str: str | None = None
        try:
            import rioxarray  # noqa: F401 — ensure extension is registered

            if da.rio.crs is not None:
                crs_str = da.rio.crs.to_string()
        except Exception as exc:
            logger.debug("Could not determine CRS via rioxarray: %s", exc)

        # --- Approximate in-memory size ---------------------------------------
        nbytes = int(da.nbytes) if hasattr(da, "nbytes") else 0
        size_mb = nbytes / (1024 ** 2)

        metadata: dict[str, Any] = {
            "variable": variable,
            "dims": all_dims,
            "shape": tuple(da.shape),
            "spatial_dims": spatial_dim_names,
            "time_dims": time_dim_names,
            "crs": crs_str,
            "size_mb": round(size_mb, 2),
        }

        logger.info(
            "Loaded NetCDF '%s': variable=%s, dims=%s, shape=%s, CRS=%s",
            file_name,
            variable,
            all_dims,
            da.shape,
            crs_str,
        )

        return da, metadata

    finally:
        # Clean up the temp file.
        try:
            import os

            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# compute_basin_area_km2
# ---------------------------------------------------------------------------
def compute_basin_area_km2(
    polygon: gpd.GeoDataFrame,
    equal_area_crs: str = EQUAL_AREA_CRS,
) -> float:
    """Compute the area of a basin polygon in square kilometres.

    The input GeoDataFrame is reprojected to an equal-area CRS so that the
    geometric area calculation is accurate regardless of the original
    coordinate reference system.  The default CRS is
    ``ESRI:54034`` (World Cylindrical Equal Area, metres).

    Parameters
    ----------
    polygon : gpd.GeoDataFrame
        A GeoDataFrame containing one or more polygon (or multipolygon)
        geometries representing the basin boundary.
    equal_area_crs : str, optional
        An equal-area CRS string used for the area calculation.
        Defaults to :pydata:`EQUAL_AREA_CRS` from the package config
        (``"ESRI:54034"`` — World Cylindrical Equal Area).

    Returns
    -------
    float
        Total basin area in km².  Returns ``0.0`` if the GeoDataFrame
        is empty or contains no valid geometries.
    """
    if polygon.empty or polygon.geometry.is_empty.all():
        logger.warning("Empty or all-null geometry passed to compute_basin_area_km2.")
        return 0.0

    source_crs = polygon.crs
    if source_crs is None:
        logger.warning("GeoDataFrame has no CRS; assuming EPSG:4326 (WGS 84).")
        polygon = polygon.set_crs("EPSG:4326")

    # Reproject to equal-area CRS for accurate area measurement.
    reprojected = polygon.to_crs(equal_area_crs)
    area_m2 = reprojected.geometry.area.sum()
    area_km2 = float(area_m2 / 1_000_000)

    logger.info(
        "Basin area: %.4f km² (source CRS=%s, equal-area CRS=%s)",
        area_km2,
        source_crs,
        equal_area_crs,
    )
    return area_km2
