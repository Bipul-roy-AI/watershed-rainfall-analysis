"""Area-weighted zonal statistics with ARF correction.

Provides hydrologically correct basin-rainfall estimation by:
1. Clipping a raster to the basin polygon (all_touched=True).
2. Masking out nodata, sentinel, and non-finite values.
3. Computing area-weighted descriptive statistics on valid pixels.
4. Applying an Areal Reduction Factor (ARF) to convert the
   spatial mean into a hydrologically representative basin rainfall.

References:
    Srikanthan, R. & McMahon, T.A. (2007). Stochastic generation
    of annual rainfall data. *Journal of Hydrology*, 228, 56-69.

    Reed, S. (1999). Flood estimation for ungauged catchments.
    USGS Water-Supply Paper 2375.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr

from watershed_analyzer.config import (
    DEFAULT_ARF_METHOD,
    DEFAULT_RAINFALL_UNIT,
    EQUAL_AREA_CRS,
    SENTINEL_VALUES,
)
from watershed_analyzer.core.arf import compute_arf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hydrologically_correct_zonal_stats(
    polygon: gpd.GeoDataFrame,
    raster_memfile: rasterio.MemoryFile,
    equal_area_crs: str = EQUAL_AREA_CRS,
    arf_method: str = DEFAULT_ARF_METHOD,
    rainfall_unit: str = DEFAULT_RAINFALL_UNIT,
) -> dict[str, Any]:
    """Compute area-weighted, ARF-corrected basin rainfall from a raster.

    Uses :func:`rasterio.mask.mask` to clip the raster to the polygon
    (``all_touched=True``), then computes area-weighted statistics on
    valid pixels only.

    Steps:
        1. Open the MemoryFile dataset.
        2. Reproject the polygon to the raster's CRS.
        3. Mask the raster to the polygon geometry.
        4. Build a valid-pixel mask (exclude nodata, sentinel values,
           and non-finite values).
        5. Compute pixel area from the raster geotransform.
        6. Compute spatial statistics on valid pixels.
        7. Compute basin area via reprojection to an equal-area CRS.
        8. Compute the ARF and derive basin rainfall + volume.

    Args:
        polygon: A single-row GeoDataFrame representing the basin boundary.
        raster_memfile: An in-memory raster (rasterio.MemoryFile) containing
            rainfall data.
        equal_area_crs: EPSG/ESRI authority string for an equal-area
            projection used to compute true basin area.  Defaults to
            ``ESRI:54034`` (World Cylindrical Equal Area).
        arf_method: Areal Reduction Factor method name.  See
            :func:`watershed_analyzer.core.arf.compute_arf` for options.
        rainfall_unit: Human-readable label for the rainfall unit
            (e.g. ``"mm/month"``).  Stored in the output dict for
            provenance; no unit conversion is performed.

    Returns:
        A dictionary containing:

        - ``n_valid_pixels`` (int) — count of valid pixels inside the polygon.
        - ``n_total_pixels`` (int) — total pixels touched by the polygon.
        - ``coverage_pct`` (float) — valid pixels / total pixels × 100.
        - ``spatial_mean_mm`` (float) — arithmetic mean of valid pixel values.
        - ``spatial_std_mm`` (float) — standard deviation of valid pixels.
        - ``spatial_cv_pct`` (float) — coefficient of variation (%).
        - ``min_mm`` (float) — minimum valid pixel value.
        - ``max_mm`` (float) — maximum valid pixel value.
        - ``median_mm`` (float) — median of valid pixel values.
        - ``q25_mm`` (float) — 25th percentile of valid pixel values.
        - ``q75_mm`` (float) — 75th percentile of valid pixel values.
        - ``basin_area_km2`` (float) — true basin area (equal-area CRS).
        - ``arf`` (float) — Areal Reduction Factor applied.
        - ``arf_method`` (str) — ARF method name used.
        - ``basin_rainfall_mm`` (float) — ARF-corrected basin rainfall.
        - ``volume_m3`` (float) — total rainfall volume over the basin.
        - ``pixel_area_m2`` (float) — area of one raster pixel.
        - ``rainfall_unit`` (str) — rainfall unit label.
        - ``equal_area_crs`` (str) — CRS used for area computation.

    Raises:
        ValueError: If the polygon is empty, the raster has no valid
            pixels, or CRS information is missing.

    References:
        Srikanthan, R. & McMahon, T.A. (2007). *Journal of Hydrology*, 228.
        Reed, S. (1999). *USGS Water-Supply Paper 2375*.
    """
    # ---- 1. Open dataset ----
    with raster_memfile.open() as dataset:
        if dataset.crs is None:
            raise ValueError("Raster CRS is None — cannot perform zonal stats.")
        raster_crs = dataset.crs
        nodata = dataset.nodata

        # ---- 2. Reproject polygon to raster CRS ----
        poly_reproj = polygon.to_crs(raster_crs)

        if poly_reproj.empty:
            raise ValueError("Polygon is empty after CRS reprojection.")

        geometries = poly_reproj.geometry.values.tolist()

        # ---- 3. Mask raster to polygon ----
        try:
            out_image, out_transform = rasterio.mask.mask(
                dataset,
                geometries,
                crop=True,
                all_touched=True,
            )
        except ValueError as exc:
            raise ValueError(
                f"rasterio.mask.mask failed — is the polygon valid? {exc}"
            ) from exc

        # Squeeze single-band rasters to 1-D
        if out_image.ndim == 3:
            out_image = out_image.squeeze(axis=0)

    # ---- 4. Build valid-pixel mask ----
    valid_mask = np.ones_like(out_image, dtype=bool)

    # Exclude nodata
    if nodata is not None:
        valid_mask &= out_image != float(nodata)

    # Exclude sentinel values from config
    for sentinel in SENTINEL_VALUES:
        valid_mask &= out_image != float(sentinel)

    # Exclude non-finite values (inf, nan)
    valid_mask &= np.isfinite(out_image)

    valid_values = out_image[valid_mask]
    n_valid_pixels = int(np.sum(valid_mask))
    n_total_pixels = int(out_image.size)

    if n_valid_pixels == 0:
        raise ValueError(
            "No valid pixels found inside the polygon after masking "
            "nodata/sentinel/non-finite values."
        )

    coverage_pct = (n_valid_pixels / n_total_pixels) * 100.0

    # ---- 5. Pixel area ----
    pixel_area_m2 = abs(out_transform.a * out_transform.e)

    # ---- 6. Spatial statistics ----
    spatial_mean_mm = float(np.mean(valid_values))
    spatial_std_mm = float(np.std(valid_values, ddof=0))
    spatial_cv_pct = (spatial_std_mm / spatial_mean_mm * 100.0) if spatial_mean_mm != 0 else 0.0
    min_mm = float(np.min(valid_values))
    max_mm = float(np.max(valid_values))
    median_mm = float(np.median(valid_values))
    q25_mm = float(np.percentile(valid_values, 25))
    q75_mm = float(np.percentile(valid_values, 75))

    # ---- 7. Basin area (equal-area CRS) ----
    basin_area_m2 = float(polygon.to_crs(equal_area_crs).geometry.area.sum())
    basin_area_km2 = basin_area_m2 / 1e6

    # ---- 8. ARF ----
    arf = compute_arf(basin_area_km2, method=arf_method)

    # ---- 9. Basin rainfall & volume ----
    basin_rainfall_mm = spatial_mean_mm * arf
    volume_m3 = basin_rainfall_mm * 1e-3 * basin_area_km2 * 1e6

    logger.info(
        "Zonal stats complete: %d valid/%d total pixels (%.1f%%), "
        "mean=%.2f mm, ARF=%.4f, basin rainfall=%.2f mm, "
        "volume=%.0f m³",
        n_valid_pixels,
        n_total_pixels,
        coverage_pct,
        spatial_mean_mm,
        arf,
        basin_rainfall_mm,
        volume_m3,
    )

    # ---- 10. Return everything ----
    return {
        "n_valid_pixels": n_valid_pixels,
        "n_total_pixels": n_total_pixels,
        "coverage_pct": round(coverage_pct, 4),
        "spatial_mean_mm": round(spatial_mean_mm, 6),
        "spatial_std_mm": round(spatial_std_mm, 6),
        "spatial_cv_pct": round(spatial_cv_pct, 4),
        "min_mm": round(min_mm, 6),
        "max_mm": round(max_mm, 6),
        "median_mm": round(median_mm, 6),
        "q25_mm": round(q25_mm, 6),
        "q75_mm": round(q75_mm, 6),
        "basin_area_km2": round(basin_area_km2, 6),
        "arf": round(arf, 6),
        "arf_method": arf_method,
        "basin_rainfall_mm": round(basin_rainfall_mm, 6),
        "volume_m3": round(volume_m3, 4),
        "pixel_area_m2": round(pixel_area_m2, 4),
        "rainfall_unit": rainfall_unit,
        "equal_area_crs": equal_area_crs,
    }


# ---------------------------------------------------------------------------
# NetCDF helper
# ---------------------------------------------------------------------------

def zonal_stats_from_netcdf(
    polygon: gpd.GeoDataFrame,
    data_array: xr.DataArray,
    equal_area_crs: str = EQUAL_AREA_CRS,
    arf_method: str = DEFAULT_ARF_METHOD,
    rainfall_unit: str = DEFAULT_RAINFALL_UNIT,
) -> dict[str, Any]:
    """Compute zonal statistics from an xarray DataArray backed by NetCDF.

    The DataArray is reprojected to the polygon's CRS using ``rioxarray``,
    then written to an in-memory raster so the same mask-based pipeline
    used by :func:`hydrologically_correct_zonal_stats` can be applied.

    Args:
        polygon: A single-row GeoDataFrame representing the basin boundary.
        data_array: An xarray DataArray with spatial coordinates (x, y)
            and a CRS set via the ``.rio`` accessor.
        equal_area_crs: EPSG/ESRI authority string for equal-area
            projection used to compute true basin area.
        arf_method: Areal Reduction Factor method name.
        rainfall_unit: Human-readable label for the rainfall unit.

    Returns:
        Same dictionary as :func:`hydrologically_correct_zonal_stats`.

    Raises:
        ValueError: If the DataArray lacks CRS information or spatial
            dimensions.

    References:
        Srikanthan, R. & McMahon, T.A. (2007). *Journal of Hydrology*, 228.
    """
    if not hasattr(data_array, "rio"):
        raise ValueError(
            "DataArray does not have a .rio accessor.  Ensure rioxarray is "
            "imported and the DataArray has spatial coordinates."
        )

    data_crs = data_array.rio.crs
    if data_crs is None:
        raise ValueError(
            "DataArray CRS is None — set it with data_array.rio.write_crs()"
        )

    poly_crs = polygon.crs
    if poly_crs is None:
        raise ValueError("Polygon CRS is None — cannot reproject DataArray.")

    # Reproject the DataArray to the polygon CRS if they differ
    if data_crs != poly_crs:
        logger.info(
            "Reprojecting DataArray from %s to polygon CRS %s",
            data_crs,
            poly_crs,
        )
        data_array = data_array.rio.reproject(poly_crs)

    # Write to an in-memory raster
    memfile = rasterio.MemoryFile()
    with memfile.open(
        driver="GTiff",
        width=int(data_array.rio.width),
        height=int(data_array.rio.height),
        count=1,
        dtype=str(data_array.dtype),
        crs=data_array.rio.crs,
        transform=data_array.rio.transform(),
        nodata=float(data_array.rio.nodata) if data_array.rio.nodata is not None else None,
    ) as dst:
        # data_array.values may be 2-D or 3-D; squeeze to 2-D
        arr = np.asarray(data_array)
        if arr.ndim == 3:
            arr = arr.squeeze(axis=0)
        dst.write(arr, 1)

    # Delegate to the main zonal stats function
    return hydrologically_correct_zonal_stats(
        polygon=polygon,
        raster_memfile=memfile,
        equal_area_crs=equal_area_crs,
        arf_method=arf_method,
        rainfall_unit=rainfall_unit,
    )
