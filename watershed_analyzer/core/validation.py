"""Raster validation and data-masking utilities for the Watershed Rainfall Analyzer.

This module provides four public helpers:

* ``validate_raster`` — inspect a rasterio MemoryFile and confirm it meets the
  minimum requirements for rainfall analysis (single-band, valid CRS, numeric
  dtype, non-empty data, sentinel-value detection, negative-value check).
* ``detect_rainfall_unit`` — infer or validate the rainfall unit associated
  with a raster, either from explicit user input or from raster metadata.
* ``check_raster_consistency`` — verify that a batch of rasters share the same
  resolution, CRS, and dtype so they can be compared or composited.
* ``mask_sentinel_values`` — replace both nodata and sentinel fill-values with
  ``np.nan`` and return a boolean mask of valid pixels.

All functions use Python 3.10+ union syntax (``X | None``) for type hints
and Google-style docstrings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import rasterio
from rasterio import MemoryFile

from watershed_analyzer.config import (
    DEFAULT_RAINFALL_UNIT,
    RAINFALL_UNITS,
    SENTINEL_VALUES,
)

logger = logging.getLogger(__name__)

# Numeric dtypes acceptable for rainfall rasters.
_NUMERIC_DTYPES: set[str] = {
    "int16", "int32", "int64",
    "uint8", "uint16", "uint32",
    "float16", "float32", "float64",
}

# Mapping of keyword fragments in raster descriptions to rainfall unit keys.
_UNIT_HINTS: dict[str, str] = {
    "hourly": "mm/hr",
    "hour": "mm/hr",
    "hr": "mm/hr",
    "daily": "mm/day",
    "day": "mm/day",
    "monthly": "mm/month",
    "month": "mm/month",
    "era5": "kg_m2_s",
    "kg m-2 s": "kg_m2_s",
    "kg/m2/s": "kg_m2_s",
}


# ---------------------------------------------------------------------------
# validate_raster
# ---------------------------------------------------------------------------
def validate_raster(
    memfile: MemoryFile,
    file_name: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate a single raster opened as a rasterio MemoryFile.

    Performs the following checks in order:

    1. **Single-band** — the raster must contain exactly one band.
    2. **Valid CRS** — the coordinate reference system must not be ``None``.
    3. **Numeric dtype** — the data type must be a recognised numeric type
       (integer or floating-point).
    4. **Not all NoData** — at least one pixel must hold a real value.
    5. **Sentinel-value detection** — scans the data for any of the
       sentinel values defined in :pydata:`SENTINEL_VALUES` and records
       the first match.
    6. **Negative-value check** — counts negative (non-sentinel) values
       and emits a warning because rainfall rasters should normally be
       non-negative.

    Parameters
    ----------
    memfile : rasterio.MemoryFile
        An already-constructed :class:`rasterio.MemoryFile`.  The function
        opens it internally via ``memfile.open()``.
    file_name : str
        Human-readable name of the file (used for log messages only).

    Returns
    -------
    tuple[bool, str, dict[str, Any] | None]
        A 3-tuple ``(is_valid, message, metadata)``.

        * *is_valid* — ``True`` when the raster passes every check.
        * *message* — a human-readable summary.  On success it contains
          the key metadata; on failure it explains the first failing check.
        * *metadata* — a dictionary (or ``None`` on early failure) with keys:
          ``bands``, ``width``, ``height``, ``crs``, ``nodata``, ``dtype``,
          ``resolution``, ``transform``, ``sentinel_detected``,
          ``sentinel_value``, ``negative_count``.
    """
    try:
        with memfile.open() as dst:
            bands = dst.count
            width = dst.width
            height = dst.height
            crs_str = dst.crs.to_string() if dst.crs else None
            nodata = dst.nodata
            dtype_str = str(dst.dtypes[0])
            resolution = (dst.res[0], dst.res[1])
            transform = dst.transform
            description = dst.descriptions[0] if dst.descriptions else ""

            # Read band 1 data.
            data = dst.read(1)

    except Exception as exc:
        msg = f"Failed to open raster '{file_name}': {exc}"
        logger.error(msg)
        return False, msg, None

    # ------------------------------------------------------------------
    # 1. Single-band check
    # ------------------------------------------------------------------
    if bands != 1:
        msg = (
            f"Raster '{file_name}' has {bands} band(s); "
            f"expected exactly 1 band for rainfall analysis."
        )
        logger.error(msg)
        return False, msg, None

    # ------------------------------------------------------------------
    # 2. CRS check
    # ------------------------------------------------------------------
    if crs_str is None:
        msg = f"Raster '{file_name}' has no CRS defined."
        logger.error(msg)
        return False, msg, None

    # ------------------------------------------------------------------
    # 3. Numeric dtype check
    # ------------------------------------------------------------------
    if dtype_str not in _NUMERIC_DTYPES:
        msg = (
            f"Raster '{file_name}' has dtype '{dtype_str}'; "
            f"expected one of {sorted(_NUMERIC_DTYPES)}."
        )
        logger.error(msg)
        return False, msg, None

    # ------------------------------------------------------------------
    # 4. All-NoData check
    # ------------------------------------------------------------------
    if nodata is not None:
        all_nodata = bool(np.all(data == nodata))
    else:
        all_nodata = bool(np.all(np.isnan(data.astype(np.float64))))

    if all_nodata:
        msg = f"Raster '{file_name}' is entirely NoData."
        logger.error(msg)
        return False, msg, None

    # ------------------------------------------------------------------
    # 5. Sentinel-value detection
    # ------------------------------------------------------------------
    sentinel_detected = False
    sentinel_value: float | None = None

    for sv in SENTINEL_VALUES:
        if np.any(data == sv):
            sentinel_detected = True
            sentinel_value = float(sv)
            logger.info(
                "Sentinel value %s detected in '%s'.",
                sv,
                file_name,
            )
            break  # report the first match only

    # ------------------------------------------------------------------
    # 6. Negative-value check (rainfall warning)
    # ------------------------------------------------------------------
    # Build a mask of values that are NOT nodata and NOT any sentinel.
    valid_mask = np.ones_like(data, dtype=bool)
    if nodata is not None:
        valid_mask &= data != nodata
    for sv in SENTINEL_VALUES:
        valid_mask &= data != sv

    negative_count = int(np.sum(data[valid_mask] < 0))
    if negative_count > 0:
        logger.warning(
            "Raster '%s' contains %d negative value(s) outside "
            "nodata/sentinel pixels. Rainfall data is normally non-negative.",
            file_name,
            negative_count,
        )

    # ------------------------------------------------------------------
    # Build metadata dict
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "bands": bands,
        "width": width,
        "height": height,
        "crs": crs_str,
        "nodata": nodata,
        "dtype": dtype_str,
        "resolution": resolution,
        "transform": transform,
        "sentinel_detected": sentinel_detected,
        "sentinel_value": sentinel_value,
        "negative_count": negative_count,
        "description": description,
    }

    msg = (
        f"Raster '{file_name}' passed validation: {width}x{height}, "
        f"CRS={crs_str}, dtype={dtype_str}, nodata={nodata}, "
        f"sentinel={sentinel_value}, negatives={negative_count}."
    )
    logger.info(msg)
    return True, msg, metadata


# ---------------------------------------------------------------------------
# detect_rainfall_unit
# ---------------------------------------------------------------------------
def detect_rainfall_unit(
    metadata: dict[str, Any],
    user_selection: str | None = None,
) -> str:
    """Determine the rainfall unit for a raster.

    If *user_selection* is provided it is validated against the keys of
    :pydata:`RAINFALL_UNITS` and returned directly.  Otherwise the function
    attempts to infer the unit from the raster's ``description`` metadata
    field by searching for known keyword fragments (e.g. ``"hourly"`` →
    ``"mm/hr"``).  If no hint is found the function falls back to
    :pydata:`DEFAULT_RAINFALL_UNIT`.

    Parameters
    ----------
    metadata : dict[str, Any]
        A raster metadata dictionary such as the one returned by
        :func:`validate_raster`.  The ``description`` key (case-insensitive
        string) is used for heuristic inference when *user_selection* is
        ``None``.
    user_selection : str | None, optional
        Explicit unit key chosen by the user.  Must match one of the keys
        in :pydata:`RAINFALL_UNITS`.  If ``None`` (default), inference is
        attempted.

    Returns
    -------
    str
        The resolved rainfall unit key (e.g. ``"mm/month"``, ``"mm/hr"``).

    Raises
    ------
    ValueError
        If *user_selection* is non-empty but is not a recognised key in
        :pydata:`RAINFALL_UNITS`.
    """
    # --- Explicit user selection ----------------------------------------
    if user_selection is not None:
        user_selection = user_selection.strip()
        if user_selection not in RAINFALL_UNITS:
            raise ValueError(
                f"Unknown rainfall unit '{user_selection}'. "
                f"Valid options: {sorted(RAINFALL_UNITS.keys())}"
            )
        logger.info("Using user-specified rainfall unit: '%s'.", user_selection)
        return user_selection

    # --- Heuristic inference from raster description --------------------
    description = str(metadata.get("description", "")).lower()
    for hint, unit_key in _UNIT_HINTS.items():
        if hint in description:
            logger.info(
                "Inferred rainfall unit '%s' from description: '%s'.",
                unit_key,
                metadata.get("description", ""),
            )
            return unit_key

    # --- Fallback -------------------------------------------------------
    logger.info(
        "Could not infer rainfall unit; defaulting to '%s'.",
        DEFAULT_RAINFALL_UNIT,
    )
    return DEFAULT_RAINFALL_UNIT


# ---------------------------------------------------------------------------
# check_raster_consistency
# ---------------------------------------------------------------------------
def check_raster_consistency(
    metadata_list: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Check that a batch of rasters share the same resolution, CRS, and dtype.

    This is a lightweight consistency gate that should be called before
    compositing, differencing, or time-series analysis of multiple rasters.
    Only the first raster's properties are used as the reference.

    Parameters
    ----------
    metadata_list : list[dict[str, Any]]
        A list of metadata dictionaries (as returned by :func:`validate_raster`).
        Must contain at least two entries; a single-element list is
        trivially consistent.

    Returns
    -------
    tuple[bool, str]
        ``(is_consistent, message)``.  ``is_consistent`` is ``True`` when
        every raster matches the reference on resolution, CRS, and dtype.
    """
    n = len(metadata_list)
    if n < 2:
        return True, "Only one raster provided; consistency check is trivially satisfied."

    ref = metadata_list[0]
    ref_res = ref.get("resolution")
    ref_crs = ref.get("crs")
    ref_dtype = ref.get("dtype")

    for idx, meta in enumerate(metadata_list[1:], start=2):
        # Resolution
        if meta.get("resolution") != ref_res:
            msg = (
                f"Resolution mismatch: raster #{idx} has {meta.get('resolution')}, "
                f"expected {ref_res} (raster #1)."
            )
            logger.error(msg)
            return False, msg

        # CRS
        if meta.get("crs") != ref_crs:
            msg = (
                f"CRS mismatch: raster #{idx} has '{meta.get('crs')}', "
                f"expected '{ref_crs}' (raster #1)."
            )
            logger.error(msg)
            return False, msg

        # Dtype
        if meta.get("dtype") != ref_dtype:
            msg = (
                f"Dtype mismatch: raster #{idx} has '{meta.get('dtype')}', "
                f"expected '{ref_dtype}' (raster #1)."
            )
            logger.error(msg)
            return False, msg

    msg = (
        f"All {n} rasters are consistent: resolution={ref_res}, "
        f"CRS={ref_crs}, dtype={ref_dtype}."
    )
    logger.info(msg)
    return True, msg


# ---------------------------------------------------------------------------
# mask_sentinel_values
# ---------------------------------------------------------------------------
def mask_sentinel_values(
    data: np.ndarray,
    nodata: float | None = None,
    sentinel_values: tuple[float, ...] = SENTINEL_VALUES,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace nodata and sentinel fill-values with ``np.nan``.

    The function builds a boolean *valid mask* that is ``True`` for pixels
    whose value is **not** nodata and **not** any of the supplied sentinel
    values.  It then returns a copy of *data* where invalid pixels are
    replaced by ``np.nan``.

    Parameters
    ----------
    data : np.ndarray
        2-D raster band data (or any numpy array).
    nodata : float | None, optional
        The raster's official nodata value.  If ``None``, nodata masking
        is skipped (only sentinel values are masked).  Default is ``None``.
    sentinel_values : tuple[float, ...], optional
        A tuple of sentinel fill-values to mask.  Defaults to
        :pydata:`SENTINEL_VALUES` from the package config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A 2-tuple ``(masked_data, valid_mask)`` where:

        * *masked_data* — a float64 copy of *data* with all invalid pixels
          set to ``np.nan``.
        * *valid_mask* — a boolean array of the same shape, ``True`` where
          the original pixel was valid (not nodata, not a sentinel value).
    """
    # Work on a float64 copy so we can safely store np.nan.
    masked = data.astype(np.float64, copy=True)

    # Start with everything valid.
    valid_mask = np.ones_like(masked, dtype=bool)

    # Mask nodata.
    if nodata is not None:
        nodata_mask = masked == float(nodata)
        valid_mask &= ~nodata_mask
        masked[nodata_mask] = np.nan
        logger.debug(
            "Masked %d nodata pixel(s) (nodata=%s).",
            int(nodata_mask.sum()),
            nodata,
        )

    # Mask each sentinel value.
    for sv in sentinel_values:
        sv_mask = masked == float(sv)
        count = int(sv_mask.sum())
        if count > 0:
            valid_mask &= ~sv_mask
            masked[sv_mask] = np.nan
            logger.debug(
                "Masked %d sentinel pixel(s) (sentinel=%s).",
                count,
                sv,
            )

    logger.info(
        "mask_sentinel_values: %d / %d pixels valid (%.1f%%).",
        int(valid_mask.sum()),
        valid_mask.size,
        100.0 * valid_mask.sum() / max(valid_mask.size, 1),
    )
    return masked, valid_mask
