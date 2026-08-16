"""Digital Elevation Model (DEM) processing for watershed analysis.

Provides three core functions:

1. **process_dem** — Read a DEM from a :class:`rasterio.MemoryFile`,
   optionally clip to a polygon, and compute summary elevation statistics.

2. **compute_elevation_rainfall_regression** — Simple linear regression of
   rainfall against elevation using :func:`scipy.stats.linregress`, returning
   slope, significance, and an orographic interpretation.

3. **compute_twi** — Compute the Topographic Wetness Index (TWI) using a
   self-contained D8 flow-accumulation algorithm (no external hydrology
   library required).

All functions use Python 3.10+ type hints, Google-style docstrings,
and structured logging via the ``logging`` stdlib module.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from scipy.stats import linregress

from watershed_analyzer.config import SENTINEL_VALUES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DEM processing & statistics
# ---------------------------------------------------------------------------


def process_dem(
    memfile: rasterio.MemoryFile,
    polygon: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    """Open a DEM raster, apply masks, optionally clip, and compute statistics.

    Reads band 1 from the in-memory raster, masks out nodata cells and
    :pydata:`SENTINEL_VALUES`, and (optionally) clips the raster to the
    provided polygon using :func:`rasterio.mask.mask` with
    ``crop=True, all_touched=True``.

    Parameters
    ----------
    memfile:
        An in-memory raster file containing the DEM (elevation data in
        band 1).
    polygon:
        Optional :class:`geopandas.GeoDataFrame` whose geometry is used
        to clip the raster.  When ``None`` the full raster extent is
        used.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:

        - ``'min_elevation'``, ``'max_elevation'``, ``'mean_elevation'``,
          ``'elevation_range'``, ``'std_elevation'``, ``'median_elevation'``,
          ``'q25_elevation'``, ``'q75_elevation'`` — summary statistics
          over valid pixels (float).
        - ``'crs'`` — CRS string of the source raster.
        - ``'resolution'`` — Tuple ``(pixel_width, pixel_height)`` in the
          raster's coordinate units.
        - ``'n_valid_pixels'`` — Number of valid (unmasked) pixels.

    Raises
    ------
    ValueError
        If band 1 cannot be read or if no valid pixels remain after
        masking.
    """
    with memfile.open() as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        # --- Optional polygon clipping ---
        if polygon is not None:
            logger.info(
                "Clipping DEM to polygon (%d feature(s)).", len(polygon)
            )
            geometries = polygon.geometry.values.tolist()
            try:
                dem_data, transform = rio_mask(
                    src, geometries, crop=True, all_touched=True
                )
            except ValueError as exc:
                raise ValueError(
                    "Polygon clipping failed — check that the polygon "
                    "geometry intersects the DEM extent."
                ) from exc
            # rio_mask returns (1, rows, cols); squeeze to 2-D
            dem_data = dem_data.squeeze(axis=0)
            logger.info(
                "Clipped DEM shape: %s, new transform: %s",
                dem_data.shape,
                transform,
            )

        # --- Build validity mask ---
        valid_mask = np.ones_like(dem_data, dtype=bool)

        # Mask nodata
        if nodata is not None:
            valid_mask &= dem_data != nodata
            logger.debug("Applied nodata mask (nodata=%s).", nodata)

        # Mask sentinel values from config
        for sentinel in SENTINEL_VALUES:
            valid_mask &= dem_data != sentinel
        logger.debug(
            "Applied sentinel mask for %d sentinel values.",
            len(SENTINEL_VALUES),
        )

        # Mask NaN / inf
        valid_mask &= np.isfinite(dem_data)

        valid_pixels = dem_data[valid_mask]
        n_valid = valid_pixels.size

        if n_valid == 0:
            raise ValueError(
                "No valid DEM pixels remain after applying nodata, "
                "sentinel, and finiteness masks."
            )

        # --- Compute statistics ---
        result: dict[str, Any] = {
            "min_elevation": float(np.min(valid_pixels)),
            "max_elevation": float(np.max(valid_pixels)),
            "mean_elevation": float(np.mean(valid_pixels)),
            "elevation_range": float(np.ptp(valid_pixels)),
            "std_elevation": float(np.std(valid_pixels, ddof=1)),
            "median_elevation": float(np.median(valid_pixels)),
            "q25_elevation": float(np.percentile(valid_pixels, 25)),
            "q75_elevation": float(np.percentile(valid_pixels, 75)),
            "crs": str(crs) if crs is not None else None,
            "resolution": (abs(transform.a), abs(transform.e)),
            "n_valid_pixels": int(n_valid),
        }

        logger.info(
            "DEM statistics: min=%.2f  max=%.2f  mean=%.2f  range=%.2f  "
            "std=%.2f  median=%.2f  n_valid=%d  crs=%s",
            result["min_elevation"],
            result["max_elevation"],
            result["mean_elevation"],
            result["elevation_range"],
            result["std_elevation"],
            result["median_elevation"],
            n_valid,
            result["crs"],
        )

    return result


# ---------------------------------------------------------------------------
# 2. Elevation–rainfall regression
# ---------------------------------------------------------------------------


def compute_elevation_rainfall_regression(
    elevation_data: np.ndarray,
    rainfall_data: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Simple linear regression of rainfall against elevation.

    Uses :func:`scipy.stats.linregress` to fit
    ``rainfall = slope × elevation + intercept`` over pixels where
    *valid_mask* is ``True``.  Results include regression coefficients,
    significance, and a qualitative orographic interpretation.

    Parameters
    ----------
    elevation_data:
        2-D array of elevation values (metres).
    rainfall_data:
        2-D array of rainfall values (mm) with the same shape as
        *elevation_data*.
    valid_mask:
        Boolean 2-D array with the same shape indicating which pixels
        to include in the regression.

    Returns
    -------
    dict[str, Any]
        - ``'slope'`` — Regression slope (mm m⁻¹).
        - ``'intercept'`` — Y-intercept (mm).
        - ``'r_value'`` — Pearson correlation coefficient.
        - ``'r_squared'`` — Coefficient of determination (R²).
        - ``'p_value'`` — Two-tailed *p*-value for the slope.
        - ``'std_err'`` — Standard error of the slope estimate.
        - ``'lapse_rate_mm_per_100m'`` — ``slope × 100``.
        - ``'interpretation'`` — Qualitative description (str).
        - ``'n_pairs'`` — Number of data points used.

    Raises
    ------
    ValueError
        If fewer than 3 valid data points are available, or if the input
        arrays have mismatched shapes.
    """
    if elevation_data.shape != rainfall_data.shape:
        raise ValueError(
            f"Shape mismatch: elevation {elevation_data.shape} vs "
            f"rainfall {rainfall_data.shape}."
        )
    if elevation_data.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch: elevation {elevation_data.shape} vs "
            f"mask {valid_mask.shape}."
        )

    elev = elevation_data[valid_mask].ravel().astype(np.float64)
    rain = rainfall_data[valid_mask].ravel().astype(np.float64)

    # Drop NaN / inf from either array
    both_valid = np.isfinite(elev) & np.isfinite(rain)
    elev = elev[both_valid]
    rain = rain[both_valid]

    n_pairs = len(elev)
    if n_pairs < 3:
        raise ValueError(
            f"Need at least 3 valid (elevation, rainfall) pairs for "
            f"regression; got {n_pairs}."
        )

    result = linregress(elev, rain)

    slope = float(result.slope)
    r_value = float(result.rvalue)
    r_squared = r_value**2
    p_value = float(result.pvalue)
    std_err = float(result.stderr)
    intercept = float(result.intercept)
    lapse_rate = slope * 100.0

    # --- Qualitative interpretation ---
    if slope > 0 and p_value < 0.05:
        interpretation = (
            "orographic enhancement — rainfall increases significantly "
            "with elevation"
        )
    elif slope < 0 and p_value < 0.05:
        interpretation = (
            "rain shadow effect — rainfall decreases significantly "
            "with elevation"
        )
    elif slope > 0 and p_value >= 0.05:
        interpretation = (
            "weak orographic enhancement — positive slope but "
            "not statistically significant"
        )
    elif slope < 0 and p_value >= 0.05:
        interpretation = (
            "weak rain shadow — negative slope but not statistically "
            "significant"
        )
    else:
        interpretation = "no meaningful elevation–rainfall relationship"

    output: dict[str, Any] = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err": std_err,
        "lapse_rate_mm_per_100m": lapse_rate,
        "interpretation": interpretation,
        "n_pairs": int(n_pairs),
    }

    logger.info(
        "Elevation–rainfall regression: slope=%.4f mm/m  R²=%.4f  "
        "p=%.6f  lapse_rate=%.2f mm/100m  n=%d",
        slope,
        r_squared,
        p_value,
        lapse_rate,
        n_pairs,
    )
    logger.info("  interpretation: %s", interpretation)

    return output


# ---------------------------------------------------------------------------
# 3. Topographic Wetness Index (TWI)
# ---------------------------------------------------------------------------

# D8 direction codes (ArcGIS convention): (row_offset, col_offset, distance, code)
_D8_DIRS: list[tuple[int, int, float, int]] = [
    (-1, 0, 1.0, 64),  # N
    (-1, 1, np.sqrt(2), 128),  # NE
    (0, 1, 1.0, 1),  # E
    (1, 1, np.sqrt(2), 2),  # SE
    (1, 0, 1.0, 4),  # S
    (1, -1, np.sqrt(2), 8),  # SW
    (0, -1, 1.0, 16),  # W
    (-1, -1, np.sqrt(2), 32),  # NW
]

# Reverse lookup: direction code → (dr, dc)
_CODE_TO_OFFSET: dict[int, tuple[int, int]] = {
    64: (-1, 0),
    128: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
}

# 8-connected neighbour offsets for pit-filling neighbourhood scans
_NEIGHBOUR_OFFSETS: list[tuple[int, int]] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _fill_pits(
    dem: np.ndarray,
    valid_mask: np.ndarray,
    max_iter: int = 500,
) -> np.ndarray:
    """Fill depressions in a DEM by raising pits to the minimum neighbour.

    A cell is considered a *pit* when its elevation is strictly lower
    than every valid neighbour.  Such cells are raised to the elevation
    of their lowest neighbour and the process repeats until stable.

    This is a simple, dependency-light approach suitable for
    research-grade analysis.  It does **not** implement the full
    Priority-Flood algorithm and may not converge for DEMs with
    extensive flat areas.

    Parameters
    ----------
    dem:
        2-D elevation array (float).
    valid_mask:
        Boolean mask of valid (non-nodata) cells.
    max_iter:
        Maximum number of filling iterations before giving up.

    Returns
    -------
    numpy.ndarray
        A copy of *dem* with depressions filled.
    """
    filled = dem.copy().astype(np.float64)
    rows, cols = filled.shape

    # Pre-compute a padded valid mask (constant beyond edges → edge cells
    # have fewer valid neighbours, which is correct).
    padded_valid = np.pad(
        valid_mask, 1, mode="constant", constant_values=False
    )

    for iteration in range(max_iter):
        padded = np.pad(filled, 1, mode="edge")

        # Minimum valid-neighbour elevation for every cell
        neighbour_min = np.full(
            (rows, cols), np.inf, dtype=np.float64
        )
        has_valid_neighbour = np.zeros((rows, cols), dtype=bool)

        for dr, dc in _NEIGHBOUR_OFFSETS:
            n_vals = padded[1 + dr : rows + 1 + dr, 1 + dc : cols + 1 + dc]
            n_mask = padded_valid[
                1 + dr : rows + 1 + dr, 1 + dc : cols + 1 + dc
            ]
            masked = np.where(n_mask, n_vals, np.inf)
            better = masked < neighbour_min
            neighbour_min = np.where(better, masked, neighbour_min)
            has_valid_neighbour |= n_mask

        # Strict pits: valid cell with valid neighbours that is strictly
        # lower than every one of them.
        is_pit = (
            valid_mask
            & has_valid_neighbour
            & (filled < neighbour_min)
            & np.isfinite(neighbour_min)
        )

        n_pits = int(np.sum(is_pit))
        if n_pits == 0:
            logger.debug(
                "Pit filling converged after %d iteration(s).",
                iteration + 1,
            )
            break

        filled[is_pit] = neighbour_min[is_pit]
    else:
        # Loop exhausted without convergence
        remaining = int(np.sum(is_pit))  # type: ignore[possibly-undefined]
        logger.warning(
            "Pit filling did not fully converge within %d iterations "
            "(%d pit(s) remain).",
            max_iter,
            remaining,
        )

    return filled


def _d8_flow_accumulation(
    dem: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute D8 flow accumulation on a pit-filled DEM.

    Each cell flows to the neighbour with the steepest downhill slope.
    Flow accumulation is computed by sorting cells by elevation
    (descending) and propagating each cell's accumulation to its
    downstream neighbour.  Cells with no valid downhill path are
    treated as outlets (accumulation is not propagated further).

    Parameters
    ----------
    dem:
        2-D elevation array (pits should already be filled).
    valid_mask:
        Boolean mask of valid cells.

    Returns
    -------
    numpy.ndarray
        2-D float64 array of flow-accumulation values (unit count
        including the cell itself).  Invalid cells are set to 0.
    """
    rows, cols = dem.shape

    # --- Flow direction (steepest-descent D8) ---
    padded = np.pad(dem, 1, mode="constant", constant_values=np.inf)
    padded_valid = np.pad(
        valid_mask, 1, mode="constant", constant_values=False
    )

    flow_dir = np.zeros((rows, cols), dtype=np.uint8)
    max_slope = np.full((rows, cols), -np.inf, dtype=np.float64)

    for dr, dc, dist, code in _D8_DIRS:
        n_vals = padded[1 + dr : rows + 1 + dr, 1 + dc : cols + 1 + dc]
        n_valid = padded_valid[
            1 + dr : rows + 1 + dr, 1 + dc : cols + 1 + dc
        ]
        # Slope = (current - neighbour) / distance; positive = downhill
        slope = (dem - n_vals) / dist
        slope = np.where(n_valid, slope, -np.inf)
        better = slope > max_slope
        flow_dir = np.where(better, np.uint8(code), flow_dir)
        max_slope = np.where(better, slope, max_slope)

    # Cells with no downhill path → code 0 (outlet / flat)
    flow_dir[max_slope <= 0] = 0

    # --- Flow accumulation (descending elevation sort) ---
    accum = np.zeros((rows, cols), dtype=np.float64)
    valid_indices = np.argwhere(valid_mask)

    if valid_indices.size == 0:
        return accum

    elevations = dem[valid_indices[:, 0], valid_indices[:, 1]]
    sort_order = np.argsort(-elevations)  # descending

    # Every valid cell starts with an accumulation of 1 (itself)
    accum[valid_indices[:, 0], valid_indices[:, 1]] = 1.0

    for idx in sort_order:
        r = int(valid_indices[idx, 0])
        c = int(valid_indices[idx, 1])
        fd = int(flow_dir[r, c])
        if fd == 0:
            continue  # outlet — do not propagate
        dr, dc = _CODE_TO_OFFSET[fd]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and valid_mask[nr, nc]:
            accum[nr, nc] += accum[r, c]

    return accum


def compute_twi(
    dem_data: np.ndarray,
    transform: rasterio.transform.Affine,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute the Topographic Wetness Index (TWI) from a DEM.

    TWI is defined as:

    .. math::

        TWI = \\ln\\!\\left(\\frac{a}{\\tan\\beta}\\right)

    where *a* is the specific catchment area and *β* is the local slope
    in radians.

    Implementation details:

    * **Slope** is derived from the DEM using :func:`numpy.gradient`.
    * **Catchment area** is computed via a self-contained D8
      flow-accumulation algorithm (pit filling + steepest-descent
      routing).  The result is converted from cell counts to metres
      using the pixel resolution.
    * **tan(β)** is clamped to a minimum of 0.01 to avoid division by
      near-zero slopes.

    Parameters
    ----------
    dem_data:
        2-D array of elevation values (metres).
    transform:
        Affine geotransform of the raster (used for pixel spacing).
    valid_mask:
        Boolean 2-D array; ``True`` marks pixels to include in the
        computation.

    Returns
    -------
    dict[str, Any]
        - ``'twi_mean'`` — Mean TWI over valid cells (float).
        - ``'twi_std'`` — Standard deviation (float, ddof=1).
        - ``'twi_min'`` — Minimum (float).
        - ``'twi_max'`` — Maximum (float).
        - ``'twi_median'`` — Median (float).
        - ``'twi_array'`` — Full 2-D TWI raster (invalid cells = NaN).

    Raises
    ------
    ValueError
        If no valid pixels are provided or if the DEM shape is
        inconsistent with the mask.
    """
    if dem_data.shape != valid_mask.shape:
        raise ValueError(
            f"Shape mismatch: DEM {dem_data.shape} vs "
            f"mask {valid_mask.shape}."
        )

    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise ValueError(
            "No valid pixels in the DEM mask for TWI computation."
        )

    # ------------------------------------------------------------------
    # 1. Slope via numpy.gradient
    # ------------------------------------------------------------------
    cell_width = abs(transform.a)  # |dx|
    cell_height = abs(transform.e)  # |dy| (negative by GeoTIFF convention)

    dy, dx = np.gradient(dem_data, cell_height, cell_width)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    tan_beta = np.tan(slope_rad)
    tan_beta = np.clip(tan_beta, 0.01, None)  # avoid ÷0

    logger.debug(
        "Slope computed: tan_beta range [%.6f, %.6f].",
        float(np.min(tan_beta[valid_mask])),
        float(np.max(tan_beta[valid_mask])),
    )

    # ------------------------------------------------------------------
    # 2. D8 flow accumulation (pit-fill → steepest descent → accumulate)
    # ------------------------------------------------------------------
    logger.info("Starting pit filling and D8 flow accumulation …")
    filled_dem = _fill_pits(dem_data, valid_mask)
    flow_accum = _d8_flow_accumulation(filled_dem, valid_mask)
    logger.info("D8 flow accumulation complete.")

    # Specific catchment area *a*: convert cell-count accumulation to metres.
    # For a square cell: a = flow_accum × cell_size.
    # For a rectangular cell we use the mean of width and height.
    cell_size = (cell_width + cell_height) / 2.0
    a = flow_accum * cell_size  # metres

    # ------------------------------------------------------------------
    # 3. TWI = ln(a / tan β)
    # ------------------------------------------------------------------
    twi = np.full_like(dem_data, np.nan, dtype=np.float64)
    twi[valid_mask] = np.log(a[valid_mask] / tan_beta[valid_mask])

    twi_valid = twi[valid_mask]

    result: dict[str, Any] = {
        "twi_mean": float(np.mean(twi_valid)),
        "twi_std": float(np.std(twi_valid, ddof=1)),
        "twi_min": float(np.min(twi_valid)),
        "twi_max": float(np.max(twi_valid)),
        "twi_median": float(np.median(twi_valid)),
        "twi_array": twi,
    }

    logger.info(
        "TWI computed: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  median=%.4f",
        result["twi_mean"],
        result["twi_std"],
        result["twi_min"],
        result["twi_max"],
        result["twi_median"],
    )

    return result
