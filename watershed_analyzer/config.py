"""Global configuration constants for the Watershed Rainfall Analyzer.

All magic numbers, default CRS strings, and behavioural flags live here
so they can be audited and tuned in a single location.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tool identity
# ---------------------------------------------------------------------------
TOOL_NAME: str = "watershed-rainfall-analyzer"
TOOL_VERSION: str = "3.0.0"

# ---------------------------------------------------------------------------
# Sentinel / NoData values commonly found in geospatial rasters
# ---------------------------------------------------------------------------
SENTINEL_VALUES: tuple[float, ...] = (-9999.0, -32768.0, -9999.0, -999.0)

# ---------------------------------------------------------------------------
# CRS defaults
# ---------------------------------------------------------------------------
EQUAL_AREA_CRS: str = "ESRI:54034"  # World Cylindrical Equal Area (m²)

# ---------------------------------------------------------------------------
# ARF (Areal Reduction Factor) parameters
#
# IMPORTANT: ARF converts POINT-GAUGE rainfall to an areal estimate — it
# corrects for the fact that a single rain gauge overestimates the true
# spatial-average rainfall as basin area grows. It does NOT apply to
# already-gridded/satellite rainfall products (CHIRPS, ERA5, IMERG, etc.),
# where each pixel is already an areal estimate; applying ARF on top of a
# spatial mean of such data double-corrects and biases results low. The
# lookup-table methods are also built for sub-daily storm-design durations,
# not monthly accumulations, so there is no valid duration to select for
# the monthly use case this tool targets.
#
# Only enable ARF if you are specifically working with point-gauge data
# interpolated onto a grid, and can justify the accumulation duration.
# ---------------------------------------------------------------------------
ARF_METHODS: dict[str, str] = {
    "srikanthan_mcmahon": (
        "Srikanthan & McMahon (2007): ARF = 1 - 0.04 x A^0.4 "
        "(point-gauge-to-areal correction — NOT for gridded rainfall)"
    ),
    "usgs_reed": (
        "USGS Reed (1999): area-duration-frequency based "
        "(sub-daily storm design — NOT for monthly gridded rainfall)"
    ),
    "none": "No areal reduction applied (raw grid-cell average) — correct default for gridded rainfall products",
}
DEFAULT_ARF_METHOD: str = "none"

# ---------------------------------------------------------------------------
# Rainfall unit labels
# ---------------------------------------------------------------------------
RAINFALL_UNITS: dict[str, str] = {
    "mm/month": "Monthly accumulated rainfall (mm)",
    "mm/day": "Daily rainfall depth (mm)",
    "mm/hr": "Hourly rainfall intensity (mm hr⁻¹)",
    "kg_m2_s": "ERA5 format — kg m⁻² s⁻¹ (convert to mm/day × 86400)",
}
DEFAULT_RAINFALL_UNIT: str = "mm/month"

# ---------------------------------------------------------------------------
# SPI (Standardized Precipitation Index) scales
# ---------------------------------------------------------------------------
SPI_SCALES: list[int] = [1, 3, 6, 9, 12, 24, 36, 48]
DEFAULT_SPI_SCALE: int = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# CSV provenance header template
# ---------------------------------------------------------------------------
PROVENANCE_TEMPLATE: str = """# Tool: {tool_name} v{tool_version}
# Run timestamp: {timestamp}
# Shapefile: {shapefile_name}, CRS: {shapefile_crs}, n_features: {n_features}
# Selected region: {region_name}
# Basin area (km²): {basin_area_km2:.4f}
# Rasters: {raster_names}
# Zonal method: area-weighted mask, all_touched=True
# ARF method: {arf_method} (ARF={arf_value:.4f})
# Rainfall unit: {rainfall_unit}
# Equal-area CRS for area computation: {equal_area_crs}
#"""
