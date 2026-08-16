"""Streamlit upload and configuration components for the Watershed Analyzer UI.

This module provides all user-input widgets that appear in the sidebar and
main area of the Streamlit application.  Each function renders one logical
group of controls and returns the user's selection so the calling page can
wire it into the analysis pipeline.

Functions:
    render_shapefile_upload: ZIP file uploader for shapefile packages.
    render_region_selector: Two-step column / value selector for a GeoDataFrame.
    render_dem_upload: Optional DEM raster uploader (GeoTIFF).
    render_raster_upload: Multi-file rainfall raster uploader (GeoTIFF).
    render_netcdf_upload: Optional NetCDF uploader.
    render_analysis_settings: Expander with ARF, units, and SPI scale settings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import geopandas as gpd
    import streamlit as st
    from streamlit.runtime.uploaded_file_manager import UploadedFile

from watershed_analyzer.config import (
    ARF_METHODS,
    DEFAULT_ARF_METHOD,
    DEFAULT_RAINFALL_UNIT,
    DEFAULT_SPI_SCALE,
    RAINFALL_UNITS,
    SPI_SCALES,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_shapefile_upload() -> UploadedFile | None:
    """Render a file uploader for a zipped shapefile package.

    The widget accepts ``.zip`` files only.  A help tooltip informs the
    user that the archive must contain at least ``.shp``, ``.shx``, and
    ``.dbf`` components (and optionally ``.prj`` for CRS information).

    Returns:
        The uploaded ``UploadedFile`` object, or ``None`` if nothing has
        been uploaded yet.
    """
    import streamlit as st

    logger.debug("Rendering shapefile upload widget")
    uploaded = st.file_uploader(
        label="📁 Upload Shapefile (ZIP)",
        type=["zip"],
        key="shapefile_upload",
        help=(
            "Upload a ZIP archive containing the shapefile components: "
            "`.shp`, `.shx`, `.dbf` (required), and `.prj` (recommended)."
        ),
    )
    if uploaded is not None:
        logger.info("Shapefile uploaded: %s (%d bytes)", uploaded.name, uploaded.size)
    return uploaded


def render_region_selector(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, str, str]:
    """Render a two-step region selector from a GeoDataFrame.

    The first ``st.selectbox`` lets the user choose an attribute column
    (all columns except ``'geometry'`` are offered).  The second
    ``st.selectbox`` shows the unique values from that column so the user
    can pick a specific region.

    The returned GeoDataFrame is filtered to rows matching the chosen
    value.

    Args:
        gdf: GeoDataFrame loaded from the user's shapefile.

    Returns:
        A 3-tuple of ``(selected_gdf, column_name, region_name)`` where
        *selected_gdf* contains only the rows matching *region_name* in
        *column_name*, *column_name* is the attribute column the user
        picked, and *region_name* is the chosen region value.
    """
    import streamlit as st

    logger.debug("Rendering region selector (n_features=%d)", len(gdf))

    # Exclude the geometry column from the column chooser
    selectable_columns: list[str] = [
        col for col in gdf.columns if col.lower() != "geometry"
    ]

    if not selectable_columns:
        logger.error("No non-geometry columns found in GeoDataFrame.")
        st.error(
            "The shapefile has no attribute columns to select a region. "
            "Please use a shapefile with at least one text/numeric field."
        )
        return gdf, "", ""

    column_name: str = st.selectbox(
        label="🔍 Select Region Column",
        options=selectable_columns,
        index=0,
        key="region_column_select",
        help="Choose the attribute column that identifies different regions/watersheds.",
    )

    unique_values: list[str] = gdf[column_name].astype(str).unique().tolist()
    region_name: str = st.selectbox(
        label="🏔 Select Region",
        options=unique_values,
        index=0,
        key="region_value_select",
        help="Pick the specific watershed or region to analyse.",
    )

    # Filter the GeoDataFrame to the selected region
    selected_gdf = gdf[gdf[column_name].astype(str) == region_name].copy()

    logger.info(
        "Region selected: column='%s', value='%s', n_rows=%d",
        column_name,
        region_name,
        len(selected_gdf),
    )
    return selected_gdf, column_name, region_name


def render_dem_upload() -> UploadedFile | None:
    """Render an optional file uploader for a DEM raster.

    Accepts ``.tif`` and ``.tiff`` files.  The upload is optional —
    if the user skips it, downstream analysis simply omits
    elevation-dependent results.

    Returns:
        The uploaded ``UploadedFile`` object, or ``None``.
    """
    import streamlit as st

    logger.debug("Rendering DEM upload widget")
    uploaded = st.file_uploader(
        label="🌄 Upload DEM (optional)",
        type=["tif", "tiff"],
        key="dem_upload",
        help=(
            "Optionally upload a Digital Elevation Model (DEM) as a GeoTIFF. "
            "Used for elevation statistics and orographic rainfall analysis."
        ),
    )
    if uploaded is not None:
        logger.info("DEM uploaded: %s (%d bytes)", uploaded.name, uploaded.size)
    return uploaded


def render_raster_upload() -> list[UploadedFile]:
    """Render a multi-file uploader for rainfall raster data.

    Accepts one or more ``.tif`` / ``.tiff`` files.  Each file is
    expected to represent a single time-step of gridded rainfall.

    Returns:
        A list of ``UploadedFile`` objects (may be empty).
    """
    import streamlit as st

    logger.debug("Rendering raster upload widget")
    uploaded = st.file_uploader(
        label="🌧️ Upload Rainfall Rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="raster_upload",
        help=(
            "Upload one or more GeoTIFF rasters representing gridded rainfall. "
            "Each file should correspond to a single time step (e.g. one month)."
        ),
    )
    files = list(uploaded) if uploaded else []
    if files:
        logger.info("Raster upload: %d file(s)", len(files))
    return files


def render_netcdf_upload() -> UploadedFile | None:
    """Render an optional file uploader for NetCDF rainfall data.

    Accepts ``.nc`` and ``.nc4`` files.  When provided, the application
    extracts time slices from the NetCDF rather than expecting individual
    GeoTIFF rasters.

    Returns:
        The uploaded ``UploadedFile`` object, or ``None``.
    """
    import streamlit as st

    logger.debug("Rendering NetCDF upload widget")
    uploaded = st.file_uploader(
        label="📊 Upload NetCDF (optional)",
        type=["nc", "nc4"],
        key="netcdf_upload",
        help=(
            "Optionally upload a NetCDF file containing gridded rainfall data. "
            "The file should have a time dimension and a 2-D spatial variable."
        ),
    )
    if uploaded is not None:
        logger.info("NetCDF uploaded: %s (%d bytes)", uploaded.name, uploaded.size)
    return uploaded


def render_analysis_settings() -> dict[str, Any]:
    """Render an expander with analysis configuration controls.

    The following settings are exposed:

    - **ARF method** — Selectbox populated from
      :pydata:`config.ARF_METHODS` keys.
    - **Rainfall unit** — Selectbox populated from
      :pydata:`config.RAINFALL_UNITS` keys.
    - **SPI scales** — Multi-select populated from
      :pydata:`config.SPI_SCALES`.

    Returns:
        A dictionary with keys ``'arf_method'``, ``'rainfall_unit'``,
        and ``'spi_scales'`` reflecting the user's selections.
    """
    import streamlit as st

    logger.debug("Rendering analysis settings expander")

    with st.expander(
        label="⚙️ Analysis Settings",
        expanded=False,
    ):
        arf_method: str = st.selectbox(
            label="ARF Method",
            options=list(ARF_METHODS.keys()),
            index=list(ARF_METHODS.keys()).index(DEFAULT_ARF_METHOD),
            key="arf_method_select",
            help="Select the Areal Reduction Factor method to apply.",
        )

        # Display the description beneath the selector
        st.caption(ARF_METHODS[arf_method])

        rainfall_unit: str = st.selectbox(
            label="Rainfall Unit",
            options=list(RAINFALL_UNITS.keys()),
            index=list(RAINFALL_UNITS.keys()).index(DEFAULT_RAINFALL_UNIT),
            key="rainfall_unit_select",
            help="Select the unit of the input rainfall data.",
        )

        # Display the description beneath the selector
        st.caption(RAINFALL_UNITS[rainfall_unit])

        # Default selection for SPI scales
        default_index: list[int] = []
        if DEFAULT_SPI_SCALE in SPI_SCALES:
            default_index = [SPI_SCALES.index(DEFAULT_SPI_SCALE)]

        spi_scales: list[int] = st.multiselect(
            label="SPI Scales (months)",
            options=SPI_SCALES,
            default=SPI_SCALES if not default_index else [DEFAULT_SPI_SCALE],
            key="spi_scale_multiselect",
            help=(
                "Select one or more accumulation windows for the "
                "Standardized Precipitation Index."
            ),
        )

    settings: dict[str, Any] = {
        "arf_method": arf_method,
        "rainfall_unit": rainfall_unit,
        "spi_scales": spi_scales,
    }

    logger.info("Analysis settings: %s", settings)
    return settings
