Watershed Rainfall Analyzer v3.0 — Research-Grade Hydrological Analysis.

This is the Streamlit entry-point. It orchestrates the correct workflow:
    1. Upload watershed shapefile
    2. Select region of interest
    3. (Optional) Upload DEM
    4. Upload rainfall rasters (GeoTIFF or NetCDF)
    5. Configure analysis settings (ARF, unit, SPI scales)
    6. Run analysis & view results

The application now uses area-weighted zonal statistics with ARF
correction, interactive Plotly charts, Folium maps, and rigorous
statistical tests (Mann-Kendall, Sen's slope, Pettitt, SPI, SI, PCI).
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st

from watershed_analyzer import __version__
from watershed_analyzer.config import (
    ARF_METHODS,
    EQUAL_AREA_CRS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    PROVENANCE_TEMPLATE,
    RAINFALL_UNITS,
    SPI_SCALES,
    TOOL_NAME,
    TOOL_VERSION,
)
from watershed_analyzer.core.dem import (
    compute_elevation_rainfall_regression,
    process_dem,
)
from watershed_analyzer.core.io import (
    compute_basin_area_km2,
    load_raster_bytes,
    load_shapefile,
)
from watershed_analyzer.core.netcdf_support import netcdf_to_geotiff_bytes
from watershed_analyzer.core.stats import (
    compute_all_statistics,
    compute_spi,
)
from watershed_analyzer.core.validation import (
    check_raster_consistency,
    validate_raster,
)
from watershed_analyzer.core.zonal import hydrologically_correct_zonal_stats
from watershed_analyzer.ui.charts import (
    plot_annual_totals,
    plot_elevation_rainfall_scatter,
    plot_monthly_rainfall_bar,
    plot_rainfall_trend,
    plot_spatial_variability,
    plot_spi_chart,
)
from watershed_analyzer.ui.maps import (
    create_interactive_map,
    create_raster_overlay_map,
    folium_to_html,
)
from watershed_analyzer.ui.results import (
    render_data_table,
    render_dem_characteristics,
    render_drought_analysis,
    render_seasonal_characteristics,
    render_spatial_stats_table,
    render_summary_metrics,
    render_trend_analysis,
)
from watershed_analyzer.ui.upload import (
    render_analysis_settings,
    render_dem_upload,
    render_netcdf_upload,
    render_raster_upload,
    render_region_selector,
    render_shapefile_upload,
)

# ---------------------------------------------------------------------------
# Logging configuration — writes to both file and stderr (streamlit)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("watershed_run.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Watershed Rainfall Analyzer",
    page_icon="\U0001F327\uFE0F",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "results_store" not in st.session_state:
    st.session_state.results_store: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

# ---------------------------------------------------------------------------
# Title & Description (NO "delineation" claim)
# ---------------------------------------------------------------------------
st.title("\U0001F327\uFE0F Watershed Rainfall Analyzer")
st.caption(f"v{TOOL_VERSION}  —  Research-grade area-weighted zonal statistics with ARF correction")
st.markdown(
    """
This tool computes **hydrologically defensible basin rainfall** from gridded
precipitation products over watershed polygons.

### Key Features
- **Area-weighted zonal statistics** (rasterio.mask, `all_touched=True`)
- **Areal Reduction Factor** (Srikanthan & McMahon 2007; USGS Reed 1999)
- **Basin area in equal-area CRS** (ESRI:54034)
- **Sentinel-value detection** (−9999, −32768, …)
- **Mann–Kendall, Sen's slope, Pettitt change-point** tests
- **SPI, Seasonality Index, PCI** drought/rainfall indices
- **Interactive Plotly charts & Folium maps**
- **NetCDF / Zarr support** via xarray
- **Multi-format export** (CSV w/ provenance, GeoJSON, GeoPackage)

### References
- Lam & De Cola (1993). *Fractals in Geography*. Prentice Hall.
- Srikanthan & McMahon (2007). J. Hydrol., 228, 56–69.
- Reed (1999). USGS Water-Supply Paper 2375.
- Mann (1945). Econometrica, 13, 245–259.
- McKee et al. (1993). AMS 8th Conf. Applied Climatology.
"""
)

# ======================================================================
# STEP 1: Upload Watershed Shapefile
# ======================================================================
st.header("1. Upload Watershed Boundary")
uploaded_zip = render_shapefile_upload()

gdf: gpd.GeoDataFrame | None = None

if uploaded_zip is not None:
    zip_buffer = io.BytesIO(uploaded_zip.read())
    with st.spinner("Loading shapefile …"):
        gdf = load_shapefile(zip_buffer)

    if gdf is None:
        st.error("Failed to load shapefile. Ensure the ZIP contains a valid .shp and its component files.")
    else:
        st.success(f"Shapefile loaded — {len(gdf)} region(s) found.  CRS: `{gdf.crs}`")
        with st.expander("View all regions"):
            st.dataframe(gdf.drop(columns="geometry"), use_container_width=True)

        # ==================================================================
        # STEP 2: Select Region of Interest
        # ==================================================================
        st.header("2. Select Region of Interest")
        selected_poly, col_name, region_name = render_region_selector(gdf)

        # Show interactive map
        st.subheader("Region Map")
        folium_map = create_interactive_map(
            gdf, selected_poly, highlight_column=col_name
        )
        st.components.v1.html(folium_to_html(folium_map), height=500)

        # Compute true basin area
        basin_area_km2 = compute_basin_area_km2(selected_poly)
        st.info(f"**Basin area** (equal-area CRS `{EQUAL_AREA_CRS}`): **{basin_area_km2:,.2f} km²**")

        # ==================================================================
        # STEP 3: (Optional) Upload DEM
        # ==================================================================
        st.header("3. (Optional) Upload DEM")
        uploaded_dem = render_dem_upload()

        dem_stats: dict[str, Any] | None = None
        dem_memfile = None
        if uploaded_dem is not None:
            from rasterio import MemoryFile

            dem_bytes = uploaded_dem.read()
            dem_memfile = MemoryFile(dem_bytes)
            try:
                dem_stats = process_dem(dem_memfile, selected_poly)
                if dem_stats:
                    st.success("DEM processed successfully")
                    render_dem_characteristics(dem_stats)
            except (ValueError, TypeError) as exc:
                logger.warning("DEM processing failed: %s", exc)
                st.warning(f"DEM processing issue: {exc}")

        # ==================================================================
        # STEP 4: Upload Rainfall Data
        # ==================================================================
        st.header("4. Upload Rainfall Data")

        tab_geotiff, tab_netcdf = st.tabs(["GeoTIFF", "NetCDF / Zarr"])

        uploaded_tifs: list[Any] = []
        with tab_geotiff:
            uploaded_tifs = render_raster_upload()

        uploaded_nc: Any = None
        with tab_netcdf:
            uploaded_nc = render_netcdf_upload()

        # Handle NetCDF → convert to GeoTIFF bytes list
        nc_tif_pairs: list[tuple[str, bytes]] = []
        if uploaded_nc is not None:
            try:
                from watershed_analyzer.core.io import load_netcdf

                nc_bytes = uploaded_nc.read()
                da, nc_meta = load_netcdf(nc_bytes, uploaded_nc.name)
                nc_tif_pairs = netcdf_to_geotiff_bytes(da)
                st.success(
                    f"NetCDF loaded: variable `{nc_meta['variable']}`, "
                    f"{len(nc_tif_pairs)} time step(s) extracted."
                )
            except (ValueError, ImportError, OSError) as exc:
                logger.error("NetCDF processing failed: %s", exc)
                st.error(f"NetCDF processing failed: {exc}")

        # Combine all raster sources
        all_raster_sources: list[tuple[str, bytes]] = []
        for tif_file in uploaded_tifs:
            all_raster_sources.append((tif_file.name, tif_file.read()))
        all_raster_sources.extend(nc_tif_pairs)

        # ==================================================================
        # STEP 5: Analysis Settings
        # ==================================================================
        if all_raster_sources:
            st.header("5. Analysis Settings")
            settings = render_analysis_settings()

            # ==================================================================
            # Validation
            # ==================================================================
            with st.expander("\U0001F50D Validate Rasters"):
                if st.button("Run Validation", key="validate_btn"):
                    valid_meta_list: list[dict[str, Any]] = []
                    for fname, fbytes in all_raster_sources:
                        from rasterio import MemoryFile as MF

                        mf = MF(fbytes)
                        is_valid, msg, meta = validate_raster(mf, fname)
                        if "passed" in msg.lower() or is_valid:
                            st.success(msg)
                            if meta:
                                valid_meta_list.append(meta)
                        elif "warning" in msg.lower() or "negative" in msg.lower():
                            st.warning(msg)
                            if meta:
                                valid_meta_list.append(meta)
                        else:
                            st.error(msg)
                    if len(valid_meta_list) > 1:
                        is_cons, cons_msg = check_raster_consistency(valid_meta_list)
                        if is_cons:
                            st.success(cons_msg)
                        else:
                            st.warning(cons_msg)

            # ==================================================================
            # STEP 6: Run Analysis
            # ==================================================================
            st.header("6. Run Analysis")

            if st.button(
                "\U0001F680 Run Rainfall Analysis",
                type="primary",
                use_container_width=True,
                key="run_analysis_btn",
            ):
                from rasterio import MemoryFile as MF

                results: list[dict[str, Any]] = []
                skipped: list[tuple[str, str]] = []
                progress = st.progress(0, text="Starting …")

                arf_method = settings.get("arf_method", "srikanthan_mcmahon")
                rainfall_unit = settings.get("rainfall_unit", "mm/month")
                spi_scales = settings.get("spi_scales", [3])

                for idx, (fname, fbytes) in enumerate(all_raster_sources):
                    progress.progress(
                        (idx + 1) / len(all_raster_sources),
                        text=f"Processing {fname} ({idx + 1}/{len(all_raster_sources)})",
                    )
                    try:
                        mf = MF(fbytes)

                        # Validate
                        is_valid, val_msg, _ = validate_raster(mf, fname)
                        if not is_valid:
                            skipped.append((fname, val_msg))
                            continue

                        # Zonal stats
                        zonal = hydrologically_correct_zonal_stats(
                            polygon=selected_poly,
                            raster_memfile=mf,
                            equal_area_crs=EQUAL_AREA_CRS,
                            arf_method=arf_method,
                            rainfall_unit=rainfall_unit,
                        )

                        month_label = fname.rsplit(".", 1)[0]
                        row: dict[str, Any] = {
                            "Month": month_label,
                            "Spatial_Mean_mm": zonal["spatial_mean_mm"],
                            "Spatial_Std_mm": zonal["spatial_std_mm"],
                            "Spatial_CV_pct": zonal["spatial_cv_pct"],
                            "Min_mm": zonal["min_mm"],
                            "Max_mm": zonal["max_mm"],
                            "Median_mm": zonal["median_mm"],
                            "Q25_mm": zonal["q25_mm"],
                            "Q75_mm": zonal["q75_mm"],
                            "Basin_Rainfall_mm": zonal["basin_rainfall_mm"],
                            "Volume_m3": zonal["volume_m3"],
                            "Basin_Area_km2": zonal["basin_area_km2"],
                            "ARF": zonal["arf"],
                            "Valid_Pixels": zonal["n_valid_pixels"],
                            "Total_Pixels": zonal["n_total_pixels"],
                            "Coverage_pct": zonal["coverage_pct"],
                        }
                        results.append(row)

                        logger.info("Processed %s: basin rainfall = %.2f mm", fname, zonal["basin_rainfall_mm"])

                    except (ValueError, TypeError) as exc:
                        logger.warning("Skipping %s: %s", fname, exc)
                        skipped.append((fname, str(exc)))
                    except Exception as exc:
                        logger.error("Error processing %s: %s", fname, exc, exc_info=True)
                        skipped.append((fname, f"Unexpected error: {exc}"))

                progress.empty()

                # Show skipped
                if skipped:
                    with st.expander(f"\u26A0\uFE0F {len(skipped)} file(s) skipped"):
                        for fn, reason in skipped:
                            st.warning(f"**{fn}**: {reason}")

                if results:
                    df = pd.DataFrame(results)

                    # Chronological sort — try MM-YYYY, then YYYY-MM, then
                    # any embedded 4-digit year + month token in the filename.
                    date_parsed = pd.Series([pd.NaT] * len(df), index=df.index)
                    for fmt in ("%m-%Y", "%Y-%m", "%Y_%m", "%m_%Y"):
                        candidate = pd.to_datetime(df["Month"], format=fmt, errors="coerce")
                        date_parsed = date_parsed.fillna(candidate)
                    if date_parsed.isna().any():
                        # Fall back to loose parsing for any remaining rows.
                        loose = pd.to_datetime(df["Month"], errors="coerce")
                        date_parsed = date_parsed.fillna(loose)

                    n_unparsed = int(date_parsed.isna().sum())
                    if n_unparsed == 0:
                        df["_date_sort"] = date_parsed
                        dupes = df["_date_sort"][df["_date_sort"].duplicated(keep=False)]
                        if not dupes.empty:
                            dup_months = sorted(
                                df.loc[dupes.index, "Month"].astype(str).unique()
                            )
                            st.warning(
                                "Multiple files map to the same month after date "
                                f"parsing: {', '.join(dup_months)}. Check for "
                                "duplicate or mis-named rasters — this can skew "
                                "the trend chart and variability statistics."
                            )
                        df = df.sort_values("_date_sort").drop(columns=["_date_sort"])
                    elif n_unparsed < len(df):
                        st.warning(
                            f"Could not parse a date from {n_unparsed} of {len(df)} "
                            "filenames. Results are shown in upload order, NOT "
                            "chronological order — trend charts and 'wettest/"
                            "driest month' below may be misleading. Rename files "
                            "as MM-YYYY or YYYY-MM (e.g. 01-2022.tif)."
                        )
                        logger.warning(
                            "Date sort skipped: %d/%d filenames unparseable.",
                            n_unparsed, len(df),
                        )
                    else:
                        st.warning(
                            "None of the uploaded filenames could be parsed as "
                            "dates. Results are in upload order, not chronological "
                            "order. Rename files as MM-YYYY or YYYY-MM "
                            "(e.g. 01-2022.tif) for correct sorting and trend "
                            "analysis."
                        )
                        logger.warning("Date sort failed for all %d rows.", len(df))

                    # Store in session state for comparison mode
                    st.session_state.results_store[region_name] = (df, settings)

                    # ---- Compute statistics ----
                    rainfall_series = df["Basin_Rainfall_mm"]

                    # Determine first ARF from the first row
                    first_arf = float(df["ARF"].iloc[0]) if len(df) > 0 else 1.0

                    # ---- Summary ----
                    st.subheader("Analysis Results")
                    render_summary_metrics(df, region_name, basin_area_km2, first_arf)

                    # ---- Spatial variability ----
                    st.subheader("Intra-Watershed Spatial Variability")
                    render_spatial_stats_table(df)
                    st.plotly_chart(
                        plot_spatial_variability(
                            df,
                            "Month",
                            "Spatial_Std_mm",
                            "Spatial_CV_pct",
                        ),
                        use_container_width=True,
                    )

                    # ---- DEM integration ----
                    elevation_regression: dict[str, Any] | None = None
                    if dem_stats is not None and dem_memfile is not None:
                        st.subheader("\U0001F3D4\uFE0F Elevation Analysis")
                        render_dem_characteristics(dem_stats)

                        # Elevation-rainfall regression (on last raster)
                        try:
                            last_mf = MF(all_raster_sources[-1][1])
                            with last_mf.open() as src:
                                import rasterio.mask

                                poly_r = selected_poly.to_crs(src.crs)
                                out_img, _ = rasterio.mask.mask(
                                    src, poly_r.geometry.values.tolist(),
                                    crop=True, all_touched=True,
                                )
                                if out_img.ndim == 3:
                                    out_img = out_img.squeeze(0)

                            with dem_memfile.open() as dem_src:
                                dem_data = dem_src.read(1)
                                # Mask to same shape
                                if dem_data.shape != out_img.shape:
                                    import rasterio.mask as rm2
                                    dem_img, _ = rm2.mask(
                                        dem_src, poly_r.geometry.values.tolist(),
                                        crop=True, all_touched=True,
                                    )
                                    if dem_img.ndim == 3:
                                        dem_img = dem_img.squeeze(0)
                                else:
                                    dem_img = dem_data

                                valid = (
                                    np.isfinite(out_img)
                                    & np.isfinite(dem_img)
                                    & (out_img > 0)
                                )
                                if valid.sum() > 10:
                                    elevation_regression = (
                                        compute_elevation_rainfall_regression(
                                            dem_img[valid],
                                            out_img[valid],
                                            valid,
                                        )
                                    )
                                    if elevation_regression:
                                        st.plotly_chart(
                                            plot_elevation_rainfall_scatter(
                                                dem_img[valid],
                                                out_img[valid],
                                                elevation_regression,
                                            ),
                                            use_container_width=True,
                                        )
                        except Exception as exc:
                            logger.warning("Elevation-rainfall regression failed: %s", exc)

                    # ---- Trend analysis ----
                    st.subheader("\U0001F4C8 Trend Analysis")
                    if len(rainfall_series) >= 4:
                        stats_results = compute_all_statistics(rainfall_series)
                        render_trend_analysis(stats_results)

                        # Store MK results for chart annotation
                        if "mann_kendall" in stats_results and "error" not in stats_results["mann_kendall"]:
                            st.session_state["mann_kendall"] = stats_results["mann_kendall"]
                    else:
                        st.info("Need at least 4 data points for trend analysis.")
                        stats_results = {}

                    # ---- Drought analysis (SPI) ----
                    st.subheader("\U0001F4A7 Drought Analysis (SPI)")
                    if len(rainfall_series) >= spi_scales[0] if spi_scales else 12:
                        try:
                            spi_results: dict[int, np.ndarray] = {}
                            for scale in spi_scales:
                                if len(rainfall_series) >= scale:
                                    spi_arr = compute_spi(rainfall_series.values, scale=scale)
                                    spi_results[scale] = spi_arr

                            if spi_results:
                                spi_df = pd.DataFrame(spi_results, index=df["Month"].values)
                                spi_df.index.name = "Month"
                                render_drought_analysis(spi_results, df["Month"].tolist())
                                st.plotly_chart(plot_spi_chart(spi_df, list(spi_results.keys())), use_container_width=True)
                        except (ValueError, TypeError) as exc:
                            logger.warning("SPI computation failed: %s", exc)
                            st.warning(f"SPI computation issue: {exc}")
                    else:
                        st.info(f"Need at least {spi_scales[0] if spi_scales else 12} months for SPI computation.")

                    # ---- Seasonal characteristics ----
                    st.subheader("\U0001F3D6\uFE0F Seasonal Characteristics")
                    if len(rainfall_series) >= 12:
                        # Try to get 12 calendar-month means
                        try:
                            df_temp = df.copy()
                            df_temp["_date"] = pd.to_datetime(df_temp["Month"], format="%m-%Y", errors="coerce")
                            if df_temp["_date"].isna().all():
                                df_temp["_date"] = pd.to_datetime(df_temp["Month"], format="%Y-%m", errors="coerce")
                            if df_temp["_date"].notna().any():
                                monthly_means = df_temp.groupby(df_temp["_date"].dt.month)["Basin_Rainfall_mm"].mean()
                                if len(monthly_means) == 12:
                                    from watershed_analyzer.core.stats import (
                                        precipitation_concentration_index,
                                        seasonality_index,
                                    )
                                    si_result = seasonality_index(monthly_means.values)
                                    pci_result = precipitation_concentration_index(monthly_means.values)
                                    render_seasonal_characteristics(si_result, pci_result)
                                else:
                                    st.info("Need data covering all 12 calendar months for SI/PCI.")
                            else:
                                st.info("Could not parse dates for seasonal analysis.")
                        except (ValueError, TypeError) as exc:
                            logger.warning("Seasonal analysis failed: %s", exc)
                    else:
                        st.info("Need at least 12 months for Seasonality Index and PCI.")

                    # ---- Provenance header ----
                    provenance = PROVENANCE_TEMPLATE.format(
                        tool_name=TOOL_NAME,
                        tool_version=TOOL_VERSION,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        shapefile_name=uploaded_zip.name,
                        shapefile_crs=str(gdf.crs),
                        n_features=len(gdf),
                        region_name=region_name,
                        basin_area_km2=basin_area_km2,
                        raster_names=", ".join(r[0] for r in all_raster_sources),
                        arf_method=arf_method,
                        arf_value=first_arf,
                        rainfall_unit=rainfall_unit,
                        equal_area_crs=EQUAL_AREA_CRS,
                    )

                    # ---- Tabs: Table, Charts, Map, Annual ----
                    tab_data, tab_bar, tab_trend, tab_map, tab_annual = st.tabs([
                        "\U0001F4CB Data Table",
                        "\U0001F4CA Bar Chart",
                        "\U0001F4C8 Trend",
                        "\U0001F5FA\uFE0F Map",
                        "\U0001F4C5 Annual",
                    ])

                    with tab_data:
                        render_data_table(df, region_name, provenance, gdf, col_name)

                    with tab_bar:
                        st.plotly_chart(
                            plot_monthly_rainfall_bar(
                                df, "Month", "Basin_Rainfall_mm",
                                f"Monthly Basin Rainfall — {region_name}",
                            ),
                            use_container_width=True,
                        )

                    with tab_trend:
                        st.plotly_chart(
                            plot_rainfall_trend(
                                df, "Month", "Basin_Rainfall_mm",
                                f"Rainfall Trend — {region_name}",
                            ),
                            use_container_width=True,
                        )

                    with tab_map:
                        if all_raster_sources:
                            last_fname, last_bytes = all_raster_sources[-1]
                            last_mf = MF(last_bytes)
                            fmap = create_raster_overlay_map(
                                selected_poly, last_mf, last_fname
                            )
                            st.components.v1.html(folium_to_html(fmap), height=500)

                    with tab_annual:
                        annual_fig = plot_annual_totals(df)
                        if annual_fig is not None:
                            st.plotly_chart(annual_fig, use_container_width=True)
                        else:
                            st.info("Could not determine year from filenames.")

                    # ---- Comparison mode ----
                    if len(st.session_state.results_store) > 1:
                        st.subheader("\U0001F4CA Comparison Mode")
                        st.caption("Compare results across multiple regions or re-runs.")
                        selected_comparisons = st.multiselect(
                            "Select regions to compare:",
                            list(st.session_state.results_store.keys()),
                            default=list(st.session_state.results_store.keys()),
                        )
                        if len(selected_comparisons) > 1:
                            from watershed_analyzer.ui.charts import plot_comparison

                            comp_data = {
                                k: st.session_state.results_store[k][0]
                                for k in selected_comparisons
                            }
                            st.plotly_chart(
                                plot_comparison(comp_data),
                                use_container_width=True,
                            )

                    logger.info(
                        "Analysis complete for '%s': %d months processed, "
                        "%d skipped.",
                        region_name,
                        len(results),
                        len(skipped),
                    )

                else:
                    st.error(
                        "No results generated. All files were skipped due to validation errors."
                    )

        else:
            st.info("Upload rainfall rasters (GeoTIFF or NetCDF) to continue.")

    # Bare except removed — specific exception handling above

else:
    st.info("\U0001F446 Upload a watershed shapefile (ZIP) to begin.")
    st.markdown(
        """
---
### File Requirements
- **Shapefile ZIP**: must contain `.shp`, `.shx`, `.dbf`, `.prj`
- **Raster files**: GeoTIFF (`.tif`/`.tiff`) or NetCDF (`.nc`)
- **Single-band** rasters for rainfall analysis
- **File naming**: include dates (e.g., `01-2022.tif`, `2022-01.nc`)

### Data Quality
- The app automatically reprojects to match CRS
- All rasters should share the same resolution and CRS
- Sentinel values (−9999, −32768) are detected and masked
- Basin area is computed in an equal-area projection

### Scientific Methodology
- Zonal statistics via `rasterio.mask` with `all_touched=True`
- Areal Reduction Factor (Srikanthan & McMahon 2007)
- Mann–Kendall trend test, Sen's slope, Pettitt change-point
- SPI (McKee et al. 1993), Seasonality Index (Walsh & Lawler 1981),
  PCI (Oliver 1980)
    """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"\U0001F327\uFE0F **Watershed Rainfall Analyzer** v{TOOL_VERSION} "
    f"| Area-weighted zonal statistics with ARF correction "
    f"| [{__version__}](https://github.com/)"
)
