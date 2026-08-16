"""Streamlit results display components for the Watershed Analyzer UI.

This module provides functions that render analysis results in the
Streamlit main area.  Each function is responsible for one logical section
of the results dashboard (summary metrics, data tables, statistical tests,
drought indices, etc.).

Functions:
    render_summary_metrics: Key basin rainfall metrics via st.metrics.
    render_data_table: Interactive dataframe with CSV / GeoJSON / GPKG downloads.
    render_spatial_stats_table: Spatial variability statistics table.
    render_trend_analysis: Mann-Kendall, Sen's slope, Pettitt results.
    render_drought_analysis: SPI table and drought classification counts.
    render_seasonal_characteristics: Seasonality Index and PCI displays.
    render_dem_characteristics: DEM elevation statistics and regression.
    render_comparison_mode: Tabbed multi-watershed / multi-period comparison.
"""

from __future__ import annotations

import io
import csv as csv_mod
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd
    import streamlit as st
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# SPI drought classification thresholds (McKee et al. 1993)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────

_SPI_CLASSIFICATIONS: list[tuple[str, tuple[float, float]]] = [
    ("Extremely Wet", (2.0, float("inf"))),
    ("Very Wet", (1.5, 2.0)),
    ("Moderately Wet", (1.0, 1.5)),
    ("Near Normal", (-1.0, 1.0)),
    ("Moderately Dry", (-1.5, -1.0)),
    ("Severely Dry", (-2.0, -1.5)),
    ("Extremely Dry", (float("-inf"), -2.0)),
]


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _classify_spi(spi_value: float) -> str:
    """Classify a single SPI value using McKee et al. (1993) thresholds.

    Args:
        spi_value: A single SPI number.

    Returns:
        The classification label string.
    """
    for label, (lo, hi) in _SPI_CLASSIFICATIONS:
        if lo <= spi_value < hi:
            return label
    return "Unknown"


def _build_csv_with_provenance(
    df: pd.DataFrame,
    provenance_header: str,
) -> str:
    """Serialise a DataFrame to CSV prefixed with comment-style provenance.

    Each line of *provenance_header* is emitted as-is (it is expected to
    start with ``#``).  An empty line separates the header from the CSV
    body.

    Args:
        df: DataFrame to export.
        provenance_header: Multi-line string of comment lines.

    Returns:
        A single string containing the provenance header followed by
        the CSV data.
    """
    buf = io.StringIO()
    buf.write(provenance_header)
    if not provenance_header.endswith("\n"):
        buf.write("\n")
    buf.write("\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _fmt(val: Any, precision: int = 2) -> str:
    """Format a numeric value for display, handling None and NaN.

    Args:
        val: The value to format.
        precision: Decimal places for floats.

    Returns:
        A human-readable string.
    """
    if val is None:
        return "N/A"
    try:
        fval = float(val)
        if np.isnan(fval):
            return "N/A"
        return f"{fval:.{precision}f}"
    except (TypeError, ValueError):
        return str(val)


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 1. Summary metrics
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_summary_metrics(
    df: pd.DataFrame,
    region_name: str,
    basin_area_km2: float,
    arf: float,
) -> None:
    """Render key basin rainfall metrics using ``st.metric``.

    Displays six metrics in two rows of three:

    1. Total basin rainfall (sum of ``Basin_Rainfall_mm`` or the first
       numeric column).
    2. Mean monthly rainfall.
    3. Wettest month (max value and label).
    4. Driest month (min value and label).
    5. Basin area (km²).
    6. Areal Reduction Factor.

    Args:
        df: DataFrame with at least one numeric rainfall column and a
            month/time label column.
        region_name: Display name of the selected region.
        basin_area_km2: Computed basin area in square kilometres.
        arf: Areal Reduction Factor applied.
    """
    import streamlit as st

    logger.debug("Rendering summary metrics for region '%s'", region_name)

    # Identify the primary rainfall column
    rainfall_col: str | None = None
    for candidate in ("Basin_Rainfall_mm", "Rainfall_mm", "rainfall"):
        if candidate in df.columns:
            rainfall_col = candidate
            break
    if rainfall_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        rainfall_col = numeric_cols[0] if numeric_cols else None

    if rainfall_col is None:
        st.warning("No numeric rainfall column found for summary metrics.")
        return

    values = df[rainfall_col].astype(float)
    total = float(values.sum())
    mean_monthly = float(values.mean())
    max_val = float(values.max())
    min_val = float(values.min())

    # Identify month / label column for wettest/driest labels
    label_col: str | None = None
    for candidate in ("Month", "month", "Date", "date", "Time", "time"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        non_numeric = [c for c in df.columns if c != rainfall_col]
        label_col = non_numeric[0] if non_numeric else None

    if label_col is not None:
        max_idx = int(values.idxmax())
        min_idx = int(values.idxmin())
        wettest_label = str(df[label_col].iloc[max_idx])
        driest_label = str(df[label_col].iloc[min_idx])
    else:
        wettest_label = ""
        driest_label = ""

    st.subheader(f"📊 Summary Metrics — {region_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Basin Rainfall", value=f"{total:,.1f} mm")
    with col2:
        st.metric(label="Mean Monthly Rainfall", value=f"{mean_monthly:,.1f} mm")
    with col3:
        st.metric(label="Basin Area", value=f"{basin_area_km2:,.2f} km²")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric(
            label="Wettest Month",
            value=f"{max_val:,.1f} mm",
            delta=wettest_label if wettest_label else None,
        )
    with col5:
        st.metric(
            label="Driest Month",
            value=f"{min_val:,.1f} mm",
            delta=driest_label if driest_label else None,
        )
    with col6:
        st.metric(label="ARF Applied", value=f"{arf:.4f}")

    logger.info(
        "Summary metrics rendered: total=%.1f mm, mean=%.1f mm, area=%.2f km²",
        total,
        mean_monthly,
        basin_area_km2,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 2. Data table with downloads
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_data_table(
    df: pd.DataFrame,
    region_name: str,
    provenance_header: str,
    gdf: gpd.GeoDataFrame | None = None,
    join_column: str | None = None,
) -> None:
    """Render an interactive dataframe with CSV, GeoJSON, and GeoPackage downloads.

    The CSV download prepends the *provenance_header* as comment lines
    (each starting with ``#``).  The GeoJSON and GeoPackage downloads
    join the results DataFrame back to the original polygon geometries
    when *gdf* and *join_column* are provided.

    Args:
        df: Results DataFrame to display.
        region_name: Display name of the region (used in section header).
        provenance_header: Multi-line provenance comment string to
            prepend to the CSV download.
        gdf: Optional original GeoDataFrame whose geometry column
            should be attached to the downloads.
        join_column: Column name used to join *df* back to *gdf*.
            When ``None`` the join is attempted on the index.
    """
    import streamlit as st

    logger.debug(
        "Rendering data table for region '%s' (n_rows=%d)",
        region_name,
        len(df),
    )

    st.subheader(f"📋 Data Table — {region_name}")
    st.dataframe(df, use_container_width=True, height=420)

    # ── Download buttons ───────────────────────────────────────────────
    dl_cols = st.columns(3)

    # 1. CSV download with provenance header
    csv_content = _build_csv_with_provenance(df, provenance_header)
    with dl_cols[0]:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_content.encode("utf-8"),
            file_name=f"{region_name}_rainfall_results.csv",
            mime="text/csv",
            help="CSV with provenance comment header.",
        )

    # 2. GeoJSON download (if geometry available)
    if gdf is not None:
        try:
            result_gdf = _join_results_to_gdf(df, gdf, join_column)
            geojson_str = result_gdf.to_json()
            with dl_cols[1]:
                st.download_button(
                    label="⬇️ Download GeoJSON",
                    data=geojson_str.encode("utf-8"),
                    file_name=f"{region_name}_rainfall_results.geojson",
                    mime="application/geo+json",
                    help="GeoJSON with results joined to polygon geometry.",
                )

            # 3. GeoPackage download
            gpkg_buf = io.BytesIO()
            result_gdf.to_file(gpkg_buf, driver="GPKG")
            gpkg_bytes = gpkg_buf.getvalue()
            with dl_cols[2]:
                st.download_button(
                    label="⬇️ Download GeoPackage",
                    data=gpkg_bytes,
                    file_name=f"{region_name}_rainfall_results.gpkg",
                    mime="application/geopackage+sqlite3",
                    help="GeoPackage with results joined to polygon geometry.",
                )
        except Exception as exc:
            logger.warning("Spatial download failed: %s", exc)
            with dl_cols[1]:
                st.caption("GeoJSON unavailable")
            with dl_cols[2]:
                st.caption("GeoPackage unavailable")
    else:
        with dl_cols[1]:
            st.caption("GeoJSON — no geometry")
        with dl_cols[2]:
            st.caption("GeoPackage — no geometry")

    logger.info("Data table and download buttons rendered for '%s'", region_name)


def _join_results_to_gdf(
    df: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    join_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Join a results DataFrame back to a GeoDataFrame.

    If *join_column* is provided, an equi-join on that column is
    performed.  Otherwise the first non-geometry column common to both
    DataFrames is used.  Falls back to index-based assignment.

    Args:
        df: Results DataFrame.
        gdf: GeoDataFrame with geometry to preserve.
        join_column: Optional explicit join column name.

    Returns:
        A new GeoDataFrame containing the original geometry plus all
        columns from *df*.
    """
    import geopandas as gpd

    result = gdf.copy()

    if join_column and join_column in df.columns and join_column in gdf.columns:
        result = result.merge(df, on=join_column, how="left")
    else:
        # Try finding a common non-geometry column
        common_cols = [
            c for c in df.columns
            if c in gdf.columns and c.lower() != "geometry"
        ]
        if common_cols:
            result = result.merge(df, on=common_cols, how="left")
        else:
            # Last resort: assign by position
            for col in df.columns:
                if col not in result.columns:
                    result[col] = df[col].values[: len(result)]

    return result


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 3. Spatial statistics table
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_spatial_stats_table(df: pd.DataFrame) -> None:
    """Display spatial variability statistics in a clean table.

    Looks for columns matching the patterns ``*Std*``, ``*CV*``,
    ``*Min*``, ``*Max*``, ``*Median*``, ``*Q25*``, ``*Q75*`` (case-
    insensitive substring match) and presents them in an
    ``st.dataframe`` with formatted values.

    If no matching columns are found, a warning is shown.

    Args:
        df: DataFrame containing spatial variability columns.
    """
    import streamlit as st

    logger.debug("Rendering spatial stats table (n_cols=%d)", len(df.columns))

    # Identify spatial variability columns via flexible substring matching
    spatial_patterns = ["std", "cv", "min", "max", "median", "q25", "q75"]
    matched_cols: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pat in col_lower for pat in spatial_patterns):
            matched_cols.append(col)

    # Also try to include a label / month column for context
    label_col: str | None = None
    for candidate in ("Month", "month", "Date", "date"):
        if candidate in df.columns:
            label_col = candidate
            break

    if not matched_cols:
        st.warning(
            "No spatial variability columns detected. "
            "Expected columns containing: Std, CV, Min, Max, Median, Q25, Q75."
        )
        logger.warning("No spatial variability columns found in DataFrame.")
        return

    display_cols = [label_col, *matched_cols] if label_col else matched_cols
    display_cols = [c for c in display_cols if c in df.columns]

    st.subheader("📏 Spatial Variability Statistics")
    st.dataframe(df[display_cols], use_container_width=True, height=350)
    logger.info("Spatial stats table rendered with %d columns.", len(display_cols))


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 4. Trend analysis
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_trend_analysis(stats_results: dict[str, Any]) -> None:
    """Render trend and change-point test results in a structured layout.

    Displays results from three statistical tests:

    - **Mann-Kendall** — trend direction, Z-statistic, p-value,
      significance stars.
    - **Sen's Slope** — median slope and 95 % confidence interval.
    - **Pettitt** — change-point location, K-statistic, p-value, and
      significance flag.

    Each test is shown as a set of ``st.metric`` widgets.  Methodology
    details are provided in collapsible expanders.

    Args:
        stats_results: Dictionary with keys ``'mann_kendall'``,
            ``'sen_slope'``, and ``'pettitt'`` (as returned by
            :func:`core.stats.compute_all_statistics`).
    """
    import streamlit as st

    logger.debug("Rendering trend analysis")
    st.subheader("📈 Trend Analysis")

    # ── Mann-Kendall ───────────────────────────────────────────────────
    mk = stats_results.get("mann_kendall", {})
    if "error" not in mk:
        st.markdown("**Mann-Kendall Trend Test**")
        mk_col1, mk_col2, mk_col3 = st.columns(3)
        with mk_col1:
            st.metric(
                label="Trend",
                value=mk.get("trend", "N/A"),
                delta=mk.get("significance", "ns"),
            )
        with mk_col2:
            st.metric(label="Z-statistic", value=_fmt(mk.get("z"), 4))
        with mk_col3:
            st.metric(label="p-value", value=_fmt(mk.get("p_value"), 6))

        with st.expander("📖 Mann-Kendall Methodology"):
            st.markdown(
                r"The Mann-Kendall test is a non-parametric method for "
                r"detecting monotonic trends in time-series data. It computes "
                r"Kendall's *S* statistic from the signs of all pairwise "
                r"differences, then derives a standardised *Z* statistic with "
                r"continuity correction. Significance: \*\*\* *p* < 0.001, "
                r"\*\* *p* < 0.01, \* *p* < 0.05, ns = not significant."
            )
    else:
        st.warning(f"Mann-Kendall test error: {mk.get('error', 'unknown')}")

    st.divider()

    # ── Sen's Slope ────────────────────────────────────────────────────
    ss = stats_results.get("sen_slope", {})
    if "error" not in ss:
        st.markdown("**Sen's Slope Estimator**")
        ss_col1, ss_col2 = st.columns(2)
        with ss_col1:
            st.metric(
                label="Median Slope",
                value=f"{_fmt(ss.get('slope'), 4)} mm/step",
            )
        with ss_col2:
            st.metric(
                label="95 % CI",
                value=(
                    f"[{_fmt(ss.get('slope_lower'), 4)}, "
                    f"{_fmt(ss.get('slope_upper'), 4)}]"
                ),
            )

        with st.expander("📖 Sen's Slope Methodology"):
            st.markdown(
                "Sen's slope (Theil-Sen estimator) is the median of all "
                "pairwise slopes *(y_j − y_i) / (j − i)*. It provides a "
                "robust estimate of trend magnitude that is resistant to "
                "outliers. The 95 % confidence interval is derived from the "
                "2.5th and 97.5th percentiles of the slope distribution."
            )
    else:
        st.warning(f"Sen's slope error: {ss.get('error', 'unknown')}")

    st.divider()

    # ── Pettitt Test ───────────────────────────────────────────────────
    pt = stats_results.get("pettitt", {})
    if "error" not in pt:
        st.markdown("**Pettitt Change-Point Test**")
        pt_col1, pt_col2, pt_col3 = st.columns(3)
        with pt_col1:
            st.metric(
                label="Change Point (index)",
                value=str(pt.get("change_point_index", "N/A")),
            )
        with pt_col2:
            st.metric(
                label="K-statistic",
                value=_fmt(pt.get("k_statistic"), 2),
            )
        with pt_col3:
            sig_label = ("Significant" if pt.get("significant") else "Not Significant")
            st.metric(
                label="p-value",
                value=_fmt(pt.get("p_value"), 6),
                delta=sig_label,
            )

        with st.expander("📖 Pettitt Test Methodology"):
            st.markdown(
                r"The Pettitt test detects a single shift in the mean of a "
                r"time-series. It computes *U_{t,T} = Σ sign(x_t − x_j)* for "
                r"each position *t*, and the test statistic is *K_T = max|U|*. "
                r"The approximate p-value is *p ≈ 2·exp(−6·K²/(T³+T²))*. "
                r"A change point is considered significant when *p* < 0.05."
            )
    else:
        st.warning(f"Pettitt test error: {pt.get('error', 'unknown')}")

    logger.info("Trend analysis rendered (MK, Sen, Pettitt).")


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 5. Drought analysis (SPI)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_drought_analysis(
    spi_results: dict[int, np.ndarray],
    month_labels: list[str],
) -> None:
    """Render SPI values and drought classification counts per scale.

    For each SPI scale in *spi_results*, the function:

    1. Builds a DataFrame of SPI values with *month_labels* as the index.
    2. Classifies each value using McKee et al. (1993) thresholds.
    3. Displays a summary count of drought categories in a compact table.

    If multiple scales are provided, each scale is shown in its own
    ``st.expander`` to keep the layout manageable.

    Args:
        spi_results: Mapping of ``{scale: spi_array}`` where each array
            contains SPI values aligned with *month_labels*.
        month_labels: List of string labels (e.g. month names or dates)
            corresponding to the SPI values.
    """
    import streamlit as st

    logger.debug(
        "Rendering drought analysis for %d scale(s)", len(spi_results)
    )

    if not spi_results:
        st.info("No SPI results available (either no data or computation failed).")
        return

    st.subheader("🏜️ Drought Analysis — SPI")

    for scale, spi_arr in spi_results.items():
        if not isinstance(spi_arr, np.ndarray):
            logger.warning("SPI-%d is not a numpy array — skipping.", scale)
            continue

        n_values = min(len(spi_arr), len(month_labels))
        labels = month_labels[:n_values]
        values = spi_arr[:n_values]

        # Build a compact DataFrame
        spi_df = pd.DataFrame({
            "Period": labels,
            f"SPI-{scale}": values,
        })
        spi_df["Classification"] = spi_df[f"SPI-{scale}"].apply(_classify_spi)

        with st.expander(label=f"SPI-{scale} ({n_values} periods)", expanded=(scale == min(spi_results.keys()))):
            st.dataframe(spi_df, use_container_width=True, height=300)

            # Classification counts
            counts = spi_df["Classification"].value_counts().to_dict()

            count_cols = st.columns(len(_SPI_CLASSIFICATIONS))
            for idx, (label, _bounds) in enumerate(_SPI_CLASSIFICATIONS):
                with count_cols[idx]:
                    st.metric(
                        label=label,
                        value=str(counts.get(label, 0)),
                    )

    logger.info("Drought analysis rendered for scales: %s", list(spi_results.keys()))


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 6. Seasonal characteristics
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_seasonal_characteristics(
    si_result: dict[str, Any],
    pci_result: dict[str, Any],
) -> None:
    """Display Seasonality Index and Precipitation Concentration Index.

    Each index is rendered as a set of ``st.metric`` widgets showing
    the computed value and its classification.  If a result contains an
    ``'error'`` key, a warning message is displayed instead.

    Args:
        si_result: Dictionary from :func:`core.stats.seasonality_index`
            with keys ``'si_value'`` and ``'classification'``.
        pci_result: Dictionary from
            :func:`core.stats.precipitation_concentration_index` with
            keys ``'pci_value'`` and ``'classification'``.
    """
    import streamlit as st

    logger.debug("Rendering seasonal characteristics")
    st.subheader("🗓️ Seasonal Characteristics")

    si_col, pci_col = st.columns(2)

    # ── Seasonality Index ──────────────────────────────────────────────
    with si_col:
        st.markdown("**Seasonality Index (Walsh & Lawler 1981)**")
        if "error" not in si_result:
            st.metric(
                label="SI Value",
                value=_fmt(si_result.get("si_value"), 4),
            )
            st.info(
                f"**Classification:** {si_result.get('classification', 'N/A')}"
            )
        else:
            st.warning(f"SI error: {si_result.get('error', 'unknown')}")

    # ── PCI ────────────────────────────────────────────────────────────
    with pci_col:
        st.markdown("**Precipitation Concentration Index (Oliver 1980)**")
        if "error" not in pci_result:
            st.metric(
                label="PCI Value",
                value=_fmt(pci_result.get("pci_value"), 2),
            )
            st.info(
                f"**Classification:** {pci_result.get('classification', 'N/A')}"
            )
        else:
            st.warning(f"PCI error: {pci_result.get('error', 'unknown')}")

    logger.info("Seasonal characteristics rendered.")


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 7. DEM characteristics
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_dem_characteristics(
    dem_stats: dict[str, Any],
    regression: dict[str, Any] | None = None,
) -> None:
    """Display DEM elevation statistics and elevation-rainfall regression.

    The elevation statistics are shown as a grid of ``st.metric`` widgets.
    If a *regression* dict is provided (with keys ``'slope'``,
    ``'intercept'``, ``'r_squared'``, ``'p_value'``), the regression
    results are displayed below.

    Args:
        dem_stats: Dictionary of DEM statistics as returned by
            :func:`core.dem.process_dem`. Expected keys include
            ``'min_elevation'``, ``'max_elevation'``, ``'mean_elevation'``,
            ``'elevation_range'``, ``'std_elevation'``,
            ``'median_elevation'``, ``'q25_elevation'``, ``'q75_elevation'``,
            ``'n_valid_pixels'``, ``'resolution'``, and ``'crs'``.
        regression: Optional regression result dict from
            :func:`core.dem.compute_elevation_rainfall_regression`.
    """
    import streamlit as st

    logger.debug("Rendering DEM characteristics")
    st.subheader("🏔️ DEM Characteristics")

    stat_keys = [
        ("min_elevation", "Min Elevation", " m"),
        ("max_elevation", "Max Elevation", " m"),
        ("mean_elevation", "Mean Elevation", " m"),
        ("median_elevation", "Median Elevation", " m"),
        ("elevation_range", "Elevation Range", " m"),
        ("std_elevation", "Std Deviation", " m"),
        ("q25_elevation", "Q25 Elevation", " m"),
        ("q75_elevation", "Q75 Elevation", " m"),
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    rows = [row1, row2]

    for idx, (key, label, unit) in enumerate(stat_keys):
        col_idx = idx % 4
        row_idx = idx // 4
        with rows[row_idx][col_idx]:
            val = dem_stats.get(key)
            st.metric(
                label=label,
                value=f"{_fmt(val)}{unit}" if val is not None else "N/A",
            )

    # Additional info row
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        n_pixels = dem_stats.get("n_valid_pixels")
        st.metric(label="Valid Pixels", value=_fmt(n_pixels, 0))
    with meta_col2:
        resolution = dem_stats.get("resolution")
        if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
            st.metric(
                label="Resolution",
                value=f"{resolution[0]:.1f} × {resolution[1]:.1f} (map units)",
            )
        else:
            st.metric(label="Resolution", value="N/A")

    # ── Elevation-Rainfall Regression ──────────────────────────────────
    if regression is not None:
        st.divider()
        st.markdown("**Elevation–Rainfall Regression**")

        reg_col1, reg_col2, reg_col3 = st.columns(3)
        with reg_col1:
            st.metric(
                label="Slope",
                value=f"{_fmt(regression.get('slope'), 6)} mm/m",
            )
        with reg_col2:
            st.metric(
                label="R²",
                value=_fmt(regression.get("r_squared"), 4),
            )
        with reg_col3:
            p_val = regression.get("p_value")
            sig = "Significant" if p_val is not None and float(p_val) < 0.05 else "Not significant"
            st.metric(
                label="p-value",
                value=_fmt(p_val, 6),
                delta=sig,
            )

        interpretation = regression.get("interpretation", "")
        if interpretation:
            st.caption(f"*{interpretation}*" )

    logger.info("DEM characteristics rendered.")


# ──────────────────────────────────────────────────────────────────────────────────────────────────────
# 8. Comparison mode
# ──────────────────────────────────────────────────────────────────────────────────────────────────────


def render_comparison_mode(
    all_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
) -> None:
    """Render a tabbed comparison of multiple watersheds or time periods.

    Each entry in *all_results* is displayed in its own ``st.tabs`` tab.
    Within each tab the function renders:

    - The region name and metadata (from the settings dict).
    - A summary metrics row.
    - The data table.

    If only a single result is provided, a single non-tabbed section is
    rendered instead.

    Args:
        all_results: Mapping of ``{label: (df, settings_dict)}`` where
            *df* is a results DataFrame and *settings_dict* may contain
            keys like ``'region_name'``, ``'arf_method'``, etc.
    """
    import streamlit as st

    logger.debug(
        "Rendering comparison mode with %d dataset(s)", len(all_results)
    )

    if not all_results:
        st.info("No results to compare. Run at least one analysis first.")
        return

    st.subheader("🔄 Comparison Mode")

    if len(all_results) == 1:
        label, (df, meta) = next(iter(all_results.items()))
        _render_single_comparison_panel(label, df, meta)
    else:
        tabs = st.tabs(list(all_results.keys()))
        for tab, (label, (df, meta)) in zip(tabs, all_results.items()):
            with tab:
                _render_single_comparison_panel(label, df, meta)

    logger.info("Comparison mode rendered for %d dataset(s).", len(all_results))


def _render_single_comparison_panel(
    label: str,
    df: pd.DataFrame,
    meta: dict[str, Any],
) -> None:
    """Render the inner content for a single comparison tab.

    Args:
        label: Display label for the tab.
        df: Results DataFrame.
        meta: Metadata dictionary with optional region name, ARF, etc.
    """
    import streamlit as st

    region_name = meta.get("region_name", label)
    st.caption(f"**Region:** {region_name}")

    # Quick metadata badges
    meta_items: list[str] = []
    if "arf_method" in meta:
        meta_items.append(f"ARF: {meta['arf_method']}")
    if "rainfall_unit" in meta:
        meta_items.append(f"Unit: {meta['rainfall_unit']}")
    if meta_items:
        st.caption(" | ".join(meta_items))

    # Identify rainfall column for quick metrics
    rainfall_col: str | None = None
    for candidate in ("Basin_Rainfall_mm", "Rainfall_mm", "rainfall"):
        if candidate in df.columns:
            rainfall_col = candidate
            break
    if rainfall_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        rainfall_col = numeric_cols[0] if numeric_cols else None

    if rainfall_col is not None:
        values = df[rainfall_col].astype(float)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Total", value=f"{values.sum():,.1f} mm")
        with m2:
            st.metric(label="Mean", value=f"{values.mean():,.1f} mm")
        with m3:
            st.metric(label="Std Dev", value=f"{values.std():,.1f} mm")

    st.dataframe(df, use_container_width=True, height=380)
