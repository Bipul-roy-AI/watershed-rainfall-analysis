"""Interactive Plotly charts for the Watershed Analyzer UI.

This module provides a collection of Plotly-based chart functions that replace
legacy matplotlib visualizations with fully interactive equivalents.  Each
function returns a ``plotly.graph_objects.Figure`` instance that can be
directly rendered in a Streamlit application via ``st.plotly_chart``.

Functions:
    plot_monthly_rainfall_bar: Bar chart with min/max highlighting and mean line.
    plot_rainfall_trend: Line + scatter plot with optional Mann-Kendall annotation.
    plot_spi_chart: Multi-scale SPI classification chart.
    plot_elevation_rainfall_scatter: Scatter plot with optional regression overlay.
    plot_spatial_variability: Dual y-axis bar + line chart for std dev and CV.
    plot_annual_totals: Annual aggregation bar chart (conditional on year data).
    plot_comparison: Overlaid line chart for multi-watershed / multi-period comparison.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

if TYPE_CHECKING:
    import streamlit as st

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_BASE_LAYOUT = go.Layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Open Sans, Arial, sans-serif", size=12, color="#333"),
    margin=dict(l=60, r=30, t=50, b=60),
    xaxis=dict(showgrid=False, linecolor="#ccc"),
    yaxis=dict(showgrid=True, gridcolor="#eee", linecolor="#ccc", zeroline=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def _apply_base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared white-background layout to a figure.

    Args:
        fig: Plotly figure to style.
        title: Chart title.

    Returns:
        The same figure with updated layout.
    """
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)), **_BASE_LAYOUT)
    return fig


def _bar_color_gradient(n: int, base_rgb: tuple[int, int, int] = (37, 99, 235)) -> list[str]:
    """Generate *n* colour strings graduating from light to the base colour.

    Args:
        n: Number of colours to produce.
        base_rgb: Target (R, G, B) tuple for the darkest shade.

    Returns:
        List of ``#rrggbb`` hex strings.
    """
    colours: list[str] = []
    for i in range(n):
        frac = (i + 1) / n  # 0→1, lightest → darkest
        r = int(220 + (base_rgb[0] - 220) * frac)
        g = int(220 + (base_rgb[1] - 220) * frac)
        b = int(220 + (base_rgb[2] - 220) * frac)
        colours.append(f"rgb({r},{g},{b})")
    return colours


# ──────────────────────────────────────────────────────────────────────────────
# Public chart functions
# ──────────────────────────────────────────────────────────────────────────────


def plot_monthly_rainfall_bar(
    df: pd.DataFrame,
    month_col: str = "Month",
    value_col: str = "Basin_Rainfall_mm",
    title: str = "Monthly Basin Rainfall",
) -> go.Figure:
    """Render an interactive bar chart of monthly rainfall.

    Produces a colour-gradient blue bar for each month, highlights the
    maximum value in green and the minimum in red, and overlays a
    horizontal dashed line at the arithmetic mean.

    Args:
        df: DataFrame containing at least *month_col* and *value_col*.
        month_col: Column name for month labels.
        value_col: Column name for rainfall values (mm).
        title: Chart title.

    Returns:
        A styled ``go.Figure`` ready for ``st.plotly_chart``.
    """
    logger.debug("Building monthly rainfall bar chart (title=%s)", title)

    values = df[value_col].astype(float)
    months = df[month_col].astype(str)
    n = len(df)

    # Determine extrema indices
    max_idx = int(values.idxmax())
    min_idx = int(values.idxmin())
    mean_val = float(values.mean())

    # Colour array – gradient blue, then override max / min
    colours = _bar_color_gradient(n)
    colours[max_idx] = "#22c55e"  # green
    colours[min_idx] = "#ef4444"  # red

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months,
            y=values,
            marker_color=colours,
            hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
            name="Rainfall",
        )
    )

    # Mean reference line
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text=f"Mean: {mean_val:.1f} mm",
        annotation_position="top right",
    )

    fig = _apply_base_layout(fig, title)
    fig.update_yaxes(title_text="Rainfall (mm)")
    fig.update_xaxes(title_text="Month")
    return fig


def plot_rainfall_trend(
    df: pd.DataFrame,
    month_col: str,
    value_col: str,
    title: str,
) -> go.Figure:
    """Render a rainfall trend line chart with optional Mann-Kendall annotation.

    Displays a scatter plot with connecting lines and a semi-transparent fill,
    plus a linear-regression trendline.  If the Streamlit session state
    contains Mann-Kendall test results (key ``mann_kendall``) an annotation
    block showing trend direction, Z-statistic, and p-value is added.

    Args:
        df: DataFrame with month and value columns.
        month_col: Column for x-axis labels.
        value_col: Column for y-axis values.
        title: Chart title.

    Returns:
        A styled ``go.Figure``.
    """
    logger.debug("Building rainfall trend chart (title=%s)", title)

    x = np.arange(len(df))
    y = df[value_col].astype(float).values
    labels = df[month_col].astype(str).tolist()

    fig = go.Figure()

    # Filled area + line + scatter
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=y,
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=6, color="#2563eb"),
            hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
            name="Rainfall",
        )
    )

    # Linear trendline
    if len(x) >= 2:
        slope, intercept, r_value, _p, _se = stats.linregress(x, y)
        trend_y = slope * x + intercept
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=trend_y,
                mode="lines",
                line=dict(dash="dash", color="#f97316", width=2),
                hovertemplate="Trend: %{y:.1f} mm<extra></extra>",
                name=f"Trend (r²={r_value**2:.3f})",
            )
        )

    # Mann-Kendall annotation (if available in session state)
    try:
        import streamlit as st

        mk: dict | None = st.session_state.get("mann_kendall")
        if mk is not None:
            trend_txt = mk.get("trend", "N/A")
            z_stat = mk.get("z_stat", "N/A")
            p_val = mk.get("p_value", "N/A")
            annotation = (
                f"Mann-Kendall Test<br>"
                f"Trend: <b>{trend_txt}</b><br>"
                f"Z: {z_stat}<br>"
                f"p-value: {p_val}"
            )
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.97,
                xanchor="right",
                yanchor="top",
                text=annotation,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#d1d5db",
                borderwidth=1,
                font=dict(size=11),
            )
    except Exception as exc:
        logger.debug("Could not read Mann-Kendall from session_state: %s", exc)

    fig = _apply_base_layout(fig, title)
    fig.update_yaxes(title_text="Rainfall (mm)")
    fig.update_xaxes(title_text="Month")
    return fig


def plot_spi_chart(
    spi_df: pd.DataFrame,
    scales: list[int] | None = None,
) -> go.Figure:
    """Render a multi-scale SPI (Standardized Precipitation Index) chart.

    Each SPI scale is drawn as a separate line.  Segments are colour-coded
    by drought / wet classification:

    * **SPI > 1** → green (moderately wet or wetter)
    * **−1 ≤ SPI ≤ 1** → gray (near normal)
    * **SPI < −1** → red (moderately dry or drier)

    Horizontal dashed reference lines are drawn at ±1 and ±2.

    Args:
        spi_df: DataFrame where columns are named ``SPI_1``, ``SPI_3``, etc.
        scales: Optional list of SPI scales to plot.  If *None*, all columns
            whose names start with ``SPI_`` are used.

    Returns:
        A styled ``go.Figure``.
    """
    logger.debug("Building SPI chart (scales=%s)", scales)

    # Auto-detect scale columns
    if scales is None:
        spi_cols = [c for c in spi_df.columns if c.upper().startswith("SPI")]
    else:
        spi_cols = [f"SPI_{s}" for s in scales]
        # Fall back to case-insensitive match
        available = {c.upper(): c for c in spi_df.columns}
        spi_cols = [available.get(sc.upper(), sc) for sc in spi_cols]

    # Use index as x-axis (assumed to be time-like)
    x_vals = spi_df.index.astype(str).tolist()

    # Palette for different scales
    scale_palette = {
        "SPI_1": "#2563eb",
        "SPI_3": "#7c3aed",
        "SPI_6": "#db2777",
        "SPI_9": "#ea580c",
        "SPI_12": "#16a34a",
        "SPI_24": "#0891b2",
        "SPI_36": "#4f46e5",
        "SPI_48": "#b91c1c",
    }

    fig = go.Figure()
    for col in spi_cols:
        if col not in spi_df.columns:
            logger.warning("SPI column '%s' not found in DataFrame – skipping.", col)
            continue

        y_vals = spi_df[col].astype(float).values
        base_color = scale_palette.get(col, "#6b7280")

        # Segment-wise colouring: we draw one trace per classification bucket
        # for correct colour transitions.  For performance we use a single
        # trace with per-point colour via `marker.color` and a single line
        # with the base colour.
        point_colors = [
            "#22c55e" if v > 1 else ("#ef4444" if v < -1 else "#9ca3af")
            for v in y_vals
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                line=dict(color=base_color, width=1.5),
                marker=dict(size=4, color=point_colors),
                name=col,
                hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
            )
        )

    # Reference lines
    for ref, label in [(2, "Extremely Wet"), (1, "Moderately Wet"), (-1, "Moderately Dry"), (-2, "Extremely Dry")]:
        fig.add_hline(
            y=ref,
            line_dash="dash",
            line_color="#d1d5db",
            line_width=1,
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=9, color="#6b7280"),
        )

    title = "Standardized Precipitation Index (SPI)"
    fig = _apply_base_layout(fig, title)
    fig.update_yaxes(title_text="SPI Value", zeroline=True, zerolinecolor="#d1d5db")
    fig.update_xaxes(title_text="Time Period")
    return fig


def plot_elevation_rainfall_scatter(
    elevation: np.ndarray,
    rainfall: np.ndarray,
    regression: dict | None = None,
) -> go.Figure:
    """Render a scatter plot of elevation vs. rainfall with optional regression.

    Point colours are mapped to the rainfall value via a sequential colourscale.

    Args:
        elevation: 1-D array of elevation values (m).
        rainfall: 1-D array of rainfall values (mm), same length as *elevation*.
        regression: Optional dict with keys ``slope``, ``intercept``, and
            ``r_squared`` to overlay a regression line.

    Returns:
        A styled ``go.Figure``.
    """
    logger.debug(
        "Building elevation-rainfall scatter (n=%d, has_regression=%s)",
        len(elevation),
        regression is not None,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=elevation,
            y=rainfall,
            mode="markers",
            marker=dict(
                size=10,
                color=rainfall,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Rainfall (mm)"),
                line=dict(width=0.5, color="#fff"),
            ),
            hovertemplate="Elev: %{x:.0f} m | Rain: %{y:.1f} mm<extra></extra>",
            name="Stations",
        )
    )

    if regression is not None:
        slope = float(regression.get("slope", 0))
        intercept = float(regression.get("intercept", 0))
        r_sq = float(regression.get("r_squared", 0))
        x_range = np.linspace(elevation.min(), elevation.max(), 100)
        y_range = slope * x_range + intercept
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                line=dict(dash="dash", color="#f97316", width=2),
                name=f"Regression (R²={r_sq:.3f})",
                hovertemplate="Predicted: %{y:.1f} mm<extra></extra>",
            )
        )

    fig = _apply_base_layout(fig, "Elevation vs. Rainfall")
    fig.update_xaxes(title_text="Elevation (m)")
    fig.update_yaxes(title_text="Rainfall (mm)")
    return fig


def plot_spatial_variability(
    df: pd.DataFrame,
    month_col: str,
    std_col: str,
    cv_col: str,
) -> go.Figure:
    """Dual y-axis chart showing spatial variability across months.

    The left y-axis displays standard deviation as a bar chart, while the
    right y-axis displays the coefficient of variation (CV %) as a line.

    Args:
        df: DataFrame with month, std-dev, and CV columns.
        month_col: Column for month labels.
        std_col: Column for standard deviation values.
        cv_col: Column for coefficient of variation values (%).

    Returns:
        A styled ``go.Figure`` with two y-axes.
    """
    logger.debug("Building spatial variability chart")

    months = df[month_col].astype(str).tolist()
    std_vals = df[std_col].astype(float).values
    cv_vals = df[cv_col].astype(float).values

    fig = go.Figure()

    # Bars – std dev (left axis)
    fig.add_trace(
        go.Bar(
            x=months,
            y=std_vals,
            name="Std Dev (mm)",
            marker_color="rgba(37,99,235,0.55)",
            marker_line_color="rgba(37,99,235,1)",
            marker_line_width=1,
            hovertemplate="%{x}: %{y:.1f} mm<extra>Std Dev</extra>",
            yaxis="y",
        )
    )

    # Line – CV % (right axis)
    fig.add_trace(
        go.Scatter(
            x=months,
            y=cv_vals,
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7, color="#ef4444"),
            name="CV (%)",
            hovertemplate="%{x}: %{y:.1f}%<extra>CV</extra>",
            yaxis="y2",
        )
    )

    fig.update_layout(
        yaxis=dict(
            title=dict(text="Standard Deviation (mm)", font=dict(size=12)),
            side="left",
            showgrid=True,
            gridcolor="#eee",
            linecolor="#ccc",
        ),
        yaxis2=dict(
            title=dict(text="CV (%)", font=dict(size=12)),
            side="right",
            overlaying="y",
            showgrid=False,
            linecolor="#ccc",
        ),
    )

    fig = _apply_base_layout(fig, "Spatial Variability of Rainfall")
    fig.update_xaxes(title_text="Month")
    return fig


def plot_annual_totals(df: pd.DataFrame) -> go.Figure | None:
    """Aggregate data to annual totals and render a bar chart.

    The function looks for a column named ``Year`` (case-insensitive) in the
    DataFrame.  If none is found, *None* is returned and the caller can
    decide how to handle the absence of yearly data.

    Args:
        df: DataFrame that ideally contains a ``Year`` column alongside
            numeric rainfall columns.

    Returns:
        A styled ``go.Figure``, or *None* if no year column is detected.
    """
    year_col: str | None = None
    for candidate in df.columns:
        if candidate.strip().lower() == "year":
            year_col = candidate
            break

    if year_col is None:
        logger.info("No 'Year' column found – skipping annual totals chart.")
        return None

    logger.debug("Building annual totals chart from column '%s'", year_col)

    # Identify numeric (non-year) columns to sum
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != year_col]
    if not numeric_cols:
        logger.warning("No numeric columns to aggregate for annual totals.")
        return None

    annual = df.groupby(year_col)[numeric_cols].sum().reset_index()

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, col in enumerate(numeric_cols):
        colour = palette[i % len(palette)]
        fig.add_trace(
            go.Bar(
                x=annual[year_col].astype(str),
                y=annual[col],
                name=col,
                marker_color=colour,
                hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
            )
        )

    fig = _apply_base_layout(fig, "Annual Rainfall Totals")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Total Rainfall (mm)")
    fig.update_layout(barmode="group")
    return fig


def plot_comparison(results_dict: dict[str, pd.DataFrame]) -> go.Figure:
    """Overlaid line chart comparing multiple watersheds or time periods.

    Each entry in *results_dict* is rendered as a separate line trace.  The
    x-axis is taken from the first column of each DataFrame and the y-axis
    from the second column.

    Args:
        results_dict: Mapping of ``{label: DataFrame}`` where each
            DataFrame has at least two columns (x-values and y-values).

    Returns:
        A styled ``go.Figure``.

    Raises:
        ValueError: If *results_dict* is empty.
    """
    if not results_dict:
        raise ValueError("results_dict must contain at least one entry.")

    logger.debug("Building comparison chart with %d series", len(results_dict))

    palette = px.colors.qualitative.Set2
    fig = go.Figure()

    for idx, (label, sub_df) in enumerate(results_dict.items()):
        cols = sub_df.columns.tolist()
        if len(cols) < 2:
            logger.warning("Skipping '%s' – DataFrame needs ≥ 2 columns.", label)
            continue

        x_col, y_col = cols[0], cols[1]
        colour = palette[idx % len(palette)]

        fig.add_trace(
            go.Scatter(
                x=sub_df[x_col].astype(str),
                y=sub_df[y_col].astype(float),
                mode="lines+markers",
                line=dict(color=colour, width=2),
                marker=dict(size=5, color=colour),
                name=label,
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.1f}} mm<extra></extra>",
            )
        )

    fig = _apply_base_layout(fig, "Watershed / Period Comparison")
    fig.update_yaxes(title_text="Rainfall (mm)")
    return fig
