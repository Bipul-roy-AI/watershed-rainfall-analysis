"""Interactive folium maps for the Watershed Analyzer UI.

This module provides functions to create Folium-based interactive maps
suitable for embedding inside a Streamlit application via
``st.components.v1.html``.

Functions:
    create_interactive_map: Region map with optional choropleth highlighting.
    folium_to_html: Render a Folium map to a styled HTML string.
    create_raster_overlay_map: Region map with optional raster image overlay.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any

import folium
import geopandas as gpd
import numpy as np

if TYPE_CHECKING:
    import rasterio

logger = logging.getLogger(__name__)

# ── Default style constants ────────────────────────────────────────────────

_DEFAULT_TILES = "CartoDB positron"
_DEFAULT_ZOOM = 8
_MAP_WIDTH_PX = "100%"
_MAP_HEIGHT_PX = "500px"

_BASE_STYLE = {
    "fillColor": "#e5e7eb",
    "color": "#9ca3af",
    "weight": 1,
    "fillOpacity": 0.4,
}

_SELECTED_STYLE = {
    "fillColor": "#3b82f6",
    "color": "#1d4ed8",
    "weight": 3,
    "fillOpacity": 0.5,
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _compute_center(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    """Return the centroid (lat, lon) of the combined geometry.

    Args:
        gdf: GeoDataFrame used to compute the visual centre.

    Returns:
        Tuple of ``(latitude, longitude)``.
    """
    merged = gdf.geometry.unary_union
    centroid = merged.centroid
    return float(centroid.y), float(centroid.x)


def _geojson_style(feature: dict[str, Any]) -> dict[str, Any]:
    """Return a simple default style dict for a GeoJSON feature.

    This is used as the ``style_function`` callback for Folium GeoJson layers.

    Args:
        feature: A GeoJSON feature dictionary (unused but required by the
            Folium API contract).

    Returns:
        Style dictionary.
    """
    return _BASE_STYLE.copy()


def _highlight_style(feature: dict[str, Any]) -> dict[str, Any]:
    """Return an emphasized style dict for hover-highlighted features.

    Args:
        feature: A GeoJSON feature dictionary (unused but required by the
            Folium API contract).

    Returns:
        Style dictionary.
    """
    return {
        "fillColor": "#2563eb",
        "color": "#1e40af",
        "weight": 3,
        "fillOpacity": 0.65,
    }


# ── Public API ──────────────────────────────────────────────────────────────


def create_interactive_map(
    gdf: gpd.GeoDataFrame,
    selected_gdf: gpd.GeoDataFrame | None = None,
    highlight_column: str | None = None,
) -> folium.Map:
    """Create a Folium map centred on the study region.

    All regions in *gdf* are drawn with a light-grey fill and thin border.
    If *selected_gdf* is provided, its geometries are drawn on top with a
    blue fill and thick border.  A tooltip showing attribute values is
    attached to every layer.

    When *highlight_column* is supplied, a choropleth colour scale is
    applied to the *selected* layer using that column's values.

    Args:
        gdf: GeoDataFrame containing all watershed/region polygons.
        selected_gdf: Optional GeoDataFrame of highlighted (selected) regions.
        highlight_column: Column name in *selected_gdf* whose values drive
            the choropleth colour scale.

    Returns:
        A ``folium.Map`` instance ready for rendering.
    """
    logger.debug(
        "Creating interactive map (n_all=%d, n_selected=%d, highlight=%s)",
        len(gdf),
        len(selected_gdf) if selected_gdf is not None else 0,
        highlight_column,
    )

    center_lat, center_lon = _compute_center(gdf)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=_DEFAULT_ZOOM,
        tiles=_DEFAULT_TILES,
        control_scale=True,
    )

    # Determine a sensible tooltip field: prefer the highlight column, then
    # fall back to the first non-geometry string column.
    tooltip_fields: list[str] = []
    if highlight_column and highlight_column in gdf.columns:
        tooltip_fields.append(highlight_column)
    for col in gdf.columns:
        if col == "geometry" or col in tooltip_fields:
            continue
        if gdf[col].dtype in ("object", "string"):
            tooltip_fields.append(col)
            break

    # Layer: all regions (base)
    all_geojson = json.loads(gdf.to_json())
    folium.GeoJson(
        all_geojson,
        style_function=_geojson_style,
        highlight_function=_highlight_style,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, sticky=True),
        name="All Regions",
    ).add_to(m)

    # Layer: selected regions
    if selected_gdf is not None and not selected_gdf.empty:
        sel_tooltip: list[str] = []
        if highlight_column and highlight_column in selected_gdf.columns:
            sel_tooltip.append(highlight_column)
        for col in selected_gdf.columns:
            if col == "geometry" or col in sel_tooltip:
                continue
            if selected_gdf[col].dtype in ("object", "string"):
                sel_tooltip.append(col)
                break

        if highlight_column and highlight_column in selected_gdf.columns:
            # Choropleth mode
            choropleth = folium.Choropleth(
                geo_data=json.loads(selected_gdf.to_json()),
                data=selected_gdf,
                columns=[selected_gdf.index.name or "index", highlight_column],
                key_on="feature.id",
                fill_color="YlGnBu",
                fill_opacity=0.6,
                line_opacity=1,
                name="Selected (choropleth)",
                highlight=True,
            )
            choropleth.add_to(m)
            # Attach a tooltip to the GeoJson layer inside the choropleth
            for child in choropleth._children.values():
                if isinstance(child, folium.features.GeoJson):
                    child.tooltip = folium.GeoJsonTooltip(fields=sel_tooltip, sticky=True)
        else:
            # Plain highlighted polygons
            sel_geojson = json.loads(selected_gdf.to_json())
            folium.GeoJson(
                sel_geojson,
                style_function=lambda _: _SELECTED_STYLE.copy(),
                highlight_function=_highlight_style,
                tooltip=folium.GeoJsonTooltip(fields=sel_tooltip, sticky=True),
                name="Selected Region",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def folium_to_html(m: folium.Map) -> str:
    """Render a Folium map to a styled HTML string.

    The returned string is intended to be embedded in a Streamlit app
    via ``st.components.v1.html(html, height=500)``.

    Args:
        m: A ``folium.Map`` instance.

    Returns:
        HTML string with inline styles that constrain the map container to
        100 % width and 500 px height.
    """
    logger.debug("Rendering folium map to HTML string")

    raw_html = m._repr_html_()
    styled_html = (
        f'<div style="width:{_MAP_WIDTH_PX}; height:{_MAP_HEIGHT_PX}; '
        f'border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.12);">'
        f"{raw_html}"
        f"</div>"
    )
    return styled_html


def create_raster_overlay_map(
    gdf: gpd.GeoDataFrame,
    raster_memfile: rasterio.MemoryFile | None = None,
    raster_title: str = "Rainfall",
) -> folium.Map:
    """Create a Folium map with an optional raster image overlay.

    The base map is centred on the geometries in *gdf*.  When a
    *raster_memfile* is supplied, the raster band is normalised, coloured
    using the ``viridis`` matplotlib colormap, converted to a PNG, and
    overlaid via ``folium.raster_layers.ImageOverlay``.

    Args:
        gdf: GeoDataFrame used to centre and frame the map.
        raster_memfile: Optional ``rasterio.MemoryFile`` containing a
            2-D rainfall (or similar) raster.
        raster_title: Title shown in the map layer control.

    Returns:
        A ``folium.Map`` instance.
    """
    logger.debug(
        "Creating raster overlay map (raster=%s, title=%s)",
        raster_memfile is not None,
        raster_title,
    )

    center_lat, center_lon = _compute_center(gdf)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=_DEFAULT_ZOOM,
        tiles=_DEFAULT_TILES,
        control_scale=True,
    )

    # Boundary outline from gdf
    boundary_geojson = json.loads(gdf.to_json())
    folium.GeoJson(
        boundary_geojson,
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#374151",
            "weight": 1.5,
            "fillOpacity": 0,
        },
        name="Watershed Boundary",
    ).add_to(m)

    if raster_memfile is not None:
        try:
            import matplotlib  # noqa: F401 – ensure colormaps available
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            import rasterio

            with raster_memfile.open() as src:
                band = src.read(1)
                transform = src.transform
                bounds = src.bounds  # left, bottom, right, top

            # Mask nodata
            nodata = getattr(src, "nodata", None) if "src" in dir() else None
            masked = np.ma.masked_invalid(band)
            if nodata is not None:
                masked = np.ma.masked_equal(band, nodata)

            # Normalise to 0-1 for the colormap
            vmin = float(masked.min()) if masked.count() > 0 else 0.0
            vmax = float(masked.max()) if masked.count() > 0 else 1.0
            if vmax - vmin < 1e-10:
                vmax = vmin + 1.0  # avoid zero-range

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")
            coloured = cmap(norm(masked.filled(np.nan)))  # RGBA array

            # Set fully transparent where data is masked
            alpha_mask = masked.mask
            coloured[..., 3] = np.where(alpha_mask, 0, 180)  # ~70 % opacity

            # Encode to PNG
            buf = io.BytesIO()
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(coloured.shape[1] / 100, coloured.shape[0] / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(coloured, aspect="auto")
            ax.axis("off")
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
            plt.close(fig)
            png_bytes = buf.getvalue()
            buf.close()

            # Encode to base64 data URI
            import base64

            b64 = base64.b64encode(png_bytes).decode("utf-8")
            image_url = f"data:image/png;base64,{b64}"

            # folium ImageOverlay expects [[south, west], [north, east]]
            image_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

            folium.raster_layers.ImageOverlay(
                image=image_url,
                bounds=image_bounds,
                opacity=0.7,
                name=raster_title,
                interactive=True,
                show=True,
            ).add_to(m)

            # Add a colour-bar legend (simple gradient image)
            gradient, _ = cm.get_cmap("viridis"), None
            legend_buf = io.BytesIO()
            fig_legend = plt.figure(figsize=(3, 0.4), dpi=100)
            ax_legend = fig_legend.add_axes([0.05, 0.3, 0.9, 0.4])
            cb = matplotlib.colorbar.ColorbarBase(ax_legend, cmap=gradient, norm=norm, orientation="horizontal")
            cb.set_label(raster_title, fontsize=9)
            fig_legend.savefig(legend_buf, format="png", bbox_inches="tight")
            plt.close(fig_legend)
            legend_b64 = base64.b64encode(legend_buf.getvalue()).decode("utf-8")
            legend_buf.close()

            # Embed legend in bottom-right
            legend_html = (
                f'<div style="position:fixed; bottom:30px; right:10px; z-index:1000; '
                f'background:white; padding:4px 8px; border-radius:4px; '
                f'box-shadow:0 1px 3px rgba(0,0,0,0.2);">'
                f'<img src="data:image/png;base64,{legend_b64}" style="width:180px;">'
                f"</div>"
            )
            m.get_root().html.add_child(folium.Element(legend_html))

            logger.info("Raster overlay added (%.1f × %.1f extent).", bounds.right - bounds.left, bounds.top - bounds.bottom)

        except Exception as exc:
            logger.error("Failed to create raster overlay: %s", exc, exc_info=True)

    folium.LayerControl().add_to(m)
    return m
