"""ui — Streamlit user-interface components."""

from watershed_analyzer.ui.charts import (
    plot_annual_totals,
    plot_comparison,
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
    render_comparison_mode,
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

__all__ = [
    # charts
    "plot_annual_totals",
    "plot_comparison",
    "plot_elevation_rainfall_scatter",
    "plot_monthly_rainfall_bar",
    "plot_rainfall_trend",
    "plot_spatial_variability",
    "plot_spi_chart",
    # maps
    "create_interactive_map",
    "create_raster_overlay_map",
    "folium_to_html",
    # results
    "render_comparison_mode",
    "render_data_table",
    "render_dem_characteristics",
    "render_drought_analysis",
    "render_seasonal_characteristics",
    "render_spatial_stats_table",
    "render_summary_metrics",
    "render_trend_analysis",
    # upload
    "render_analysis_settings",
    "render_dem_upload",
    "render_netcdf_upload",
    "render_raster_upload",
    "render_region_selector",
    "render_shapefile_upload",
]
