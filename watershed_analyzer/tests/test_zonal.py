"""Tests for watershed_analyzer.core.zonal and watershed_analyzer.core.arf.

Covers:
  - hydrologically_correct_zonal_stats with synthetic 10×10 rasters
  - ARF application and volume calculation
  - Sentinel-value masking (exclusion of -9999 pixels)
  - compute_arf edge cases (area=0, method='none', negative area)
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from watershed_analyzer.core.arf import compute_arf
from watershed_analyzer.core.zonal import hydrologically_correct_zonal_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raster(
    data: np.ndarray,
    transform: Affine | None = None,
    crs: str = "EPSG:4326",
    nodata: float | None = None,
    dtype: str = "float64",
) -> rasterio.MemoryFile:
    """Create a single-band rasterio MemoryFile from a 2-D numpy array."""
    if transform is None:
        transform = Affine.translation(0.0, 10.0) * Affine.scale(1.0, -1.0)
    rows, cols = data.shape
    memfile = rasterio.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return memfile


def _make_polygon(xmin: float, ymin: float, xmax: float, ymax: float) -> GeoDataFrame:
    """Create a single-row GeoDataFrame with a square polygon in EPSG:4326."""
    poly = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# 1. Zonal stats — uniform raster, correct mean
# ---------------------------------------------------------------------------

class TestHydrologicallyCorrectZonalStats:
    """Tests for hydrologically_correct_zonal_stats."""

    def test_uniform_raster_returns_correct_mean(self):
        """A 10×10 raster with all values = 10.0 mm should yield mean = 10.0."""
        data = np.full((10, 10), 10.0, dtype=np.float64)
        memfile = _make_raster(data)

        # Polygon covering center 5×5 pixels: x ∈ [2.5, 7.5], y ∈ [2.5, 7.5]
        poly = _make_polygon(2.5, 2.5, 7.5, 7.5)

        result = hydrologically_correct_zonal_stats(
            polygon=poly,
            raster_memfile=memfile,
            arf_method="none",
        )

        assert result["spatial_mean_mm"] == pytest.approx(10.0, abs=1e-3)

    def test_arf_is_applied(self):
        """With ARF method != 'none', basin_rainfall_mm should equal mean × ARF."""
        data = np.full((10, 10), 10.0, dtype=np.float64)
        memfile = _make_raster(data)

        poly = _make_polygon(2.5, 2.5, 7.5, 7.5)

        result = hydrologically_correct_zonal_stats(
            polygon=poly,
            raster_memfile=memfile,
            arf_method="srikanthan_mcmahon",
        )

        arf = result["arf"]
        assert 0.5 <= arf <= 1.0  # ARF should be in valid range
        # basin_rainfall = spatial_mean × ARF
        assert result["basin_rainfall_mm"] == pytest.approx(
            result["spatial_mean_mm"] * arf, abs=1e-3
        )

    def test_volume_calculation(self):
        """Volume should equal basin_rainfall_mm × 1e-3 × basin_area_km2 × 1e6."""
        data = np.full((10, 10), 10.0, dtype=np.float64)
        memfile = _make_raster(data)

        poly = _make_polygon(2.5, 2.5, 7.5, 7.5)

        result = hydrologically_correct_zonal_stats(
            polygon=poly,
            raster_memfile=memfile,
            arf_method="none",
        )

        expected_volume = (
            result["basin_rainfall_mm"]
            * 1e-3
            * result["basin_area_km2"]
            * 1e6
        )
        assert result["volume_m3"] == pytest.approx(expected_volume, rel=1e-3)

    def test_sentinel_values_are_masked(self):
        """Pixels with -9999 should be excluded from statistics."""
        data = np.full((10, 10), 10.0, dtype=np.float64)
        # Set a block of sentinel pixels in the center
        data[4:6, 4:6] = -9999.0

        memfile = _make_raster(data)

        # Polygon covers the entire raster
        poly = _make_polygon(0.0, 0.0, 10.0, 10.0)

        result = hydrologically_correct_zonal_stats(
            polygon=poly,
            raster_memfile=memfile,
            arf_method="none",
        )

        # Mean should still be 10.0 because sentinel pixels are excluded
        assert result["spatial_mean_mm"] == pytest.approx(10.0, abs=1e-3)
        # Should have fewer valid pixels than total
        assert result["n_valid_pixels"] < result["n_total_pixels"]

    def test_result_keys_present(self):
        """The returned dict should contain all documented keys."""
        data = np.full((10, 10), 10.0, dtype=np.float64)
        memfile = _make_raster(data)
        poly = _make_polygon(2.5, 2.5, 7.5, 7.5)

        result = hydrologically_correct_zonal_stats(
            polygon=poly,
            raster_memfile=memfile,
            arf_method="none",
        )

        expected_keys = {
            "n_valid_pixels", "n_total_pixels", "coverage_pct",
            "spatial_mean_mm", "spatial_std_mm", "spatial_cv_pct",
            "min_mm", "max_mm", "median_mm", "q25_mm", "q75_mm",
            "basin_area_km2", "arf", "arf_method",
            "basin_rainfall_mm", "volume_m3", "pixel_area_m2",
            "rainfall_unit", "equal_area_crs",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 2. compute_arf tests
# ---------------------------------------------------------------------------

class TestComputeARF:
    """Tests for watershed_analyzer.core.arf.compute_arf."""

    def test_zero_area_returns_one(self):
        """ARF for area = 0 km² should be 1.0 (point rainfall)."""
        assert compute_arf(0.0) == pytest.approx(1.0)

    def test_small_area_less_than_one(self):
        """For a positive area < 50 km², ARF should be slightly below 1.0."""
        arf = compute_arf(1.0, method="srikanthan_mcmahon")
        assert arf < 1.0
        assert arf >= 0.5

    def test_method_none_returns_one(self):
        """ARF method 'none' should always return 1.0 regardless of area."""
        assert compute_arf(100.0, method="none") == pytest.approx(1.0)
        assert compute_arf(10000.0, method="none") == pytest.approx(1.0)

    def test_negative_area_raises_value_error(self):
        """Negative area_km2 should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_arf(-5.0)

    def test_unknown_method_raises_value_error(self):
        """An unrecognised ARF method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown ARF method"):
            compute_arf(100.0, method="bogus_method")

    def test_usgs_reed_small_area(self):
        """USGS Reed ARF for area < 10 km² should be 1.0."""
        arf = compute_arf(5.0, method="usgs_reed", duration_hr=24.0)
        assert arf == pytest.approx(1.0)

    def test_usgs_reed_known_value(self):
        """USGS Reed: verify ARF at a known table point (100 km², 24 h)."""
        # From table: 24h at 100 km² = 0.97
        arf = compute_arf(100.0, method="usgs_reed", duration_hr=24.0)
        assert arf == pytest.approx(0.97, abs=0.005)

    def test_srikanthan_mcmahon_clamping(self):
        """ARF should be clamped to [0.5, 1.0] even for very large areas."""
        arf = compute_arf(1e8, method="srikanthan_mcmahon")
        assert 0.5 <= arf <= 1.0
