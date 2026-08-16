"""Tests for watershed_analyzer.core.validation.

Covers:
  - validate_raster with a valid single-band MemoryFile
  - validate_raster rejects multi-band rasters
  - validate_raster detects sentinel values
  - check_raster_consistency with consistent and inconsistent rasters
  - mask_sentinel_values correctly masks -9999 and -32768
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine

from watershed_analyzer.config import SENTINEL_VALUES
from watershed_analyzer.core.validation import (
    check_raster_consistency,
    mask_sentinel_values,
    validate_raster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memfile(
    data: np.ndarray,
    count: int = 1,
    crs: str = "EPSG:4326",
    nodata: float | None = None,
    dtype: str = "float32",
) -> rasterio.MemoryFile:
    """Create a rasterio MemoryFile for testing."""
    rows, cols = data.shape
    transform = Affine.translation(0.0, 10.0) * Affine.scale(1.0, -1.0)
    memfile = rasterio.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=rows,
        width=cols,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return memfile


# ---------------------------------------------------------------------------
# 1. validate_raster — valid single-band
# ---------------------------------------------------------------------------

class TestValidateRaster:
    """Tests for validate_raster."""

    def test_valid_single_band_raster(self):
        """A properly formed single-band raster should pass validation."""
        data = np.random.default_rng(seed=0).uniform(0, 50, size=(10, 10))
        memfile = _make_memfile(data.astype(np.float32))
        is_valid, msg, meta = validate_raster(memfile, "test.tif")

        assert is_valid is True
        assert meta is not None
        assert meta["bands"] == 1
        assert meta["width"] == 10
        assert meta["height"] == 10
        assert meta["crs"] == "EPSG:4326"
        assert meta["dtype"] == "float32"
        assert meta["sentinel_detected"] is False

    def test_multi_band_rejected(self):
        """A 3-band raster should be rejected (not single-band)."""
        data = np.ones((10, 10), dtype=np.float32)
        memfile = _make_memfile(data, count=3)
        is_valid, msg, meta = validate_raster(memfile, "rgb.tif")

        assert is_valid is False
        assert meta is None
        assert "3 band" in msg

    def test_no_crs_rejected(self):
        """A raster without a CRS should be rejected."""
        data = np.ones((10, 10), dtype=np.float32)
        rows, cols = data.shape
        transform = Affine.translation(0.0, 10.0) * Affine.scale(1.0, -1.0)
        memfile = rasterio.MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype="float32",
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        is_valid, msg, meta = validate_raster(memfile, "nocrs.tif")

        assert is_valid is False
        assert "no CRS" in msg

    def test_sentinel_value_detected(self):
        """When a raster contains -9999, validation should flag it."""
        data = np.full((10, 10), 10.0, dtype=np.float32)
        data[0, 0] = -9999.0

        memfile = _make_memfile(data)
        is_valid, msg, meta = validate_raster(memfile, "sentinel.tif")

        assert is_valid is True  # sentinel is detected but not a failure
        assert meta["sentinel_detected"] is True
        assert meta["sentinel_value"] == -9999.0

    def test_negative_non_sentinel_counted(self):
        """Negative values that are not sentinel/nodata should be counted."""
        data = np.full((10, 10), 5.0, dtype=np.float32)
        data[0, 0] = -1.0  # Not a sentinel value

        memfile = _make_memfile(data)
        is_valid, msg, meta = validate_raster(memfile, "neg.tif")

        assert is_valid is True
        assert meta["negative_count"] == 1

    def test_all_nodata_rejected(self):
        """A raster where every pixel is nodata should be rejected."""
        data = np.full((5, 5), -9999.0, dtype=np.float32)
        memfile = _make_memfile(data, nodata=-9999.0)
        is_valid, msg, meta = validate_raster(memfile, "allnodata.tif")

        assert is_valid is False
        assert "entirely NoData" in msg


# ---------------------------------------------------------------------------
# 2. check_raster_consistency
# ---------------------------------------------------------------------------

class TestCheckRasterConsistency:
    """Tests for check_raster_consistency."""

    def test_consistent_rasters(self):
        """Two rasters with matching resolution, CRS, and dtype are consistent."""
        meta1 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        meta2 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        is_ok, msg = check_raster_consistency([meta1, meta2])
        assert is_ok is True

    def test_inconsistent_resolution(self):
        """Different resolutions should be flagged."""
        meta1 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        meta2 = {
            "resolution": (0.5, 0.5),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        is_ok, msg = check_raster_consistency([meta1, meta2])
        assert is_ok is False
        assert "Resolution" in msg

    def test_inconsistent_crs(self):
        """Different CRS should be flagged."""
        meta1 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        meta2 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:3857",
            "dtype": "float32",
        }
        is_ok, msg = check_raster_consistency([meta1, meta2])
        assert is_ok is False
        assert "CRS" in msg

    def test_inconsistent_dtype(self):
        """Different dtypes should be flagged."""
        meta1 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        meta2 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float64",
        }
        is_ok, msg = check_raster_consistency([meta1, meta2])
        assert is_ok is False
        assert "Dtype" in msg

    def test_single_raster_trivially_consistent(self):
        """A single-element list should return True."""
        meta1 = {
            "resolution": (0.1, 0.1),
            "crs": "EPSG:4326",
            "dtype": "float32",
        }
        is_ok, msg = check_raster_consistency([meta1])
        assert is_ok is True


# ---------------------------------------------------------------------------
# 3. mask_sentinel_values
# ---------------------------------------------------------------------------

class TestMaskSentinelValues:
    """Tests for mask_sentinel_values."""

    def test_mask_minus_9999(self):
        """Pixels with value -9999 should be replaced with np.nan."""
        data = np.array([[10.0, -9999.0, 10.0],
                         [10.0, 10.0, -9999.0]], dtype=np.float64)
        masked, valid = mask_sentinel_values(data, nodata=None)

        assert np.isnan(masked[0, 1])
        assert np.isnan(masked[1, 2])
        assert masked[0, 0] == 10.0
        assert masked[0, 2] == 10.0
        assert bool(valid[0, 0]) is True
        assert bool(valid[0, 1]) is False
        assert bool(valid[1, 2]) is False

    def test_mask_minus_32768(self):
        """Pixels with value -32768 should be replaced with np.nan."""
        data = np.array([[5.0, -32768.0],
                         [5.0, 5.0]], dtype=np.float64)
        masked, valid = mask_sentinel_values(data, nodata=None)

        assert np.isnan(masked[0, 1])
        assert bool(valid[0, 1]) is False
        assert valid.sum() == 3  # only one masked

    def test_mask_nodata_value(self):
        """Explicit nodata value should also be masked."""
        data = np.array([[1.0, -9999.0],
                         [1.0, 1.0]], dtype=np.float64)
        masked, valid = mask_sentinel_values(data, nodata=-9999.0)

        # -9999 matches both nodata and sentinel; should still be masked once
        assert np.isnan(masked[0, 1])
        assert valid.sum() == 3

    def test_no_sentinels_all_valid(self):
        """Data with no sentinel or nodata values should remain untouched."""
        data = np.full((5, 5), 42.0, dtype=np.float64)
        masked, valid = mask_sentinel_values(data)

        assert np.allclose(masked, 42.0)
        assert valid.all()

    def test_custom_sentinel_values(self):
        """Should accept custom sentinel_values tuple."""
        data = np.array([[1.0, -777.0],
                         [1.0, 1.0]], dtype=np.float64)
        masked, valid = mask_sentinel_values(
            data, nodata=None, sentinel_values=(-777.0,)
        )

        assert np.isnan(masked[0, 1])
        assert bool(valid[0, 1]) is False

    def test_mask_preserves_shape(self):
        """Output arrays should have the same shape as input."""
        data = np.ones((7, 9), dtype=np.float64)
        masked, valid = mask_sentinel_values(data)

        assert masked.shape == data.shape
        assert valid.shape == data.shape
        assert valid.dtype == bool
        assert masked.dtype == np.float64
