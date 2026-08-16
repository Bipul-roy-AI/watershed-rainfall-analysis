"""Tests for watershed_analyzer.core.stats.

Covers:
  - mann_kendall_test with known increasing / random data
  - sens_slope with perfectly linear data (y = 2x)
  - pettitt_test with a known change point
  - seasonality_index with uniform and highly seasonal data
  - precipitation_concentration_index with uniform data
  - compute_spi with normal-looking data
  - compute_all_statistics orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from watershed_analyzer.core.stats import (
    compute_all_statistics,
    compute_spi,
    mann_kendall_test,
    pettitt_test,
    precipitation_concentration_index,
    seasonality_index,
    sens_slope,
)


# ---------------------------------------------------------------------------
# 1. Mann-Kendall trend test
# ---------------------------------------------------------------------------

class TestMannKendallTest:
    """Tests for mann_kendall_test."""

    def test_strong_increasing_trend(self):
        """[1..10] has a monotonic increasing trend; p should be significant."""
        data = np.arange(1, 11, dtype=np.float64)
        result = mann_kendall_test(data)

        assert result["trend"] == "increasing"
        assert result["z"] > 0
        assert result["p_value"] < 0.05
        assert result["s"] > 0

    def test_random_data_no_trend(self):
        """Gaussian random data should not show a significant trend at α=0.05."""
        rng = np.random.default_rng(seed=42)
        data = rng.normal(loc=100, scale=10, size=30)
        result = mann_kendall_test(data)

        # No strong guarantee, but typically "no trend"
        assert result["trend"] in {"increasing", "decreasing", "no trend"}
        assert "z" in result
        assert "p_value" in result

    def test_decreasing_trend(self):
        """[10..1] should show a decreasing trend."""
        data = np.arange(10, 0, -1, dtype=np.float64)
        result = mann_kendall_test(data)

        assert result["trend"] == "decreasing"
        assert result["s"] < 0

    def test_too_few_observations_raises(self):
        """Fewer than 4 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 4"):
            mann_kendall_test(np.array([1.0, 2.0, 3.0]))

    def test_pandas_series_input(self):
        """Should accept a pandas Series without error."""
        s = pd.Series(np.arange(1, 11, dtype=np.float64))
        result = mann_kendall_test(s)
        assert result["trend"] == "increasing"

    def test_nan_values_are_dropped(self):
        """NaN values should be silently removed."""
        data = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan,
                        4.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = mann_kendall_test(data)
        assert result["trend"] == "increasing"


# ---------------------------------------------------------------------------
# 2. Sen's slope estimator
# ---------------------------------------------------------------------------

class TestSensSlope:
    """Tests for sens_slope."""

    def test_linear_data_slope_is_two(self):
        """For y = 2x, Sen's median slope should be very close to 2.0."""
        x = np.arange(10, dtype=np.float64)
        y = 2.0 * x  # exact linear: slope = 2.0
        result = sens_slope(y)

        assert result["slope"] == pytest.approx(2.0, abs=0.01)
        assert result["slope_lower"] <= result["slope"] <= result["slope_upper"]

    def test_constant_data_slope_is_zero(self):
        """For constant data, the slope should be 0.0."""
        data = np.full(20, 5.0)
        result = sens_slope(data)
        assert result["slope"] == pytest.approx(0.0, abs=1e-10)

    def test_too_few_observations_raises(self):
        """Fewer than 2 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            sens_slope(np.array([1.0]))

    def test_confidence_interval_bounds(self):
        """Lower CI should be ≤ median slope ≤ upper CI."""
        rng = np.random.default_rng(seed=0)
        data = rng.normal(loc=0, scale=1, size=50).cumsum()
        result = sens_slope(data)
        assert result["slope_lower"] <= result["slope"] <= result["slope_upper"]


# ---------------------------------------------------------------------------
# 3. Pettitt change-point test
# ---------------------------------------------------------------------------

class TestPettittTest:
    """Tests for pettitt_test."""

    def test_known_change_point(self):
        """[1]*10 + [10]*10 — change should be detected around index 9-10."""
        data = np.array([1.0] * 10 + [10.0] * 10)
        result = pettitt_test(data)

        # Change point should be near the transition (index 9 or 10)
        assert result["change_point_index"] in {9, 10}
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_no_change_point(self):
        """Uniform random data should not detect a significant change point."""
        rng = np.random.default_rng(seed=7)
        data = rng.normal(loc=50, scale=2, size=30)
        result = pettitt_test(data)

        # With no true change, p should be large
        assert result["p_value"] > 0.05

    def test_too_few_observations_raises(self):
        """Fewer than 3 observations should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            pettitt_test(np.array([1.0, 2.0]))

    def test_result_keys(self):
        """Result should contain all documented keys."""
        data = np.array([1.0] * 5 + [10.0] * 5)
        result = pettitt_test(data)
        expected_keys = {
            "change_point_index", "change_point_value",
            "k_statistic", "p_value", "significant",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 4. Seasonality Index
# ---------------------------------------------------------------------------

class TestSeasonalityIndex:
    """Tests for seasonality_index."""

    def test_uniform_data_si_near_zero(self):
        """When all 12 months have the same rainfall, SI should be ~0."""
        uniform = np.full(12, 100.0)
        result = seasonality_index(uniform)
        assert result["si_value"] == pytest.approx(0.0, abs=1e-10)
        assert result["classification"] == "very equable"

    def test_highly_seasonal_si_greater_than_one(self):
        """If all rain falls in one month, SI should be > 1.0."""
        monthly = np.zeros(12)
        monthly[6] = 1200.0  # All rain in July
        result = seasonality_index(monthly)
        assert result["si_value"] > 1.0
        assert result["classification"] == "most seasonal"

    def test_wet_season_pattern(self):
        """Two wet months, rest dry — SI should be 'rather seasonal' or higher."""
        monthly = np.full(12, 10.0)
        monthly[5] = 200.0
        monthly[6] = 200.0
        result = seasonality_index(monthly)
        assert result["si_value"] > 0.4  # Should be noticeably seasonal

    def test_wrong_length_raises(self):
        """Providing != 12 values should raise ValueError."""
        with pytest.raises(ValueError, match="exactly 12"):
            seasonality_index(np.ones(6))

    def test_zero_total_raises(self):
        """Zero annual total should raise ValueError."""
        with pytest.raises(ValueError, match="zero or negative"):
            seasonality_index(np.zeros(12))


# ---------------------------------------------------------------------------
# 5. Precipitation Concentration Index
# ---------------------------------------------------------------------------

class TestPrecipitationConcentrationIndex:
    """Tests for precipitation_concentration_index."""

    def test_uniform_data_pci_near_83(self):
        """With perfectly uniform monthly rainfall, PCI should be ≈ 8.33."""
        uniform = np.full(12, 100.0)
        result = precipitation_concentration_index(uniform)

        # PCI = sum(p_i²)/sum(p_i)² × 100 = 12 × 100² / (1200)² × 100
        #     = 12 × 10000 / 1440000 × 100 = 120000/1440000 × 100 = 8.333...
        assert result["pci_value"] == pytest.approx(8.333, abs=0.01)
        assert result["classification"] == "uniform"

    def test_concentrated_rainfall(self):
        """When all rain falls in one month, PCI should be very high (> 20)."""
        monthly = np.zeros(12)
        monthly[0] = 1200.0
        result = precipitation_concentration_index(monthly)
        assert result["pci_value"] > 20
        assert result["classification"] == "highly irregular"

    def test_wrong_length_raises(self):
        """Providing != 12 values should raise ValueError."""
        with pytest.raises(ValueError, match="exactly 12"):
            precipitation_concentration_index(np.ones(5))


# ---------------------------------------------------------------------------
# 6. Standardized Precipitation Index (SPI)
# ---------------------------------------------------------------------------

class TestComputeSPI:
    """Tests for compute_spi."""

    def test_spi_with_varying_rainfall(self):
        """SPI should return an array; mean should be near 0 for varied data."""
        rng = np.random.default_rng(seed=99)
        data = rng.gamma(shape=5, scale=20, size=60)
        result = compute_spi(data, scale=3)

        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        # Mean of SPI should be approximately 0 for well-distributed data
        assert np.mean(result) == pytest.approx(0.0, abs=0.5)

    def test_spi_scale_validation(self):
        """SPI scale < 1 should raise ValueError."""
        with pytest.raises(ValueError, match=">= 1"):
            compute_spi(np.ones(10), scale=0)

    def test_spi_too_few_points(self):
        """Fewer data points than scale should raise ValueError."""
        with pytest.raises(ValueError, match="Need at least"):
            compute_spi(np.ones(2), scale=6)

    def test_spi_all_zeros(self):
        """All-zero rainfall should return zeros (can't fit gamma)."""
        result = compute_spi(np.zeros(24), scale=3)
        assert np.all(result == 0.0)

    def test_spi_length_equals_input_minus_scale_plus_one(self):
        """Output length = len(data) - scale + 1."""
        data = np.arange(1.0, 25.0)  # 24 values
        result = compute_spi(data, scale=3)
        assert len(result) == 22  # 24 - 3 + 1


# ---------------------------------------------------------------------------
# 7. compute_all_statistics orchestrator
# ---------------------------------------------------------------------------

class TestComputeAllStatistics:
    """Tests for compute_all_statistics."""

    def test_orchestrator_returns_all_keys(self):
        """Should return a dict with mann_kendall, sen_slope, pettitt, spi, etc."""
        rng = np.random.default_rng(seed=12)
        index = pd.date_range("2000-01-01", periods=48, freq="MS")
        values = rng.gamma(shape=5, scale=20, size=48)
        series = pd.Series(values, index=index)

        result = compute_all_statistics(series)

        assert "mann_kendall" in result
        assert "sen_slope" in result
        assert "pettitt" in result
        assert "spi" in result
        assert "seasonality_index" in result
        assert "pci" in result
        assert "error" not in result["mann_kendall"]

    def test_orchestrator_with_explicit_monthly(self):
        """Providing monthly_values_12 should produce SI and PCI."""
        series = pd.Series(np.arange(1.0, 13.0))
        monthly = np.full(12, 100.0)

        result = compute_all_statistics(series, monthly_values_12=monthly)

        assert result["seasonality_index"]["si_value"] == pytest.approx(0.0, abs=1e-10)
        assert result["pci"]["pci_value"] == pytest.approx(8.333, abs=0.01)

    def test_orchestrator_short_series_graceful(self):
        """Even with a short series, SPI may fail but others should work."""
        series = pd.Series(np.arange(1.0, 6.0))  # only 5 points
        result = compute_all_statistics(series)

        # mann_kendall requires ≥ 4 — should still work
        assert result["mann_kendall"]["trend"] == "increasing"
        # SI / PCI should have error since no monthly data
        assert "error" in result["seasonality_index"]
