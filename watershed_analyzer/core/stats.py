"""Statistical tests and indices for watershed rainfall analysis.

Implements the following from-scratch (no pymannkendall dependency):

- Mann-Kendall trend test
- Sen's slope estimator
- Pettitt change-point test
- Standardized Precipitation Index (SPI)
- Walsh & Lawler Seasonality Index (SI)
- Oliver Precipitation Concentration Index (PCI)

All functions use Python 3.10+ type hints, Google-style docstrings,
and structured logging via the ``logging`` stdlib module.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from watershed_analyzer.config import DEFAULT_SPI_SCALE, SPI_SCALES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: np.ndarray | pd.Series) -> np.ndarray:
    """Convert *data* to a 1-D :class:`numpy.ndarray` of float64.

    Parameters
    ----------
    data:
        Input array or pandas Series.

    Returns
    -------
    numpy.ndarray
        Cleaned, flat, float64 array with NaN values removed.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    return arr


def _significance_stars(p_value: float) -> str:
    """Return significance annotation string.

    Parameters
    ----------
    p_value:
        Two-tailed p-value from a statistical test.

    Returns
    -------
    str
        ``'***'`` for *p* < 0.001, ``'**'`` for *p* < 0.01,
        ``'*'`` for *p* < 0.05, ``'ns'`` otherwise.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# 1. Mann-Kendall trend test
# ---------------------------------------------------------------------------


def mann_kendall_test(data: np.ndarray | pd.Series) -> dict[str, Any]:
    """Perform the Mann-Kendall trend test from scratch.

    Computes Kendall's *S* statistic, its variance (with tie correction),
    the standardised *Z* statistic, and a two-tailed *p*-value using the
    standard normal CDF.

    Parameters
    ----------
    data:
        1-D time-series of numeric values.  NaN values are silently
        dropped.

    Returns
    -------
    dict[str, Any]
        ``'trend'``: ``'increasing'``, ``'decreasing'``, or ``'no trend'``
        ``'z'``: Standardised test statistic (float)
        ``'p_value'``: Two-tailed *p*-value (float)
        ``'s'``: Kendall's *S* (int)
        ``'var_s'``: Variance of *S* (float)
        ``'significance'``: ``'***'``, ``'**'``, ``'*'``, or ``'ns'``

    Raises
    ------
    ValueError
        If fewer than 4 valid observations are provided.
    """
    x = _to_numpy(data)
    n = len(x)

    if n < 4:
        raise ValueError(
            f"Mann-Kendall test requires at least 4 observations; got {n}."
        )

    # --- Kendall's S ---
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # --- Variance of S (with tie correction) ---
    # Count tied groups
    unique, counts = np.unique(x, return_counts=True)
    tied_counts = counts[counts > 1]
    tie_correction = np.sum(tied_counts * (tied_counts - 1) * (2 * tied_counts + 5))

    var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0

    # --- Z statistic (continuity correction) ---
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # --- Two-tailed p-value ---
    p_value = 2.0 * norm.sf(abs(z))

    # --- Trend direction ---
    if z > 0 and p_value < 0.05:
        trend = "increasing"
    elif z < 0 and p_value < 0.05:
        trend = "decreasing"
    else:
        trend = "no trend"

    significance = _significance_stars(p_value)

    result = {
        "trend": trend,
        "z": float(z),
        "p_value": float(p_value),
        "s": int(s),
        "var_s": float(var_s),
        "significance": significance,
    }

    logger.info(
        "Mann-Kendall: trend=%s  Z=%.4f  p=%.6f  S=%d  sig=%s",
        trend, z, p_value, s, significance,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Sen's slope estimator
# ---------------------------------------------------------------------------


def sens_slope(data: np.ndarray | pd.Series) -> dict[str, Any]:
    """Compute Sen's slope (Theil-Sen) estimator with 95 % confidence interval.

    The median of all pairwise slopes ``(y_j - y_i) / (j - i)`` for
    ``j > i`` gives the robust trend magnitude.  The confidence interval
    is derived from the 2.5-th and 97.5-th percentiles of the slope
    distribution.

    Parameters
    ----------
    data:
        1-D time-series of numeric values.

    Returns
    -------
    dict[str, Any]
        ``'slope'``: Median slope (float)
        ``'slope_lower'``: Lower 95 % CI bound (float)
        ``'slope_upper'``: Upper 95 % CI bound (float)

    Raises
    ------
    ValueError
        If fewer than 2 valid observations remain.
    """
    x = _to_numpy(data)
    n = len(x)

    if n < 2:
        raise ValueError(
            f"Sen's slope requires at least 2 observations; got {n}."
        )

    slopes: list[float] = []
    for j in range(1, n):
        for i in range(j):
            denom = j - i
            if denom != 0:
                slopes.append((x[j] - x[i]) / denom)

    if not slopes:
        raise ValueError("No valid pairwise slopes could be computed.")

    slopes_arr = np.array(slopes, dtype=np.float64)
    median_slope = float(np.median(slopes_arr))
    lower = float(np.percentile(slopes_arr, 2.5))
    upper = float(np.percentile(slopes_arr, 97.5))

    result = {
        "slope": median_slope,
        "slope_lower": lower,
        "slope_upper": upper,
    }

    logger.info(
        "Sen's slope: median=%.6f  95%% CI=[%.6f, %.6f]",
        median_slope, lower, upper,
    )
    return result


# ---------------------------------------------------------------------------
# 3. Pettitt change-point test
# ---------------------------------------------------------------------------


def pettitt_test(data: np.ndarray | pd.Series) -> dict[str, Any]:
    """Pettitt change-point test for detecting a single shift in the mean.

    Computes ``U_{t,T} = sum_{j=1}^{t} sign(x_t - x_j)`` for each
    ``t = 1, ..., T-1``.  The test statistic is ``K_T = max |U_{t,T}|``
    and the change point is ``argmax |U_{t,T}|``.

    The approximate *p*-value is ``p ≈ 2·exp(−6·K_T² / (T³ + T²))``.

    Parameters
    ----------
    data:
        1-D time-series of numeric values.

    Returns
    -------
    dict[str, Any]
        ``'change_point_index'``: 0-based index of the most likely
            change point (int)
        ``'change_point_value'``: Data value at the change point (float)
        ``'k_statistic'``: ``K_T`` (float)
        ``'p_value'``: Approximate *p*-value (float, capped at 1.0)
        ``'significant'``: Whether *p* < 0.05 (bool)

    Raises
    ------
    ValueError
        If fewer than 3 valid observations are provided.
    """
    x = _to_numpy(data)
    T = len(x)

    if T < 3:
        raise ValueError(
            f"Pettitt test requires at least 3 observations; got {T}."
        )

    # Compute U_{t,T} for t = 0 .. T-2
    # Standard Pettitt: U_{t,T} = Σ_{i=1}^{t+1} Σ_{j=t+2}^{T} sign(x_i - x_j)
    # This is the two-sample U statistic splitting the series at t.
    u_stat = np.zeros(T - 1, dtype=np.float64)
    for t in range(T - 1):
        for i in range(t + 1):
            for j in range(t + 1, T):
                u_stat[t] += np.sign(x[i] - x[j])

    abs_u = np.abs(u_stat)
    K_T = float(np.max(abs_u))
    cp_index = int(np.argmax(abs_u))

    # Approximate p-value
    denominator = T**3 + T**2
    if denominator > 0 and K_T > 0:
        p_value = 2.0 * np.exp(-6.0 * K_T**2 / denominator)
        p_value = min(p_value, 1.0)
    else:
        p_value = 1.0

    result = {
        "change_point_index": cp_index,
        "change_point_value": float(x[cp_index]),
        "k_statistic": K_T,
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }

    logger.info(
        "Pettitt test: change_point=%d (value=%.4f)  K=%.2f  p=%.6f",
        cp_index, x[cp_index], K_T, p_value,
    )
    return result


# ---------------------------------------------------------------------------
# 4. Standardized Precipitation Index (SPI)
# ---------------------------------------------------------------------------


def compute_spi(rainfall_mm: np.ndarray | pd.Series, scale: int = 3) -> np.ndarray:
    """Compute the Standardized Precipitation Index (SPI).

    A two-parameter gamma distribution (``floc=0``) is fitted to the
    non-zero rolling-sum values.  Zero values are handled with the mixed-
    distribution approach: cumulative probability for zero-rainfall periods
    is ``q = (m − 0.5) / n`` and the total CDF is
    ``P_total = q + (1 − q) · F(x)`` where *F* is the gamma CDF.
    The resulting probabilities are then transformed to standard-normal
    quantiles.

    Parameters
    ----------
    rainfall_mm:
        Monthly rainfall time-series (mm).  NaN values are dropped.
    scale:
        Accumulation window in months (e.g. ``3`` for SPI-3).
        Must be a positive integer.

    Returns
    -------
    numpy.ndarray
        SPI values with the same length as the effective input minus
        ``scale − 1`` (due to the rolling window).

    Raises
    ------
    ValueError
        If *scale* < 1, or fewer than ``scale`` data points remain,
        or all values are zero.
    """
    x = _to_numpy(rainfall_mm)

    if scale < 1:
        raise ValueError(f"SPI scale must be >= 1; got {scale}.")

    if len(x) < scale:
        raise ValueError(
            f"Need at least {scale} data points for SPI-{scale}; "
            f"got {len(x)}."
        )

    # Rolling sum
    rolled = np.convolve(x, np.ones(scale, dtype=np.float64), mode="valid")

    n = len(rolled)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Fraction of zeros
    n_zeros = int(np.sum(rolled == 0))
    q = (n_zeros - 0.5) / n if n_zeros > 0 else 0.0
    q = max(0.0, min(q, 1.0))

    # Fit gamma on non-zero values only
    positive = rolled[rolled > 0]
    if len(positive) < 2:
        logger.warning(
            "Too few positive values (%d) to fit gamma; returning zeros.",
            len(positive),
        )
        return np.zeros(n, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha, loc, beta = gamma.fit(positive, floc=0)

    if alpha <= 0 or beta <= 0:
        logger.warning(
            "Gamma fit produced invalid parameters (a=%.4f, b=%.4f); "
            "returning zeros.",
            alpha, beta,
        )
        return np.zeros(n, dtype=np.float64)

    # Cumulative probabilities
    F_gamma = gamma.cdf(rolled, a=alpha, loc=loc, scale=beta)
    P_total = q + (1.0 - q) * F_gamma

    # Clamp to avoid exactly 0 or 1 (which would give ±inf from ppf)
    P_total = np.clip(P_total, 1e-10, 1.0 - 1e-10)

    # Transform to standard normal
    spi = norm.ppf(P_total)

    logger.info(
        "SPI-%d computed: n=%d, q=%.4f, gamma(a=%.4f, scale=%.4f)",
        scale, n, q, alpha, beta,
    )
    return spi


# ---------------------------------------------------------------------------
# 5. Seasonality Index (Walsh & Lawler 1981)
# ---------------------------------------------------------------------------


def seasonality_index(monthly_rainfall: np.ndarray | pd.Series) -> dict[str, Any]:
    r"""Compute the Walsh & Lawler (1981) Seasonality Index (SI).

    .. math::

        SI = \frac{1}{R} \sum_{i=1}^{12} |r_i - R/12|

    where *R* is the mean annual total and *r_i* is the mean monthly
    rainfall.

    Classification (Walsh & Lawler 1981):

    * ``< 0.19`` – Very equable
    * ``0.20–0.39`` – Equable but with a definite wet season
    * ``0.40–0.59`` – Rather seasonal with a short drier season
    * ``0.60–0.79`` – Seasonal
    * ``0.80–0.99`` – Markedly seasonal
    * ``>= 1.00`` – Most seasonal

    Parameters
    ----------
    monthly_rainfall:
        Exactly 12 monthly rainfall values (Jan–Dec).  If a pandas Series
        with a DatetimeIndex is passed, values are grouped by calendar
        month and averaged first.

    Returns
    -------
    dict[str, Any]
        ``'si_value'``: Numerical SI value (float)
        ``'classification'``: Descriptive category (str)

    Raises
    ------
    ValueError
        If the input does not contain exactly 12 values or the annual
        total is zero.
    """
    x = _to_numpy(monthly_rainfall)

    if len(x) != 12:
        raise ValueError(
            f"Seasonality Index requires exactly 12 monthly values; "
            f"got {len(x)}."
        )

    R = float(np.sum(x))
    if R <= 0:
        raise ValueError(
            "Cannot compute Seasonality Index: annual total is zero or negative."
        )

    si = float(np.sum(np.abs(x - R / 12.0)) / R)

    if si < 0.19:
        classification = "very equable"
    elif si < 0.40:
        classification = "equable but with a definite wet season"
    elif si < 0.60:
        classification = "rather seasonal with a short drier season"
    elif si < 0.80:
        classification = "seasonal"
    elif si < 1.00:
        classification = "markedly seasonal"
    else:
        classification = "most seasonal"

    result = {
        "si_value": si,
        "classification": classification,
    }

    logger.info("Seasonality Index: SI=%.4f  class='%s'", si, classification)
    return result


# ---------------------------------------------------------------------------
# 6. Precipitation Concentration Index (Oliver 1980)
# ---------------------------------------------------------------------------


def precipitation_concentration_index(
    monthly_rainfall: np.ndarray | pd.Series,
) -> dict[str, Any]:
    r"""Compute Oliver's (1980) Precipitation Concentration Index (PCI).

    .. math::

        PCI = \frac{\sum_{i=1}^{12} p_i^2}{\left(\sum_{i=1}^{12} p_i\right)^2} \times 100

    Classification (Oliver 1980):

    * ``< 10`` – Uniform distribution
    * ``11–15`` – Moderate concentration
    * ``16–20`` – Irregular distribution
    * ``> 20`` – Highly irregular distribution

    Parameters
    ----------
    monthly_rainfall:
        Exactly 12 monthly rainfall values (Jan–Dec).

    Returns
    -------
    dict[str, Any]
        ``'pci_value'``: Numerical PCI value (float)
        ``'classification'``: Descriptive category (str)

    Raises
    ------
    ValueError
        If the input does not contain exactly 12 values or the annual
        total is zero.
    """
    x = _to_numpy(monthly_rainfall)

    if len(x) != 12:
        raise ValueError(
            f"PCI requires exactly 12 monthly values; got {len(x)}."
        )

    total = float(np.sum(x))
    if total <= 0:
        raise ValueError(
            "Cannot compute PCI: total rainfall is zero or negative."
        )

    pci = float(np.sum(x**2) / (total**2) * 100.0)

    if pci < 10:
        classification = "uniform"
    elif pci <= 15:
        classification = "moderate concentration"
    elif pci <= 20:
        classification = "irregular"
    else:
        classification = "highly irregular"

    result = {
        "pci_value": pci,
        "classification": classification,
    }

    logger.info("PCI: %.2f  class='%s'", pci, classification)
    return result


# ---------------------------------------------------------------------------
# 7. Compute all statistics (orchestrator)
# ---------------------------------------------------------------------------


def compute_all_statistics(
    rainfall_series: pd.Series,
    monthly_values_12: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run every statistical test and index, returning a combined dict.

    The function executes the following analyses:

    1. **Mann-Kendall trend test** on the full *rainfall_series*.
    2. **Sen's slope estimator** on the full *rainfall_series*.
    3. **Pettitt change-point test** on the full *rainfall_series*.
    4. **SPI** for every scale in :pydata:`SPI_SCALES` (defaulting to
       :pydata:`DEFAULT_SPI_SCALE` only on error).
    5. **Seasonality Index** (Walsh & Lawler 1981).
    6. **Precipitation Concentration Index** (Oliver 1980).

    Parameters
    ----------
    rainfall_series:
        pandas Series with a numeric rainfall time-series.  If it has a
        DatetimeIndex, monthly values are derived automatically when
        *monthly_values_12* is ``None``.
    monthly_values_12:
        Optional pre-computed array of 12 mean-monthly rainfall values
        (Jan–Dec).  When ``None``, the function attempts to extract them
        from *rainfall_series* by grouping on the calendar month.

    Returns
    -------
    dict[str, Any]
        A nested dictionary containing the results of every test, keyed
        by descriptive names (``'mann_kendall'``, ``'sen_slope'``, etc.).
        Failed tests are recorded with an ``'error'`` key instead of
        raising.
    """
    results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Trend & change-point tests on the full series
    # ------------------------------------------------------------------
    # 1. Mann-Kendall
    try:
        results["mann_kendall"] = mann_kendall_test(rainfall_series)
    except Exception as exc:
        logger.warning("Mann-Kendall test failed: %s", exc)
        results["mann_kendall"] = {"error": str(exc)}

    # 2. Sen's slope
    try:
        results["sen_slope"] = sens_slope(rainfall_series)
    except Exception as exc:
        logger.warning("Sen's slope failed: %s", exc)
        results["sen_slope"] = {"error": str(exc)}

    # 3. Pettitt
    try:
        results["pettitt"] = pettitt_test(rainfall_series)
    except Exception as exc:
        logger.warning("Pettitt test failed: %s", exc)
        results["pettitt"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 4. SPI — compute for each configured scale
    # ------------------------------------------------------------------
    spi_results: dict[str, Any] = {}
    for scale in SPI_SCALES:
        key = f"spi_{scale}"
        try:
            spi_arr = compute_spi(rainfall_series, scale=scale)
            spi_results[key] = {
                "scale": scale,
                "mean": float(np.nanmean(spi_arr)) if len(spi_arr) > 0 else None,
                "min": float(np.nanmin(spi_arr)) if len(spi_arr) > 0 else None,
                "max": float(np.nanmax(spi_arr)) if len(spi_arr) > 0 else None,
                "values": spi_arr.tolist(),
            }
        except Exception as exc:
            logger.warning("SPI-%d failed: %s", scale, exc)
            spi_results[key] = {"scale": scale, "error": str(exc)}

    results["spi"] = spi_results

    # ------------------------------------------------------------------
    # 5 & 6. Seasonality Index & PCI (need 12 monthly means)
    # ------------------------------------------------------------------
    if monthly_values_12 is not None:
        monthly = _to_numpy(monthly_values_12)
    elif isinstance(rainfall_series.index, pd.DatetimeIndex):
        monthly = (
            rainfall_series.groupby(rainfall_series.index.month).mean().values
        )
        if len(monthly) != 12:
            logger.warning(
                "Calendar-month grouping yielded %d months instead of 12; "
                "skipping SI and PCI.",
                len(monthly),
            )
            monthly = None
    else:
        monthly = None
        logger.info(
            "No DatetimeIndex and no monthly_values_12 provided; "
            "skipping SI and PCI."
        )

    if monthly is not None:
        try:
            results["seasonality_index"] = seasonality_index(monthly)
        except Exception as exc:
            logger.warning("Seasonality Index failed: %s", exc)
            results["seasonality_index"] = {"error": str(exc)}

        try:
            results["pci"] = precipitation_concentration_index(monthly)
        except Exception as exc:
            logger.warning("PCI failed: %s", exc)
            results["pci"] = {"error": str(exc)}
    else:
        results["seasonality_index"] = {
            "error": "No 12-month data available."
        }
        results["pci"] = {"error": "No 12-month data available."}

    logger.info(
        "compute_all_statistics completed: %d test groups returned.",
        len(results),
    )
    return results
