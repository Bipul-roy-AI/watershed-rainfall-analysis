"""Areal Reduction Factor (ARF) computation module.

Provides multiple ARF methods for converting point-rainfall estimates
to basin-averaged rainfall depths, accounting for the fact that a single
rainfall grid cell overestimates the true basin-average rainfall as
basin area increases.

References:
    Srikanthan, R. & McMahon, T.A. (2007). Stochastic generation of
    annual rainfall data. *Journal of Hydrology*, 228, 56–69.

    Reed, S. (1999). Flood estimation for ungauged catchments.
    USGS Water-Supply Paper 2375.
"""

from __future__ import annotations

import logging
import numpy as np

from watershed_analyzer.config import ARF_METHODS, DEFAULT_ARF_METHOD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USGS Reed (1999) lookup table
# ---------------------------------------------------------------------------
# Keys are durations in hours.  Values are (area_brackets, arf_values)
# where area_brackets are in km² and ARF is dimensionless.
_USGS_AREA_BRACKETS: list[float] = [10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

_USGS_ARF_TABLE: dict[float, list[float]] = {
    1.0:  [0.90, 0.87, 0.83, 0.80, 0.73, 0.68],
    6.0:  [0.95, 0.93, 0.90, 0.88, 0.83, 0.79],
    24.0: [0.98, 0.97, 0.95, 0.93, 0.90, 0.87],
}

_DEFAULT_DURATION: float = 24.0  # fall back duration when unspecified


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_arf(
    area_km2: float,
    method: str = DEFAULT_ARF_METHOD,
    duration_hr: float | None = None,
) -> float:
    """Compute the Areal Reduction Factor for a given basin area.

    The ARF corrects the spatial mean of gridded rainfall to account for
    the reduction in basin-average rainfall with increasing catchment
    area.  Values are clamped to [0.5, 1.0].

    Args:
        area_km2: Basin area in square kilometres.  Must be non-negative.
        method: ARF computation method.  One of:
            - ``"srikanthan_mcmahon"`` — Srikanthan & McMahon (2007)
              empirical formula (duration-independent).
            - ``"usgs_reed"`` — USGS Reed (1999) area-duration lookup
              with linear interpolation.  *duration_hr* is required;
              defaults to 24 h when not supplied.
            - ``"none"`` — no reduction (returns 1.0).
        duration_hr: Storm / accumulation duration in hours.
            Only used when *method* is ``"usgs_reed"``.

    Returns:
        Areal Reduction Factor in the range [0.5, 1.0].

    Raises:
        ValueError: If *method* is not recognised, *area_km2* is negative,
            or *method* is ``"usgs_reed"`` with no *duration_hr* and no
            default is available.

    References:
        Srikanthan, R. & McMahon, T.A. (2007). Stochastic generation
        of annual rainfall data. *Journal of Hydrology*, 228, 56–69.

        Reed, S. (1999). Flood estimation for ungauged catchments.
        USGS Water-Supply Paper 2375.
    """
    if area_km2 < 0:
        raise ValueError(f"area_km2 must be non-negative, got {area_km2}")

    if area_km2 == 0.0:
        logger.debug("Basin area is 0 km²; returning ARF = 1.0")
        return 1.0

    method = method.lower().strip()

    if method == "none":
        logger.debug("ARF method='none' — no areal reduction")
        return 1.0

    if method == "srikanthan_mcmahon":
        return _arf_srikanthan_mcmahon(area_km2)

    if method == "usgs_reed":
        dur = duration_hr if duration_hr is not None else _DEFAULT_DURATION
        return _arf_usgs_reed(area_km2, dur)

    valid = list(ARF_METHODS.keys())
    raise ValueError(
        f"Unknown ARF method '{method}'.  Valid methods: {valid}"
    )


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------

def _arf_srikanthan_mcmahon(area_km2: float) -> float:
    """Srikanthan & McMahon (2007) ARF formula.

    ARF = 1 − 0.04 × A^0.4   for A < 50 km²
    ARF = 1 − 0.04 × 50^0.4 × (A/50)^0.15   for A ≥ 50 km²

    Clamped to [0.5, 1.0].

    References:
        Srikanthan, R. & McMahon, T.A. (2007). *Journal of Hydrology*, 228.
    """
    threshold = 50.0
    c0 = 0.04 * (threshold ** 0.4)

    if area_km2 < threshold:
        arf = 1.0 - 0.04 * (area_km2 ** 0.4)
    else:
        arf = 1.0 - c0 * ((area_km2 / threshold) ** 0.15)

    arf = float(np.clip(arf, 0.5, 1.0))
    logger.debug(
        "Srikanthan-McMahon ARF: area=%.2f km² → ARF=%.4f",
        area_km2,
        arf,
    )
    return arf


def _arf_usgs_reed(area_km2: float, duration_hr: float) -> float:
    """USGS Reed (1999) lookup-table ARF with linear interpolation.

    For basins smaller than 10 km² the ARF is assumed to be 1.0
    (point-rainfall ≈ areal-rainfall).  For durations not in the
    lookup table, linear interpolation between the nearest bounding
    durations is performed.

    Args:
        area_km2: Basin area in km².
        duration_hr: Storm duration in hours.

    Returns:
        Areal Reduction Factor in [0.5, 1.0].

    References:
        Reed, S. (1999). *USGS Water-Supply Paper 2375*.
    """
    # For very small basins, no reduction
    if area_km2 < _USGS_AREA_BRACKETS[0]:
        logger.debug(
            "USGS Reed: area=%.2f km² < 10 km² → ARF=1.0",
            area_km2,
        )
        return 1.0

    durations = sorted(_USGS_ARF_TABLE.keys())
    arf_values: list[float]

    if duration_hr in durations:
        arf_values = _USGS_ARF_TABLE[duration_hr]
        arf = float(np.interp(area_km2, _USGS_AREA_BRACKETS, arf_values))
    else:
        # Interpolate between the two bounding durations
        lower_dur = max(d for d in durations if d <= duration_hr)
        upper_dur = min(d for d in durations if d >= duration_hr)
        if lower_dur == upper_dur:
            arf_values = _USGS_ARF_TABLE[lower_dur]
            arf = float(np.interp(area_km2, _USGS_AREA_BRACKETS, arf_values))
        else:
            weight = (duration_hr - lower_dur) / (upper_dur - lower_dur)
            arf_lo = float(
                np.interp(area_km2, _USGS_AREA_BRACKETS, _USGS_ARF_TABLE[lower_dur])
            )
            arf_hi = float(
                np.interp(area_km2, _USGS_AREA_BRACKETS, _USGS_ARF_TABLE[upper_dur])
            )
            arf = arf_lo + weight * (arf_hi - arf_lo)

    arf = float(np.clip(arf, 0.5, 1.0))
    logger.debug(
        "USGS Reed ARF: area=%.2f km², duration=%.1f hr → ARF=%.4f",
        area_km2,
        duration_hr,
        arf,
    )
    return arf
