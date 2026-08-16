"""Lightweight NetCDF-to-GeoTIFF extraction utilities.

This module provides two public helpers for converting a time-indexed
:class:`xarray.DataArray` into rasterio objects that downstream watershed-
analysis code can consume directly:

* ``extract_monthly_from_netcdf`` — iterates over the time dimension and
  returns a list of ``(time_label, rasterio.MemoryFile)`` tuples.
* ``netcdf_to_geotiff_bytes`` — same workflow but returns raw GeoTIFF
  ``bytes`` instead of open :class:`MemoryFile` handles.

Both functions rely on ``rioxarray`` for CRS handling and spatial metadata
derivation.  Python 3.10+ type hints and Google-style docstrings are used
throughout.
"""

from __future__ import annotations

import logging
import numpy as np
import rasterio
from affine import Affine
from rasterio import MemoryFile
from rasterio.crs import CRS as RasterCRS
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_time_label(time_val: Any) -> str:
    """Return a human-readable string label for a time coordinate value.

    If the time value encodes a day-level resolution (i.e. the day differs
    from 1) the label is formatted as ``'YYYY-MM-DD'``; otherwise it is
    truncated to ``'YYYY-MM'``.  This heuristic covers most NetCDF climate
    datasets that store monthly means with an arbitrary day-of-month anchor.

    Parameters
    ----------
    time_val:
        A scalar value compatible with :class:`numpy.datetime64` or
        :class:`pandas.Timestamp`.

    Returns
    -------
    str
        The formatted time label.
    """
    ts = np.datetime64(time_val, "ns")
    # Convert to datetime-like for attribute access.
    dt = ts.astype("datetime64[D]").astype("object")

    year = dt.year
    month = dt.month
    day = dt.day

    if day != 1:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return f"{year:04d}-{month:02d}"


def _build_geotransform(da_slice: Any) -> Affine:
    """Derive an affine geotransform from a 2-D DataArray slice.

    The transform is computed from the 1-D coordinate arrays so that it
    maps pixel indices to the centre of the corresponding grid cell.
    Coordinates are assumed to be regularly spaced in both dimensions.

    Parameters
    ----------
    da_slice:
        A 2-D :class:`xarray.DataArray` (or duck-typed object) with
        ``x``/``lon`` and ``y``/``lat`` coordinate arrays.

    Returns
    -------
    affine.Affine
        An ``Affine`` geotransform suitable for :func:`rasterio.open`.
    """
    # Discover coordinate names.
    coord_names = list(da_slice.coords)
    x_dim: str | None = None
    y_dim: str | None = None

    x_candidates = {"x", "lon", "longitude"}
    y_candidates = {"y", "lat", "latitude"}

    for name in coord_names:
        if name.lower() in x_candidates:
            x_dim = name
        elif name.lower() in y_candidates:
            y_dim = name

    if x_dim is None or y_dim is None:
        raise ValueError(
            f"Cannot identify spatial dimensions from coords {coord_names}. "
            f"Expected one of {x_candidates} and one of {y_candidates}."
        )

    x_coords = da_slice[x_dim].values
    y_coords = da_slice[y_dim].values

    # Handle descending y-axis (common in NetCDF).
    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        flip_y = True
    else:
        flip_y = False

    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])

    # Upper-left corner (pixel centre minus half-pixel offset).
    x_origin = float(x_coords[0]) - dx / 2.0
    y_origin = float(y_coords[0]) - dy / 2.0

    # rasterio expects dy > 0 for north-up orientation.
    if flip_y:
        dy = -dy
        y_origin = float(y_coords[-1]) - abs(dy) / 2.0  # type: ignore[index]

    return Affine(dx, 0.0, x_origin, 0.0, dy, y_origin)


def _get_crs(da: Any) -> RasterCRS | None:
    """Attempt to retrieve the CRS from a DataArray via rioxarray.

    Parameters
    ----------
    da:
        An :class:`xarray.DataArray` (or duck-typed object with a ``.rio``
        accessor).

    Returns
    -------
    rasterio.crs.CRS | None
        The CRS, or ``None`` if it cannot be determined.
    """
    try:
        import rioxarray  # noqa: F401 — ensure extension is registered

        crs = da.rio.crs
        if crs is not None:
            return RasterCRS.from_user_input(crs)
    except Exception as exc:
        logger.debug("Could not determine CRS via rioxarray: %s", exc)
    return None


def _ensure_2d(da_slice: Any) -> Any:
    """Squeeze a DataArray slice to exactly 2-D and return it.

    Parameters
    ----------
    da_slice:
        A slice of an :class:`xarray.DataArray` that should be reduced to
        two spatial dimensions.

    Returns
    -------
    xarray.DataArray
        The squeezed 2-D array.

    Raises
    ------
    ValueError
        If the array has more than two dimensions after squeezing.
    """
    squeezed = da_slice.squeeze()
    if squeezed.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array after squeezing, got {squeezed.ndim} "
            f"dimensions with shape {squeezed.shape}."
        )
    return squeezed


# ---------------------------------------------------------------------------
# 1. extract_monthly_from_netcdf
# ---------------------------------------------------------------------------


def extract_monthly_from_netcdf(
    da: Any,
    time_dim: str = "time",
) -> list[tuple[str, MemoryFile]]:
    """Extract each time step from a DataArray as a rasterio MemoryFile.

    Iterates over the coordinate values of *time_dim*, selects the
    corresponding 2-D spatial slice, computes an affine geotransform from
    the coordinate arrays, and writes the result to an in-memory GeoTIFF
    via :class:`rasterio.MemoryFile`.

    Parameters
    ----------
    da:
        An :class:`xarray.DataArray` with at least one spatial and one time
        dimension.  Must have a ``.rio`` accessor (i.e. ``rioxarray`` must
        be importable).
    time_dim:
        Name of the time dimension to iterate over.  Defaults to
        ``'time'``.

    Returns
    -------
    list[tuple[str, rasterio.MemoryFile]]
        A list where each element is ``(time_label, memfile)``.  *memfile*
        is an **open** :class:`rasterio.MemoryFile` that the caller is
        responsible for closing when no longer needed.

    Raises
    ------
    ValueError
        If *time_dim* is not present in the DataArray dimensions.
    RuntimeError
        If writing a particular time step to a MemoryFile fails.
    """
    import xarray as xr  # noqa: F401 — type hinting support

    if time_dim not in da.dims:
        raise ValueError(
            f"Dimension '{time_dim}' not found in DataArray dims {list(da.dims)}."
        )

    crs = _get_crs(da)
    time_coords = da[time_dim].values

    logger.info(
        "Extracting %d time steps from DataArray (shape=%s, CRS=%s)",
        len(time_coords),
        da.shape,
        crs,
    )

    results: list[tuple[str, MemoryFile]] = []

    for idx, t_val in enumerate(time_coords):
        time_label = _format_time_label(t_val)

        try:
            # Select and squeeze to 2-D.
            slice_2d = _ensure_2d(da.isel({time_dim: idx}))

            # Flip data rows if y-axis is descending (top-to-bottom).
            y_coord_name = _identify_y_dim(slice_2d)
            if y_coord_name is not None:
                y_vals = slice_2d[y_coord_name].values
                if len(y_vals) > 1 and y_vals[0] > y_vals[-1]:
                    slice_2d = slice_2d.isel({y_coord_name: slice(None, None, -1)})

            transform = _build_geotransform(slice_2d)
            data = np.asarray(slice_2d.values, dtype=np.float64)

            height, width = data.shape

            memfile = MemoryFile()
            with memfile.open(
                driver="GTiff",
                width=width,
                height=height,
                count=1,
                dtype=data.dtype,
                crs=crs,
                transform=transform,
                nodata=np.nan,
            ) as dst:
                dst.write(data, 1)

            results.append((time_label, memfile))
            logger.debug(
                "Wrote time step %d (%s): %dx%d, transform=%s",
                idx,
                time_label,
                width,
                height,
                transform,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to export time step {idx} ({time_label}): {exc}"
            ) from exc

    logger.info(
        "extract_monthly_from_netcdf: exported %d time steps.", len(results)
    )
    return results


# ---------------------------------------------------------------------------
# 2. netcdf_to_geotiff_bytes
# ---------------------------------------------------------------------------


def netcdf_to_geotiff_bytes(
    da: Any,
    time_dim: str = "time",
) -> list[tuple[str, bytes]]:
    """Convert each time step of a DataArray to raw GeoTIFF bytes.

    This is a convenience wrapper around :func:`extract_monthly_from_netcdf`
    that reads the bytes out of each :class:`MemoryFile` and closes the
    handle immediately, returning only ``(time_label, bytes)`` tuples.

    Parameters
    ----------
    da:
        An :class:`xarray.DataArray` with at least one spatial and one time
        dimension.
    time_dim:
        Name of the time dimension to iterate over.  Defaults to
        ``'time'``.

    Returns
    -------
    list[tuple[str, bytes]]
        A list of ``(time_label, geotiff_bytes)`` pairs suitable for
        serialization, HTTP responses, or on-disk writing.

    Raises
    ------
    ValueError
        If *time_dim* is not present in the DataArray dimensions.
    RuntimeError
        If writing a particular time step to a MemoryFile fails.
    """
    memfile_pairs = extract_monthly_from_netcdf(da, time_dim=time_dim)

    byte_results: list[tuple[str, bytes]] = []
    for time_label, memfile in memfile_pairs:
        try:
            raw_bytes = memfile.read()
        finally:
            memfile.close()
        byte_results.append((time_label, raw_bytes))

    logger.info(
        "netcdf_to_geotiff_bytes: exported %d GeoTIFF byte blobs.",
        len(byte_results),
    )
    return byte_results


# ---------------------------------------------------------------------------
# Shared internal helpers (used above)
# ---------------------------------------------------------------------------


def _identify_y_dim(da_slice: Any) -> str | None:
    """Return the name of the y/lat dimension from a 2-D DataArray slice.

    Parameters
    ----------
    da_slice:
        A 2-D :class:`xarray.DataArray`.

    Returns
    -------
    str | None
        The y-dimension name, or ``None`` if none matches.
    """
    y_candidates = {"y", "lat", "latitude"}
    for dim in da_slice.dims:
        if dim.lower() in y_candidates:
            return dim
    return None
