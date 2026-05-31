"""Amplitude raster I/O: square-pixel loading, streaming mean stacks, and
export of per-date amplitude GeoTIFFs from an amplitude-only zarr store.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
import zarr
from scipy.ndimage import zoom as ndi_zoom


def _resample_to_square(amp_raw, transform_raw, crs):
    """Mask zeros/NaNs and resample an anisotropic amplitude tile to square
    pixels. Returns ``(amp, transform, crs, dx)`` with ``dx`` the square pixel
    size in metres. Shared by the GeoTIFF and zarr amplitude loaders so both
    produce identical geometry."""
    amp_raw = np.where(np.isnan(amp_raw) | (amp_raw == 0), np.nan, amp_raw)
    dx_raw, dy_raw = abs(transform_raw.a), abs(transform_raw.e)
    target_px = float(min(dx_raw, dy_raw))
    filled = np.where(np.isfinite(amp_raw), amp_raw, 0.0)
    valid = np.isfinite(amp_raw).astype(np.float32)
    amp_sq = ndi_zoom(filled, (dy_raw / target_px, dx_raw / target_px), order=1)
    valid_sq = ndi_zoom(valid, (dy_raw / target_px, dx_raw / target_px), order=0)
    amp = np.where(valid_sq > 0.5, amp_sq, np.nan)
    transform = Affine(target_px, transform_raw.b, transform_raw.c,
                       transform_raw.d, -target_px, transform_raw.f)
    return amp, transform, crs, float(target_px)


def _streaming_mean(loaders):
    """Per-pixel mean over an iterable yielding ``(amp, transform, crs, dx)``.
    Accumulates one tile at a time so RAM stays bounded."""
    acc = cnt = None
    transform = crs = dx_used = None
    for a, tr, cr, dx in loaders:
        if acc is None:
            acc = np.zeros(a.shape, dtype=np.float64)
            cnt = np.zeros(a.shape, dtype=np.int32)
            transform, crs, dx_used = tr, cr, dx
        m = np.isfinite(a)
        acc[m] += a[m]
        cnt[m] += 1
    if acc is None:
        raise ValueError("no amplitude tiles to stack")
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    return out, transform, crs, dx_used


def load_amp_square(path):
    """Load an amplitude GeoTIFF, resampled to square pixels.

    Returns ``(amp, transform, crs, dx)`` where ``dx`` is the (square) pixel
    size in metres and zeros / NaNs are masked out.
    """
    with rasterio.open(path) as ds:
        amp_raw = ds.read(1).astype(np.float32)
        transform_raw = ds.transform
        crs = ds.crs
    return _resample_to_square(amp_raw, transform_raw, crs)


def mean_stack(paths):
    """Streaming per-pixel mean of amplitude GeoTIFFs.

    ``paths`` is an ordered iterable of GeoTIFF paths. Returns
    ``(mean, transform, crs, dx)``.
    """
    return _streaming_mean(load_amp_square(p) for p in paths)


class AmplitudeStore:
    """Zarr-backed amplitude source: per-date square-pixel reads and mean
    stacks straight from an amplitude-only zarr, with no GeoTIFF export.

    The geotransform and CRS are read once from the store's ``spatial_ref``;
    per-date amplitude slices are resampled to square pixels on read with the
    same logic as :func:`load_amp_square`, so results are identical whether the
    pipeline reads the zarr or exported GeoTIFFs.

    Pickle-safe: the open zarr array handle is dropped on pickling and reopened
    lazily per process, so the store can be sent to joblib workers.
    """

    def __init__(self, zarr_path):
        self.zarr_path = str(zarr_path)
        store = zarr.open_group(self.zarr_path, mode="r")
        if not dict(store.attrs).get("amplitude_only", False):
            raise ValueError(
                "This zarr store is not amplitude-only. Stream with "
                "amplitude_only=True before reading amplitudes.")
        if "amplitude" not in store:
            raise ValueError("Zarr store is amplitude-only but has no 'amplitude' array.")
        sr = dict(store["spatial_ref"].attrs)
        gt = tuple(float(v) for v in sr["GeoTransform"].split())
        self.transform_raw = Affine.from_gdal(*gt)
        self.crs = rasterio.crs.CRS.from_wkt(sr["crs_wkt"])
        self.dates = [str(t) for t in store["time"][:]]
        self._index = {d: i for i, d in enumerate(self.dates)}
        self._amp = None  # lazily opened per process

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_amp"] = None
        return state

    def _amp_array(self):
        if self._amp is None:
            self._amp = zarr.open_group(self.zarr_path, mode="r")["amplitude"]
        return self._amp

    def load(self, date_tag):
        """Square-pixel amplitude for one date: ``(amp, transform, crs, dx)``."""
        i = self._index[date_tag]
        amp_raw = np.asarray(self._amp_array()[i, :, :], dtype=np.float32)
        return _resample_to_square(amp_raw, self.transform_raw, self.crs)

    def mean_stack(self, date_tags):
        """Streaming per-pixel mean over the given dates."""
        return _streaming_mean(self.load(d) for d in date_tags)


def export_amplitudes_from_zarr(zarr_path, output_dir, overwrite=False):
    """Write one ``{YYYYMMDD}_amp.tif`` per date from an amplitude-only zarr.

    Mirrors ``prepare_zarr_to_amp.py`` but uses rasterio. Returns the list of
    written/existing GeoTIFF paths in date order.
    """
    store = zarr.open_group(str(zarr_path), mode="r")
    if not dict(store.attrs).get("amplitude_only", False):
        raise ValueError(
            "This zarr store is not amplitude-only. Stream with "
            "amplitude_only=True before exporting amplitude GeoTIFFs.")
    if "amplitude" not in store:
        raise ValueError("Zarr store is amplitude-only but has no 'amplitude' array.")

    sr_attrs = dict(store["spatial_ref"].attrs)
    gt = tuple(float(v) for v in sr_attrs["GeoTransform"].split())
    transform = Affine.from_gdal(*gt)
    crs = rasterio.crs.CRS.from_wkt(sr_attrs["crs_wkt"])

    times = store["time"][:]
    n_times, n_rows, n_cols = store["amplitude"].shape
    os.makedirs(output_dir, exist_ok=True)

    out_paths = []
    for i, t in enumerate(times):
        date_str = str(t)
        out_path = Path(output_dir) / f"{date_str}_amp.tif"
        out_paths.append(out_path)
        if out_path.exists() and not overwrite:
            continue
        amp = np.asarray(store["amplitude"][i, :, :], dtype=np.float32)
        profile = {
            "driver": "GTiff", "height": n_rows, "width": n_cols, "count": 1,
            "dtype": "float32", "crs": crs, "transform": transform,
            "nodata": float("nan"), "tiled": True,
            "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(amp, 1)
    return out_paths
