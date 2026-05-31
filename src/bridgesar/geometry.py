"""Geometry utilities: radar LOS geometry, bridge orientation, and the
amplitude-image rotation / OSM-projection helpers used by the stripe fit.

Faithful port of cells 1 and 2 of the reference notebooks. Functions are pure
(no global state) so they can be reused for any bridge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import geopandas as gpd
from rasterio.transform import rowcol as rio_rowcol
from scipy.ndimage import rotate as ndi_rotate
from shapely.geometry import MultiLineString
from shapely.ops import linemerge


# ---------------------------------------------------------------------------
# Radar LOS geometry from the streamed zarr store
# ---------------------------------------------------------------------------
def los_geometry_from_zarr(zarr_path, deck_height_m):
    """Median LOS geometry and sensing times from an OPERA CSLC-S1 zarr store.

    Returns a dict with ``theta`` (incidence, rad), ``theta_deg``, ``los_az``
    (deg, clockwise from north), ``los_e/n/u``, ``exp_layover_m`` and the
    per-date sensing-time map ``time_by_tag``.
    """
    z = zarr.open(str(zarr_path), mode="r")
    los_e = float(np.nanmedian(np.array(z["los_east"])))
    los_n = float(np.nanmedian(np.array(z["los_north"])))
    los_u = float(np.nanmedian(np.array(z["los_up"])))
    theta = float(np.arccos(np.clip(los_u, -1, 1)))
    theta_deg = float(np.degrees(theta))
    los_az = float(np.degrees(np.arctan2(los_e, los_n))) % 360.0
    exp_layover_m = deck_height_m / np.tan(theta)

    raw = z.attrs["sensing_mid_times"]
    sensing_times = pd.to_datetime(ast.literal_eval(raw) if isinstance(raw, str)
                                   else list(raw), utc=True)
    time_by_tag = {t.strftime("%Y%m%d"): t for t in sensing_times}

    return {
        "los_e": los_e, "los_n": los_n, "los_u": los_u,
        "theta": theta, "theta_deg": theta_deg, "los_az": los_az,
        "exp_layover_m": float(exp_layover_m),
        "sensing_times": sensing_times, "time_by_tag": time_by_tag,
    }


def load_bridge_line(osm_geojson, name_regex=None, utm_epsg=None):
    """Load the OSM bridge feature(s), merge into a single centerline, and
    reproject to the local UTM frame. Returns a shapely LineString.
    """
    osm = gpd.read_file(osm_geojson)
    if name_regex:
        mask = osm["name"].str.contains(name_regex, case=False, regex=True, na=False)
        sel = osm[mask]
        if sel.empty:
            sel = osm
    else:
        sel = osm
    if utm_epsg is not None:
        sel = sel.to_crs(f"EPSG:{utm_epsg}")
    geom_merged = linemerge(MultiLineString(list(sel.geometry)))
    bridge_line = (max(geom_merged.geoms, key=lambda g: g.length)
                   if isinstance(geom_merged, MultiLineString) else geom_merged)
    return bridge_line


def bridge_azimuth(bridge_line):
    """OSM bridge azimuth (deg, in [0, 180)) from PCA of the polyline vertices."""
    xy_arr = np.array(bridge_line.coords)
    xc = xy_arr - xy_arr.mean(axis=0)
    _, _, Vt = np.linalg.svd(xc, full_matrices=False)
    ev = Vt[0]
    return float(np.degrees(np.arctan2(ev[0], ev[1]))) % 180.0


# ---------------------------------------------------------------------------
# Amplitude-image rotation + OSM projection
# ---------------------------------------------------------------------------
def find_best_angle(amp, bridge_az_prior):
    """Coarse-to-fine rotation refinement that maximises the column-summed
    bright mask, centred on the OSM azimuth prior."""
    amp_pos = np.where(np.isfinite(amp) & (amp > 0), amp, 0.0).astype(np.float32)
    bright_thr = float(np.nanpercentile(amp_pos[amp_pos > 0], 95))
    bright_mask = (amp_pos > bright_thr).astype(np.float32)

    def axis_score(angle):
        rot = ndi_rotate(bright_mask, angle=angle, reshape=True, order=0,
                         cval=0.0, mode="constant")
        return float(rot.sum(axis=0).max())

    center = ((bridge_az_prior + 90.0) % 180.0) - 90.0
    coarse = np.arange(center - 5.0, center + 5.0 + 1e-6, 1.0)
    coarse_scores = np.array([axis_score(a) for a in coarse])
    c_best = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.arange(c_best - 1.0, c_best + 1.0 + 1e-6, 0.1)
    fine_scores = np.array([axis_score(a) for a in fine])
    return float(fine[int(np.argmax(fine_scores))])


def rotate_amp(amp, angle):
    """Rotate an amplitude image while preserving the NaN mask."""
    amp_filled = np.where(np.isfinite(amp), amp, 0.0)
    valid_mask = np.isfinite(amp).astype(np.float32)
    rot_amp = ndi_rotate(amp_filled, angle=angle, reshape=True, order=1, cval=0.0)
    rot_valid = ndi_rotate(valid_mask, angle=angle, reshape=True, order=1, cval=0.0)
    return np.where(rot_valid > 0.5, rot_amp, np.nan)


def detect_tower_rows(img, k_mad=4.0, expand=1):
    """Return boolean array (True = tower row). Bright cross-bridge streaks."""
    n_rows = img.shape[0]
    row_max = np.array([np.nanmax(img[r]) if np.isfinite(img[r]).any() else np.nan
                        for r in range(n_rows)])
    row_mean = np.nanmean(img, axis=1)
    med = np.nanmedian(row_max)
    mad = np.nanmedian(np.abs(row_max - med))
    thresh = med + k_mad * 1.4826 * mad
    tower = (row_max > thresh) & (row_mean > np.nanmedian(row_mean) * 1.3)
    if expand > 0:
        for i in np.where(tower)[0]:
            tower[max(0, i - expand):min(n_rows, i + expand + 1)] = True
    return tower


def utm_to_rot_pix(x_utm, y_utm, rot_shape, in_shape, angle_deg, transform):
    row_i, col_i = rio_rowcol(transform, x_utm, y_utm)
    rh, rw = rot_shape
    ih, iw = in_shape
    xi = col_i - iw / 2.0
    yi = row_i - ih / 2.0
    a = np.deg2rad(angle_deg)
    x = xi * np.cos(a) + yi * np.sin(a)
    y = -xi * np.sin(a) + yi * np.cos(a)
    return float(x + rw / 2.0), float(y + rh / 2.0)


def osm_centerline_in_rot_pix(bridge_line, rot_shape, in_shape, angle_deg, transform):
    coords = np.array(bridge_line.coords)
    pts = np.array([utm_to_rot_pix(x, y, rot_shape, in_shape, angle_deg, transform)
                    for x, y in coords])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pa = Vt[0]
    rot_deg = ((float(np.degrees(np.arctan2(pa[0], pa[1]))) + 90.0) % 180.0) - 90.0
    proj = centered @ pa
    return {"center_col": float(centroid[0]), "center_row": float(centroid[1]),
            "length_px": float(proj.max() - proj.min()),
            "rotation_deg": rot_deg}


def refine_center_row(amp_rot, init_center_row, center_col, length_px,
                      col_half_px=60, search_half_px=0):
    """Locked-near-OSM center-row refinement (plus2 default search_half_px=0)."""
    nrows, ncols = amp_rot.shape
    img = np.where(np.isfinite(amp_rot), amp_rot, 0.0).astype(np.float32)
    c0 = int(max(0, np.floor(center_col - col_half_px)))
    c1 = int(min(ncols, np.ceil(center_col + col_half_px)))
    rp = img[:, c0:c1].mean(axis=1)
    cum = np.concatenate([[0.0], np.cumsum(rp)])
    half = int(round(length_px / 2.0))
    r_lo = max(half, int(np.floor(init_center_row - search_half_px)))
    r_hi = min(nrows - half - 1, int(np.ceil(init_center_row + search_half_px)))
    best = -np.inf
    best_r = int(init_center_row)
    for r in range(r_lo, r_hi + 1):
        s = (cum[r + half] - cum[r - half]) / (2 * half)
        if s > best:
            best = s
            best_r = r
    return best_r


def polygon_corners(center_col, half_width_px, rotation_deg, length_px, center_row):
    dr, dc = length_px / 2.0, half_width_px
    local = np.array([[-dr, -dc], [-dr, +dc], [+dr, +dc], [+dr, -dc]])
    a = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = local @ R.T
    rotated[:, 0] += center_row
    rotated[:, 1] += center_col
    return rotated


def polygon_centerline(corners):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    lengths = [np.linalg.norm(corners[j] - corners[i]) for i, j in edges]
    order = np.argsort(lengths)
    short = [edges[order[0]], edges[order[1]]]
    mids = [(corners[i] + corners[j]) / 2.0 for i, j in short]
    p1, p2 = mids
    dr, dc = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dr) < 1e-9:
        return float(np.sign(dc) * 1e9), float(p1[1]), (p1, p2)
    slope = dc / dr
    intercept = p1[1] - slope * p1[0]
    return float(slope), float(intercept), (p1, p2)


def perp_separation_px(line_a, line_b, eval_row):
    """Perpendicular pixel distance between two stripe centerlines at a row."""
    sa, ia = line_a[0], line_a[1]
    sb, ib = line_b[0], line_b[1]
    ca = sa * eval_row + ia
    return float(abs((sb * eval_row - ca + ib) / np.linalg.norm([sb, -1.0])))


# Alias matching the notebook's two identical names.
_perp_sep = perp_separation_px
