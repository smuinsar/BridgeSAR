"""Stream OPERA CSLC-S1 amplitude images and export to an amplitude-only Zarr.

Trimmed from the original ``opera_cslc_s1_stream.py``: the SLC/interferogram and
Goldstein-filtering paths are removed — BridgeSAR only needs VV amplitude
mosaics plus the static LOS layers and per-date orbit metadata.

Authentication uses ``earthaccess.login()``, which reads NASA Earthdata / ASF
credentials from ``~/.netrc``.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial

import h5py
from tqdm.auto import tqdm
import numpy as np
import requests
import xarray as xr
import zarr
import dask.array as da
from pyproj import Transformer
from scipy.interpolate import interp1d
from shapely.geometry import box

import earthaccess
import fsspec
import geopandas as gpd
import rioxarray  # noqa: F401  (activates rio accessor on xr.Dataset)

BURST_DB_URL = (
    "https://github.com/opera-adt/burst_db/releases/download/"
    "v0.17.0/burst-id-geometries-simple-0.17.0.geojson.zip"
)
GRID_PATH = "data"
META_PATH = "metadata/processing_information/input_burst_metadata"

_https_session = None
_fs = None


# --------------------------------------------------------------- streaming I/O
def _init_streaming_session():
    global _https_session, _fs
    _https_session = earthaccess.get_requests_https_session()
    _fs = fsspec.filesystem("https")


def open_granule_h5(granule):
    """Stream an earthaccess granule as h5py.File (no download)."""
    url = [l for l in granule.data_links() if l.endswith(".h5")][0]
    resp = _https_session.get(url, stream=True, allow_redirects=True)
    resp.raise_for_status()
    cloudfront_url = resp.url
    resp.close()
    f = _fs.open(cloudfront_url)
    return h5py.File(f, mode="r")


# ------------------------------------------------------------- data discovery
def find_burst_ids(track, lat_min, lat_max, lon_min, lon_max):
    print("Downloading burst-ID geometry database ...")
    resp = requests.get(BURST_DB_URL, allow_redirects=True)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        gdf_bursts = gpd.read_file(io.BytesIO(zf.read(zf.namelist()[0])))
    roi_box = box(lon_min, lat_min, lon_max, lat_max)
    track_str = f"t{track:03d}"
    gdf_track = gdf_bursts[gdf_bursts["burst_id_jpl"].str.startswith(track_str)]
    gdf_overlap = gdf_track[gdf_track.geometry.intersects(roi_box)]
    burst_ids = sorted(gdf_overlap["burst_id_jpl"].str.upper()
                       .str.replace("_", "-", regex=False).unique().tolist())
    print(f"Track {track}: {len(gdf_track)} total bursts, {len(burst_ids)} overlap the ROI")
    if not burst_ids:
        raise RuntimeError(f"No bursts for track {track} overlap the ROI.")
    return burst_ids


def search_cslc_granules(burst_ids, date_start, date_end):
    granules_all = []
    for burst_id in burst_ids:
        results = earthaccess.search_data(
            short_name="OPERA_L2_CSLC-S1_V1",
            granule_name=f"*{burst_id}*",
            temporal=(date_start, date_end),
        )
        granules_all.extend(results)
        print(f"  {burst_id}: {len(results)} granules")
    print(f"Found {len(granules_all)} CSLC granules total")
    if not granules_all:
        raise RuntimeError("No results found. Check date range and burst IDs.")
    return granules_all


def search_static_granules(burst_ids):
    static_granules = {}
    for burst_id in burst_ids:
        results = earthaccess.search_data(
            short_name="OPERA_L2_CSLC-S1-STATIC_V1",
            granule_name=f"*{burst_id}*",
        )
        if results:
            static_granules[burst_id] = results[0]
    print(f"Found static layers for {len(static_granules)}/{len(burst_ids)} burst IDs")
    return static_granules


def parse_and_group_granules(granules_all, track):
    granule_info = []
    for g in granules_all:
        h5_links = [l for l in g.data_links() if l.endswith(".h5")]
        if not h5_links:
            continue
        fname = h5_links[0].split("/")[-1]
        match = re.search(r"OPERA_L2_CSLC-S1_(T(\d+)-\d+-[A-Za-z0-9]+)_(\d{8})T", fname)
        if not match:
            continue
        burst_id = match.group(1)
        trk = int(match.group(2))
        date_str = match.group(3)
        if trk != track:
            continue
        granule_info.append({"burst_id": burst_id, "date": date_str, "granule": g})
    burst_date_map = defaultdict(dict)
    for gi in granule_info:
        burst_date_map[gi["burst_id"]][gi["date"]] = gi["granule"]
    print(f"{len(granule_info)} granules for Track {track}, "
          f"{len(burst_date_map)} unique burst IDs")
    return burst_date_map


def find_common_dates(burst_date_map, skip_months=None,
                      season_start=None, season_end=None):
    common_dates = None
    for bid in sorted(burst_date_map.keys()):
        bid_dates = set(burst_date_map[bid].keys())
        common_dates = bid_dates if common_dates is None else common_dates & bid_dates
    common_dates = sorted(common_dates)
    if skip_months is not None:
        common_dates = [d for d in common_dates if int(d[4:6]) not in skip_months]
    if season_start is not None and season_end is not None:
        common_dates = [d for d in common_dates if season_start <= d[4:8] <= season_end]
    print(f"{len(common_dates)} dates with complete burst coverage")
    return common_dates


# ------------------------------------------------------------ grid & metadata
def probe_grid_metadata(burst_date_map, common_dates, pol):
    first_bid = sorted(burst_date_map.keys())[0]
    first_granule = burst_date_map[first_bid][common_dates[-1]]
    h5_probe = open_granule_h5(first_granule)
    epsg = int(h5_probe[f"{GRID_PATH}/projection"][()])
    x_spacing = int(h5_probe[f"{GRID_PATH}/x_spacing"][()])
    y_spacing = int(h5_probe[f"{GRID_PATH}/y_spacing"][()])
    h5_probe.close()
    print(f"EPSG {epsg}, spacing {x_spacing} x {y_spacing} m")
    return epsg, x_spacing, y_spacing


def transform_roi_to_projected(lat_min, lat_max, lon_min, lon_max, epsg):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    corners_lon = [lon_min, lon_max, lon_min, lon_max]
    corners_lat = [lat_min, lat_min, lat_max, lat_max]
    corners_x, corners_y = transformer.transform(corners_lon, corners_lat)
    return (min(corners_x), max(corners_x), min(corners_y), max(corners_y))


def compute_roi_slices(h5f, roi_x_min, roi_x_max, roi_y_min, roi_y_max):
    x_coords = h5f[f"{GRID_PATH}/x_coordinates"][:]
    y_coords = h5f[f"{GRID_PATH}/y_coordinates"][:]
    x_idx = np.where((x_coords >= roi_x_min) & (x_coords <= roi_x_max))[0]
    y_idx = np.where((y_coords >= roi_y_min) & (y_coords <= roi_y_max))[0]
    if len(x_idx) == 0 or len(y_idx) == 0:
        return None
    col_start, col_stop = int(x_idx[0]), int(x_idx[-1]) + 1
    row_start, row_stop = int(y_idx[0]), int(y_idx[-1]) + 1
    return (row_start, row_stop, col_start, col_stop,
            x_coords[col_start:col_stop], y_coords[row_start:row_stop])


def build_mosaic_grid(burst_roi_info, x_spacing, y_spacing):
    all_x_min = min(info[0].min() for info in burst_roi_info.values())
    all_x_max = max(info[0].max() for info in burst_roi_info.values())
    all_y_min = min(info[1].min() for info in burst_roi_info.values())
    all_y_max = max(info[1].max() for info in burst_roi_info.values())
    mosaic_x = np.arange(all_x_min, all_x_max + abs(x_spacing) / 2, abs(x_spacing))
    mosaic_y = np.arange(all_y_max, all_y_min - abs(y_spacing) / 2, -abs(y_spacing))
    return mosaic_x, mosaic_y


def place_on_mosaic(mosaic, mosaic_x, mosaic_y, burst_data, burst_x, burst_y,
                    x_spacing, y_spacing):
    dx = mosaic_x[1] - mosaic_x[0] if len(mosaic_x) > 1 else abs(x_spacing)
    dy = mosaic_y[0] - mosaic_y[1] if len(mosaic_y) > 1 else abs(y_spacing)
    x_offset = int(round((burst_x[0] - mosaic_x[0]) / dx))
    y_offset = int(round((mosaic_y[0] - burst_y[0]) / dy))
    h, w = burst_data.shape
    dst_h = min(h, mosaic.shape[0] - y_offset)
    dst_w = min(w, mosaic.shape[1] - x_offset)
    if dst_h <= 0 or dst_w <= 0 or x_offset < 0 or y_offset < 0:
        return
    src = burst_data[:dst_h, :dst_w]
    target = mosaic[y_offset:y_offset + dst_h, x_offset:x_offset + dst_w]
    fill_mask = np.isnan(target) & np.isfinite(src) & (src != 0)
    target[fill_mask] = src[fill_mask]


def probe_bursts_for_overlap(all_burst_ids, burst_date_map, common_dates,
                             roi_x_min, roi_x_max, roi_y_min, roi_y_max):
    burst_roi_info = {}
    burst_slices = {}
    for bid in all_burst_ids:
        granule = burst_date_map[bid][common_dates[-1]]
        h5f = open_granule_h5(granule)
        result = compute_roi_slices(h5f, roi_x_min, roi_x_max, roi_y_min, roi_y_max)
        h5f.close()
        if result is None:
            continue
        row_start, row_stop, col_start, col_stop, x_roi, y_roi = result
        burst_roi_info[bid] = (x_roi, y_roi)
        burst_slices[bid] = (row_start, row_stop, col_start, col_stop)
    active_bursts = sorted(burst_roi_info.keys())
    if not active_bursts:
        raise RuntimeError("No bursts overlap the ROI.")
    return burst_roi_info, burst_slices, active_bursts


def stream_los_static_layers(active_bursts, static_granules,
                             roi_x_min, roi_x_max, roi_y_min, roi_y_max,
                             mosaic_x, mosaic_y, mosaic_shape, x_spacing, y_spacing):
    los_east = np.full(mosaic_shape, np.nan, dtype=np.float32)
    los_north = np.full(mosaic_shape, np.nan, dtype=np.float32)
    inc_angle = np.full(mosaic_shape, np.nan, dtype=np.float32)
    for bid in active_bursts:
        if bid not in static_granules:
            continue
        h5f = open_granule_h5(static_granules[bid])
        result = compute_roi_slices(h5f, roi_x_min, roi_x_max, roi_y_min, roi_y_max)
        if result is None:
            h5f.close()
            continue
        rs, re_, cs, ce, x_roi_s, y_roi_s = result
        los_e = h5f["data/los_east"][rs:re_, cs:ce].astype(np.float32)
        los_n = h5f["data/los_north"][rs:re_, cs:ce].astype(np.float32)
        inc_a = h5f["data/local_incidence_angle"][rs:re_, cs:ce].astype(np.float32)
        h5f.close()
        place_on_mosaic(los_east, mosaic_x, mosaic_y, los_e, x_roi_s, y_roi_s, x_spacing, y_spacing)
        place_on_mosaic(los_north, mosaic_x, mosaic_y, los_n, x_roi_s, y_roi_s, x_spacing, y_spacing)
        place_on_mosaic(inc_angle, mosaic_x, mosaic_y, inc_a, x_roi_s, y_roi_s, x_spacing, y_spacing)
    sum_sq = np.clip(los_east ** 2 + los_north ** 2, 0.0, 1.0)
    los_up = np.sqrt(1.0 - sum_sq).astype(np.float32)
    return los_east, los_north, los_up, inc_angle


# -------------------------------------------------------------- SLC streaming
def _read_orbit_at_mid_sensing(h5f, mosaic_x, mosaic_y, epsg):
    orb = h5f["/metadata/orbit"]
    ref_epoch_str = orb["reference_epoch"][()].decode()
    orb_time = orb["time"][:]
    pos = np.column_stack([orb[f"position_{p}"][:] for p in ("x", "y", "z")])
    vel = np.column_stack([orb[f"velocity_{p}"][:] for p in ("x", "y", "z")])
    inp = h5f[f"{META_PATH}"]
    t_start_dt = datetime.fromisoformat(inp["sensing_start"][()].decode())
    t_stop_dt = datetime.fromisoformat(inp["sensing_stop"][()].decode())
    ref_epoch_dt = datetime.fromisoformat(ref_epoch_str)
    t_mid_dt = t_start_dt + (t_stop_dt - t_start_dt) / 2
    t_mid_sec = (t_mid_dt - ref_epoch_dt).total_seconds()
    pos_mid = interp1d(orb_time, pos, axis=0, kind="cubic")(t_mid_sec)
    vel_mid = interp1d(orb_time, vel, axis=0, kind="cubic")(t_mid_sec)
    cx = float(mosaic_x[len(mosaic_x) // 2])
    cy = float(mosaic_y[len(mosaic_y) // 2])
    to_ecef = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4978", always_xy=True)
    gx, gy, gz = to_ecef.transform(cx, cy, 0.0)
    slant_range = float(np.linalg.norm(pos_mid - np.array([gx, gy, gz])))
    return pos_mid, vel_mid, slant_range, t_mid_dt.isoformat()


def _process_date(date_str, active_bursts, burst_date_map, burst_slices,
                  burst_roi_info, mosaic_shape, mosaic_x, mosaic_y,
                  pol, epsg, x_spacing, y_spacing):
    mosaic = np.full(mosaic_shape, np.nan + 1j * np.nan, dtype=np.complex64)
    orbit_info = None
    for bid in active_bursts:
        granule = burst_date_map[bid][date_str]
        rs, re_, cs, ce = burst_slices[bid]
        try:
            h5f = open_granule_h5(granule)
            slc_roi = h5f[f"{GRID_PATH}/{pol}"][rs:re_, cs:ce]
            if orbit_info is None:
                pos_mid, vel_mid, sr, smid = _read_orbit_at_mid_sensing(
                    h5f, mosaic_x, mosaic_y, epsg)
                orbit_info = {"pos": pos_mid, "vel": vel_mid,
                              "slant_range": sr, "sensing_mid": smid}
            h5f.close()
        except Exception as e:
            print(f"  {date_str} / {bid}: SKIPPED -- {type(e).__name__}: {e}")
            return (date_str, None, None)
        slc_roi = slc_roi.astype(np.complex64)
        slc_roi[slc_roi == 0] = np.nan + 1j * np.nan
        burst_x, burst_y = burst_roi_info[bid]
        place_on_mosaic(mosaic, mosaic_x, mosaic_y, slc_roi, burst_x, burst_y,
                        x_spacing, y_spacing)
    return (date_str, mosaic, orbit_info)


def stream_amplitude_parallel(common_dates, active_bursts, burst_date_map,
                              burst_slices, burst_roi_info, mosaic_shape,
                              mosaic_x, mosaic_y, pol, epsg, x_spacing, y_spacing,
                              n_workers, zarr_path):
    """Stream amplitude for all dates in parallel, writing each to zarr."""
    worker = partial(_process_date, active_bursts=active_bursts,
                     burst_date_map=burst_date_map, burst_slices=burst_slices,
                     burst_roi_info=burst_roi_info, mosaic_shape=mosaic_shape,
                     mosaic_x=mosaic_x, mosaic_y=mosaic_y, pol=pol, epsg=epsg,
                     x_spacing=x_spacing, y_spacing=y_spacing)
    date_to_index = {d: i for i, d in enumerate(common_dates)}
    orbit_stack = {}
    skipped_dates = []
    store = zarr.open(zarr_path, mode="r+")
    print(f"Streaming {len(common_dates)} dates with {n_workers} workers ...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, d): d for d in common_dates}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Streaming amplitudes", unit="date"):
            date_str, mosaic, orbit_info = fut.result()
            if mosaic is not None:
                time_idx = date_to_index[date_str]
                store["amplitude"][time_idx, :, :] = np.abs(mosaic).astype(np.float32)
                del mosaic
                orbit_stack[date_str] = orbit_info
            else:
                skipped_dates.append(date_str)
    dates_sorted = sorted(orbit_stack.keys())
    print(f"Wrote {len(dates_sorted)} dates to zarr; skipped {len(skipped_dates)}")
    return orbit_stack, dates_sorted, skipped_dates


# ------------------------------------------------------------------ zarr I/O
def create_zarr_store(zarr_path, common_dates, mosaic_x, mosaic_y,
                      los_east, los_north, los_up, inc_angle,
                      epsg, x_spacing, y_spacing, active_bursts, config):
    """Pre-allocate an amplitude-only zarr store with NaN placeholders."""
    n_dates = len(common_dates)
    ny, nx = len(mosaic_y), len(mosaic_x)
    amp_ph = da.full((n_dates, ny, nx), np.nan, dtype=np.float32, chunks=(1, ny, nx))
    sat_ph = da.full((n_dates, 3), np.nan, dtype=np.float64, chunks=(1, 3))
    sr_ph = da.full((n_dates,), np.nan, dtype=np.float64, chunks=(1,))
    data_vars = {
        "amplitude": (["time", "y", "x"], amp_ph,
                      {"long_name": "SLC amplitude (|complex SLC|)", "units": "linear"}),
        "los_east": (["y", "x"], los_east, {"long_name": "LOS unit vector East"}),
        "los_north": (["y", "x"], los_north, {"long_name": "LOS unit vector North"}),
        "los_up": (["y", "x"], los_up, {"long_name": "LOS unit vector Up"}),
        "local_incidence_angle": (["y", "x"], inc_angle,
                                  {"long_name": "Local incidence angle", "units": "degrees"}),
        "sat_position": (["time", "xyz"], sat_ph, {"long_name": "Satellite ECEF position"}),
        "sat_velocity": (["time", "xyz"], sat_ph, {"long_name": "Satellite ECEF velocity"}),
        "slant_range_center": (["time"], sr_ph, {"long_name": "Slant range to center"}),
    }
    ds = xr.Dataset(data_vars, coords={"time": common_dates, "y": mosaic_y,
                                       "x": mosaic_x, "xyz": ["X", "Y", "Z"]})
    ds = ds.rio.write_crs(f"EPSG:{epsg}").rio.write_transform()
    ds.attrs.update({"mission": "Sentinel-1", "product_type": "CSLC-S1",
                     "source": "OPERA_L2_CSLC-S1_V1", "amplitude_only": True,
                     "track_number": config["track"], "polarization": config["pol"],
                     "epsg": epsg, "x_spacing": float(x_spacing),
                     "y_spacing": float(y_spacing),
                     "burst_ids": json.dumps(active_bursts),
                     "date_range": f"{config['date_start']} to {config['date_end']}"})
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5)
    encoding = {v: {"compressor": compressor} for v in ds.data_vars if v != "spatial_ref"}
    ds.to_zarr(zarr_path, mode="w", encoding=encoding)
    print(f"Pre-allocated amplitude-only zarr at {zarr_path} "
          f"(time={n_dates}, y={ny}, x={nx})")


def drop_skipped_dates(zarr_path, skipped_dates):
    if not skipped_dates:
        return
    import shutil
    ds = xr.open_zarr(zarr_path)
    keep_mask = [t not in skipped_dates for t in list(ds.time.values)]
    ds = ds.sel(time=keep_mask)
    tmp_path = str(zarr_path) + ".tmp"
    compressor = zarr.codecs.BloscCodec(cname="zstd", clevel=5)
    encoding = {v: {"compressor": compressor} for v in ds.data_vars if v != "spatial_ref"}
    ds.to_zarr(tmp_path, mode="w", encoding=encoding)
    ds.close()
    shutil.rmtree(zarr_path)
    shutil.move(tmp_path, zarr_path)
    print(f"Dropped {len(skipped_dates)} skipped dates from zarr")


def finalize_zarr_metadata(zarr_path, orbit_stack, dates_sorted, skipped_dates,
                           active_bursts, burst_date_map):
    store = zarr.open(zarr_path, mode="r+")
    for idx, d in enumerate(dates_sorted):
        if d in orbit_stack:
            store["sat_position"][idx, :] = orbit_stack[d]["pos"]
            store["sat_velocity"][idx, :] = orbit_stack[d]["vel"]
            store["slant_range_center"][idx] = orbit_stack[d]["slant_range"]
    sensing_mid = [orbit_stack[d]["sensing_mid"] for d in dates_sorted if d in orbit_stack]
    store.attrs["sensing_mid_times"] = json.dumps(sensing_mid)
    if skipped_dates:
        store.attrs["skipped_dates"] = json.dumps(sorted(skipped_dates))
    print(f"Finalized zarr metadata: {len(orbit_stack)} orbit records")


# ----------------------------------------------------------------- top-level
def stream_amplitudes(cfg, date_start, date_end, *, pol="VV", n_workers=8,
                      zarr_path=None, skip_months=None,
                      season_start=None, season_end=None):
    """Stream OPERA CSLC-S1 amplitude for a bridge config into an amplitude-only
    zarr store. Requires NASA Earthdata credentials in ``~/.netrc``.

    Returns the zarr path.
    """
    cfg.resolve_paths()
    track = int(cfg.track_number)
    zarr_path = str(zarr_path or cfg.zarr_path)

    burst_ids = find_burst_ids(track, cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max)
    earthaccess.login()
    _init_streaming_session()

    granules_all = search_cslc_granules(burst_ids, date_start, date_end)
    static_granules = search_static_granules(burst_ids)
    burst_date_map = parse_and_group_granules(granules_all, track)
    common_dates = find_common_dates(burst_date_map, skip_months, season_start, season_end)

    epsg, x_spacing, y_spacing = probe_grid_metadata(burst_date_map, common_dates, pol)
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = transform_roi_to_projected(
        cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max, epsg)
    burst_roi_info, burst_slices, active_bursts = probe_bursts_for_overlap(
        sorted(burst_date_map.keys()), burst_date_map, common_dates,
        roi_x_min, roi_x_max, roi_y_min, roi_y_max)
    mosaic_x, mosaic_y = build_mosaic_grid(burst_roi_info, x_spacing, abs(y_spacing))
    mosaic_shape = (len(mosaic_y), len(mosaic_x))

    los_east, los_north, los_up, inc_angle = stream_los_static_layers(
        active_bursts, static_granules, roi_x_min, roi_x_max, roi_y_min, roi_y_max,
        mosaic_x, mosaic_y, mosaic_shape, x_spacing, y_spacing)

    config = {"track": track, "pol": pol, "date_start": date_start, "date_end": date_end}
    create_zarr_store(zarr_path, common_dates, mosaic_x, mosaic_y,
                      los_east, los_north, los_up, inc_angle,
                      epsg, x_spacing, y_spacing, active_bursts, config)
    del los_east, los_north, los_up, inc_angle

    orbit_stack, dates_sorted, skipped_dates = stream_amplitude_parallel(
        common_dates, active_bursts, burst_date_map, burst_slices, burst_roi_info,
        mosaic_shape, mosaic_x, mosaic_y, pol, epsg, x_spacing, y_spacing,
        n_workers, zarr_path)
    drop_skipped_dates(zarr_path, skipped_dates)
    finalize_zarr_metadata(zarr_path, orbit_stack, dates_sorted, skipped_dates,
                           active_bursts, burst_date_map)
    print("Done streaming amplitudes.")
    return zarr_path
