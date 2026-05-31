"""NOAA CO-OPS reference data: water-level and air-gap fetch, temporal matching
to SAR acquisitions, and conversion to pseudo-clearance.

Ported from cells 9-11 of the reference notebooks. ``fetch_*`` pull from the
public NOAA Tides & Currents API; all times are UTC-aware.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

NOAA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
_APP = "BridgeSAR"


def url_get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _fetch_product(station, start, end, product, *, datum=None, timeout=60):
    """Pull a NOAA CO-OPS product over [start, end] in 30-day chunks."""
    out = []
    cur = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    while cur <= end:
        nxt = min(end, cur + pd.Timedelta(days=30))
        params = dict(begin_date=cur.strftime("%Y%m%d"), end_date=nxt.strftime("%Y%m%d"),
                      station=station, product=product, units="metric",
                      time_zone="gmt", format="json", application=_APP)
        if datum is not None:
            params["datum"] = datum
        url = NOAA_API + "?" + urllib.parse.urlencode(params)
        try:
            data = url_get_json(url, timeout=timeout)
        except Exception as e:
            print(f"  {station} {cur.date()}-{nxt.date()} failed: {e}")
            cur = nxt + pd.Timedelta(days=1)
            continue
        if "data" in data and data["data"]:
            d = pd.DataFrame(data["data"])
            d["t"] = pd.to_datetime(d["t"], utc=True)
            d["v"] = pd.to_numeric(d["v"], errors="coerce")
            out.append(d[["t", "v"]].dropna())
        cur = nxt + pd.Timedelta(days=1)
    if not out:
        raise RuntimeError(f"no {product} data for station {station}")
    return (pd.concat(out, ignore_index=True)
            .sort_values("t").drop_duplicates("t").reset_index(drop=True))


def fetch_waterlevel(station, start, end, *, datum="MLLW"):
    """NOAA 6-min water_level over [start, end] -> DataFrame(t, v) in metres."""
    return _fetch_product(station, start, end, "water_level", datum=datum)


def fetch_airgap(station, start, end):
    """NOAA air_gap over [start, end] -> DataFrame(t, v) in metres."""
    return _fetch_product(station, start, end, "air_gap")


def wl_window_mean(sar_times_utc, wl_times_utc, wl_values, halfwin_hours):
    """Mean of NOAA samples within +/-halfwin_hours of each SAR acquisition.
    Returns (mean_per_sar, n_samples_per_sar). Inputs must be UTC-aware."""
    sar_ns = pd.to_datetime(sar_times_utc, utc=True).astype(np.int64).values
    wl_ns = pd.to_datetime(wl_times_utc, utc=True).astype(np.int64).values
    wl_v = np.asarray(wl_values, float)
    half_ns = int(halfwin_hours * 3600 * 1_000_000_000)
    out = np.full(sar_ns.shape, np.nan)
    nobs = np.zeros(sar_ns.shape, int)
    for k, t in enumerate(sar_ns):
        lo = np.searchsorted(wl_ns, t - half_ns, side="left")
        hi = np.searchsorted(wl_ns, t + half_ns, side="right")
        v = wl_v[lo:hi]
        v = v[np.isfinite(v)]
        if v.size:
            out[k] = float(v.mean())
            nobs[k] = int(v.size)
    return out, nobs


def wl_acquisition_matched_mean(target_tags, time_by_tag, all_dates,
                                wl_times_utc, wl_values,
                                k_before, k_after, halfwin_hours):
    """Average NOAA values over the same K +/-N acquisition window that built
    each local amplitude mean. Returns (mean, total_obs, n_acq) per target."""
    wl_ns = pd.to_datetime(wl_times_utc, utc=True).astype(np.int64).values
    wl_v = np.asarray(wl_values, float)
    half_ns = int(halfwin_hours * 3600 * 1_000_000_000)
    N = len(target_tags)
    out = np.full(N, np.nan)
    nobs = np.zeros(N, int)
    n_acq = np.zeros(N, int)
    for k, tag in enumerate(target_tags):
        if tag not in all_dates:
            continue
        i = all_dates.index(tag)
        lo = max(0, i - k_before)
        hi = min(len(all_dates), i + k_after + 1)
        per_acq_means = []
        for d in all_dates[lo:hi]:
            t = time_by_tag.get(d)
            if t is None or pd.isna(t):
                continue
            t_ns = pd.Timestamp(t).tz_convert("UTC").value
            ll = np.searchsorted(wl_ns, t_ns - half_ns, side="left")
            rr = np.searchsorted(wl_ns, t_ns + half_ns, side="right")
            v = wl_v[ll:rr]
            v = v[np.isfinite(v)]
            if v.size:
                per_acq_means.append(float(v.mean()))
                nobs[k] += int(v.size)
        if per_acq_means:
            out[k] = float(np.mean(per_acq_means))
            n_acq[k] = len(per_acq_means)
    return out, nobs, n_acq


def smooth_airgap_curve(ag_df, halfwin_hours, grid_min=15, gap_steps=8):
    """Continuous +/-halfwin centered rolling-mean of the raw 6-min air gap.

    The raw samples are too noisy to plot on the clearance scale; this returns
    a uniform-grid smoothed DataFrame(t, v)."""
    grid = pd.date_range(ag_df["t"].min().floor(f"{grid_min}min"),
                         ag_df["t"].max().ceil(f"{grid_min}min"),
                         freq=f"{grid_min}min", tz="UTC")
    uniform = (ag_df.drop_duplicates("t").set_index("t")["v"]
               .reindex(grid).interpolate(method="time", limit=gap_steps))
    window = max(int(round(halfwin_hours * 2 * 60 / grid_min)), 1)
    min_per = max(int(window * 0.5), 1)
    smooth = uniform.rolling(window=window, center=True, min_periods=min_per).mean()
    return pd.DataFrame({"t": grid, "v": smooth.values}).dropna().reset_index(drop=True)


def stats(x, y):
    """Correlation R, RMSE, bias and N over finite pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, 0
    R = float(np.corrcoef(x[m], y[m])[0, 1])
    rmse = float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))
    bias = float(np.mean(x[m] - y[m]))
    return R, rmse, bias, int(m.sum())
