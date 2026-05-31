"""End-to-end clearance pipeline.

``ClearancePipeline`` holds the per-bridge geometry (radar LOS, OSM centerline,
global single-bounce fit) and drives the per-date Method-E++ stripe fit, the
stripe-spacing -> clearance conversion, the parallel time-series run, quality
filtering, and the NOAA reference comparison.

This consolidates cells 1, 3, 4, 5, 6, 8 and 11 of the reference notebooks into
a single object so the same code applies to any bridge via its
:class:`~bridgesar.config.BridgeConfig`.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BridgeConfig, HyperParams
from .amplitude import mean_stack, AmplitudeStore
from .geometry import (
    los_geometry_from_zarr, load_bridge_line, bridge_azimuth,
    find_best_angle, rotate_amp, detect_tower_rows,
    osm_centerline_in_rot_pix, refine_center_row, perp_separation_px as _perp_sep,
)
from .stripes import (
    method_E, fit_quality, snr_sigma_spacing, thickness_corrections,
)
from . import reference as _ref


class ClearancePipeline:
    def __init__(self, cfg: BridgeConfig, hp: HyperParams | None = None,
                 make_profile_fig: bool = False, fig_dir=None,
                 amplitude_source: str = "auto"):
        self.cfg = cfg.resolve_paths()
        self.hp = hp if hp is not None else cfg.hyper()
        self.make_profile_fig = make_profile_fig
        self.fig_dir = Path(fig_dir) if fig_dir is not None else (
            Path(cfg.out_dir) / "figures" if cfg.out_dir else None)
        if amplitude_source not in ("auto", "zarr", "geotiff"):
            raise ValueError("amplitude_source must be 'auto', 'zarr' or 'geotiff'")
        self.amplitude_source = amplitude_source
        self._amp_store = None
        self._is_setup = False

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Load LOS geometry, amplitude dates, OSM centerline, and fit the
        global single-bounce stripe. Call once before processing."""
        cfg = self.cfg
        geo = los_geometry_from_zarr(cfg.zarr_path, cfg.deck_height_m)
        self.los_e = geo["los_e"]
        self.los_az = geo["los_az"]
        self.theta = geo["theta"]
        self.exp_layover_m = geo["exp_layover_m"]
        self.time_by_tag = geo["time_by_tag"]
        self.sensing_times = geo["sensing_times"]

        self._init_amplitude_source()

        self.bridge_line = load_bridge_line(cfg.osm_geojson, cfg.osm.name_regex,
                                            cfg.osm.utm_epsg)
        self.psi_bridge_osm = bridge_azimuth(self.bridge_line)

        self._fit_global_single_bounce()
        self._is_setup = True
        return self

    def _init_amplitude_source(self):
        """Pick the amplitude source: read straight from the zarr (default) or
        from exported ``*_amp.tif`` GeoTIFFs. Sets ``self.all_dates`` and, for
        the GeoTIFF path, ``self.date_to_path``."""
        cfg = self.cfg
        zarr_ok = bool(cfg.zarr_path) and Path(cfg.zarr_path).exists()
        use_zarr = (self.amplitude_source == "zarr"
                    or (self.amplitude_source == "auto" and zarr_ok))

        if use_zarr:
            if not zarr_ok:
                raise RuntimeError(
                    f"amplitude_source='zarr' but no zarr store at {cfg.zarr_path}")
            self._amp_store = AmplitudeStore(cfg.zarr_path)
            self.all_dates = sorted(self._amp_store.dates)
            self.date_to_path = None
            return

        # GeoTIFF fallback (amplitude_source='geotiff', or 'auto' with no zarr).
        amp_paths = sorted(Path(cfg.amp_dir).glob("*_amp.tif"))
        pat = re.compile(r"^(\d{8})_amp\.tif$")
        self.all_dates = sorted(m.group(1) for p in amp_paths if (m := pat.match(p.name)))
        self.date_to_path = {d: Path(cfg.amp_dir) / f"{d}_amp.tif" for d in self.all_dates}
        if not self.all_dates:
            raise RuntimeError(
                f"no amplitude source found: no zarr at {cfg.zarr_path} and no "
                f"'*_amp.tif' rasters in {cfg.amp_dir}")

    def _mean_stack(self, dates):
        """Mean amplitude over ``dates`` from whichever source is active."""
        if self._amp_store is not None:
            return self._amp_store.mean_stack(dates)
        return mean_stack([self.date_to_path[d] for d in dates])

    @staticmethod
    def _suppress():
        """Combined warnings + numpy-error suppression for the heavy numeric
        kernels (All-NaN slices, invalid-value divides, etc.)."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                yield
        return _ctx()

    def _fit_global_single_bounce(self):
        cfg = self.cfg
        with self._suppress():
            self.__fit_global_single_bounce_impl()

    def __fit_global_single_bounce_impl(self):
        cfg = self.cfg
        amp_global, transform_global, _, dx = self._mean_stack(self.all_dates)
        self.dx = dx
        self.exp_dx = self.exp_layover_m / dx
        self.best_angle = find_best_angle(amp_global, self.psi_bridge_osm)
        rot_global = rotate_amp(amp_global, self.best_angle)
        if cfg.mask_towers:
            rot_global[detect_tower_rows(rot_global, k_mad=cfg.tower_k_mad,
                                         expand=cfg.tower_expand), :] = np.nan
        osm_rot = osm_centerline_in_rot_pix(self.bridge_line, rot_global.shape,
                                            amp_global.shape, self.best_angle,
                                            transform_global)
        osm_rot["center_row"] = float(refine_center_row(
            rot_global, osm_rot["center_row"], osm_rot["center_col"],
            osm_rot["length_px"]))
        self.osm_rot = osm_rot
        res_E_global = method_E(rot_global, self.exp_dx, osm_rot, dx, self.hp, self.los_e)
        self.S_global_corners = res_E_global["corners"][0]
        self.S_global_centerline = res_E_global["centerlines"][0]

    # --------------------------------------------------------------- per date
    def local_dates_for(self, date_tag):
        i = self.all_dates.index(date_tag)
        lo = max(0, i - self.cfg.k_before)
        hi = min(len(self.all_dates), i + self.cfg.k_after + 1)
        return self.all_dates[lo:hi]

    def process_one_avg(self, date_tag):
        """Method-E++ fit for one date on the local-mean amplitude, merged with
        the global single-bounce stripe; returns a row of clearance estimates."""
        cfg = self.cfg
        hp = self.hp

        local_dates = self.local_dates_for(date_tag)
        amp_local, _, _, dx_local = self._mean_stack(local_dates)
        exp_dx_local = self.exp_layover_m / dx_local
        rot_local = rotate_amp(amp_local, self.best_angle)
        if cfg.mask_towers:
            rot_local[detect_tower_rows(rot_local, k_mad=cfg.tower_k_mad,
                                        expand=cfg.tower_expand), :] = np.nan

        res = method_E(rot_local, exp_dx_local, self.osm_rot, dx_local, hp, self.los_e)
        cr = res["params"]["center_row"]

        # Merge: S from the global fit, D + T from the local fit.
        merged_corners = [self.S_global_corners, res["corners"][1], res["corners"][2]]
        merged_lines = [self.S_global_centerline, res["centerlines"][1], res["centerlines"][2]]
        res_merged = dict(res)
        res_merged["corners"] = merged_corners
        res_merged["centerlines"] = merged_lines

        q = fit_quality(rot_local, res_merged, exp_dx_local, dx_local, hp)

        if self.make_profile_fig and self.fig_dir is not None:
            from .plotting import plot_accumulated_profile_fit
            self.fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_accumulated_profile_fit(
                    res, cr, dx_local,
                    title=f"{cfg.bridge_name} {cfg.track} {date_tag} - Profile fit",
                    save_path=self.fig_dir / f"{cfg.track}_{date_tag}_local_profile_fit.png")
            except Exception as e:
                print(f"[{date_tag}] figure failed: {e}")

        sd_px = _perp_sep(merged_lines[0], merged_lines[1], cr)
        dt_px = _perp_sep(merged_lines[1], merged_lines[2], cr)
        st_px = _perp_sep(merged_lines[0], merged_lines[2], cr)

        psi_bridge = self.best_angle % 180.0
        dpsi = ((self.los_az - psi_bridge + 90.0) % 180.0) - 90.0
        geom_factor = abs(np.sin(np.deg2rad(dpsi)))
        tanT = np.tan(self.theta)
        sd_m, dt_m, st_m = sd_px * dx_local, dt_px * dx_local, st_px * dx_local
        T = cfg.bridge_thickness_m
        c_SD, c_DT, c_ST = thickness_corrections(cfg.scatter_model, T)
        H_SD_raw = sd_m * tanT / geom_factor
        H_DT_raw = dt_m * tanT / geom_factor
        H_ST_raw = (st_m * tanT) / (2.0 * geom_factor)
        H_SD = H_SD_raw + c_SD
        H_DT = H_DT_raw + c_DT
        H_ST = H_ST_raw + c_ST
        H_mean = float(np.mean([H_SD, H_DT, H_ST]))

        # SNR-based per-date uncertainty (the package's clearance uncertainty).
        sig_sd_m, sig_dt_m = snr_sigma_spacing(q, dx_local, hp)
        sigma_H_SD = sig_sd_m * tanT / geom_factor
        sigma_H_DT = sig_dt_m * tanT / geom_factor
        sigma_H_ST = (tanT / (2.0 * geom_factor)) * np.sqrt(sig_sd_m ** 2 + sig_dt_m ** 2)
        sigma_H_mean = float(np.sqrt(sigma_H_SD ** 2 + sigma_H_DT ** 2 + sigma_H_ST ** 2) / 3.0)

        return {"spacing_SD_m": sd_m, "spacing_DT_m": dt_m, "spacing_ST_m": st_m,
                "H_SD_raw_m": H_SD_raw, "H_DT_raw_m": H_DT_raw, "H_ST_raw_m": H_ST_raw,
                "H_SD_m": H_SD, "H_DT_m": H_DT, "H_ST_m": H_ST, "H_mean_m": H_mean,
                "sigma_sp_SD_m": sig_sd_m, "sigma_sp_DT_m": sig_dt_m,
                "sigma_H_SD_m": sigma_H_SD, "sigma_H_DT_m": sigma_H_DT,
                "sigma_H_ST_m": sigma_H_ST, "sigma_H_mean_m": sigma_H_mean,
                "snr_floor": float(hp.snr_floor),
                "best_angle_deg": self.best_angle, "geom_factor": geom_factor,
                "scatter_model": cfg.scatter_model,
                "corr_SD_m": c_SD, "corr_DT_m": c_DT, "corr_ST_m": c_ST,
                "converged": res["converged"], "n_iter": res["n_iter"],
                "score_SD": res["score_SD"], "score_DT": res["score_DT"],
                "dpeak_S_px": res.get("dpeak_S_px", float("nan")),
                "dpeak_D_px": res.get("dpeak_D_px", float("nan")),
                "dpeak_T_px": res.get("dpeak_T_px", float("nan")),
                "n_local_dates": len(local_dates),
                "k_before": cfg.k_before, "k_after": cfg.k_after,
                **q}

    # --------------------------------------------------------------- run loop
    def run_timeseries(self, n_jobs=-1, dates=None, out_csv=None):
        """Process every date in parallel -> per-date clearance DataFrame."""
        if not self._is_setup:
            self.setup()
        from joblib import Parallel, delayed
        dates = dates if dates is not None else self.all_dates

        def _one(tag):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    r = self.process_one_avg(tag)
                    r["date_tag"] = tag
                    r["sensing_time"] = self.time_by_tag.get(tag, pd.NaT)
                    r["ok"] = True
                    return r
                except Exception as e:
                    return {"date_tag": tag,
                            "sensing_time": self.time_by_tag.get(tag, pd.NaT),
                            "ok": False, "error": str(e)}

        rows = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one)(d) for d in dates)
        df = pd.DataFrame(rows)
        df["sensing_time"] = pd.to_datetime(df["sensing_time"], utc=True)
        if out_csv is not None:
            df.to_csv(out_csv, index=False)
        return df

    # ----------------------------------------------------------- filtering
    def filter_timeseries(self, df):
        """Apply the per-stripe contrast and H_mean MAD/abs outlier cuts.
        Returns the kept rows (reset index)."""
        cfg = self.cfg
        ok = (df.get("ok", True).fillna(False).astype(bool)
              if "ok" in df else np.ones(len(df), bool))
        df_ok = df[ok].copy()
        c_pass = df_ok["contrast_min"] >= cfg.contrast_min_threshold
        kept = df_ok[c_pass].copy()

        h = kept["H_mean_m"].values.astype(float)
        lo_abs, hi_abs = cfg.h_mean_bounds
        pass_abs = (h >= lo_abs) & (h <= hi_abs)
        if cfg.h_mean_mad_k is not None and np.isfinite(h[pass_abs]).sum() >= 5:
            med = float(np.nanmedian(h[pass_abs]))
            mad = float(np.nanmedian(np.abs(h[pass_abs] - med)))
            scale = max(1.4826 * mad, 1e-6)
            pass_mad = np.abs(h - med) / scale <= cfg.h_mean_mad_k
        else:
            pass_mad = np.ones_like(h, bool)
        kept = kept[pass_abs & pass_mad].copy().reset_index(drop=True)
        return kept

    # --------------------------------------------------- reference comparison
    def airgap_reference(self, kept, station=None, halfwin_hours=None):
        """Compare the kept BridgeSAR clearance against the NOAA on-bridge
        air-gap sensor. Returns a dict with the acquisition-matched reference
        points, a continuous smoothed reference curve, and fit statistics.

        Air gap is a direct clearance measurement, so pseudo-clearance is
        ``H_anchor + (air_gap - mean(air_gap))``.
        """
        cfg = self.cfg
        station = station or cfg.airgap_station
        if station is None:
            raise ValueError("no airgap_station configured for this bridge")
        halfwin = halfwin_hours if halfwin_hours is not None else cfg.wl_avg_halfwin_hours

        t0 = kept["sensing_time"].min()
        t1 = kept["sensing_time"].max()
        ag_df = _ref.fetch_airgap(station, t0, t1)
        ag_mean_full = float(np.nanmean(ag_df["v"]))
        H_anchor = float(kept["H_mean_m"].mean())

        ag_acq, _, _ = _ref.wl_acquisition_matched_mean(
            kept["date_tag"].tolist(), self.time_by_tag, self.all_dates,
            ag_df["t"], ag_df["v"], cfg.k_before, cfg.k_after, halfwin)
        pseudoH_acq = H_anchor + (ag_acq - ag_mean_full)

        ag_smooth = _ref.smooth_airgap_curve(ag_df, halfwin)
        ref_curve = pd.DataFrame({"t": ag_smooth["t"],
                                  "v": H_anchor + (ag_smooth["v"].values - ag_mean_full)})
        ref_points = pd.DataFrame({"sensing_time": kept["sensing_time"].values,
                                   "v": pseudoH_acq})
        R, rmse, bias, n = _ref.stats(kept["H_mean_m"].values, pseudoH_acq)
        return {"station": station, "ref_curve": ref_curve, "ref_points": ref_points,
                "pseudoH_acq": pseudoH_acq, "H_anchor": H_anchor,
                "R": R, "RMSE_m": rmse, "bias_m": bias, "N": n,
                "airgap_raw": ag_df}
