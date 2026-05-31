"""Console entry points: ``bridgesar-stream``, ``bridgesar-osm``,
``bridgesar-timeseries``."""

from __future__ import annotations

import argparse

from .config import BridgeConfig


def _load_cfg(path, project_dir=None):
    cfg = BridgeConfig.from_yaml(path)
    if project_dir is not None:
        cfg.project_dir = project_dir
    return cfg.resolve_paths()


def _add_project_dir(p):
    p.add_argument("--project-dir", default=None,
                   help="Root data folder; amplitudes/zarr/osm/output paths are "
                        "derived from it as {Bridge}/{TRACK}/... unless the YAML "
                        "sets explicit paths.")


# --------------------------------------------------------------------- stream
def stream_main(argv=None):
    p = argparse.ArgumentParser(
        prog="bridgesar-stream",
        description="Stream OPERA CSLC-S1 amplitude into an amplitude-only zarr "
                    "and export per-date GeoTIFFs. Needs ~/.netrc for Earthdata.")
    p.add_argument("--config", required=True, help="Bridge config YAML")
    p.add_argument("--date-start", required=True, help="YYYY-MM-DD")
    p.add_argument("--date-end", required=True, help="YYYY-MM-DD")
    p.add_argument("--pol", default="VV")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--zarr-path", default=None)
    p.add_argument("--skip-months", type=int, nargs="+", default=None)
    p.add_argument("--export-geotiffs", action="store_true",
                   help="Also export per-date amplitude GeoTIFFs (optional; the "
                        "pipeline reads the zarr directly, so this is only needed "
                        "for GIS inspection).")
    _add_project_dir(p)
    args = p.parse_args(argv)

    from .stream import stream_amplitudes

    cfg = _load_cfg(args.config, args.project_dir)
    zarr_path = stream_amplitudes(cfg, args.date_start, args.date_end, pol=args.pol,
                                  n_workers=args.n_workers, zarr_path=args.zarr_path,
                                  skip_months=args.skip_months)
    print(f"Streamed amplitudes to {zarr_path}")
    if args.export_geotiffs:
        from .amplitude import export_amplitudes_from_zarr
        paths = export_amplitudes_from_zarr(zarr_path, cfg.amp_dir)
        print(f"Exported {len(paths)} amplitude GeoTIFFs to {cfg.amp_dir}")


# ------------------------------------------------------------------------ osm
def osm_main(argv=None):
    p = argparse.ArgumentParser(
        prog="bridgesar-osm",
        description="Fetch the OSM bridge centerline GeoJSON for a config.")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    _add_project_dir(p)
    args = p.parse_args(argv)

    from .osm import fetch_from_config
    cfg = _load_cfg(args.config, args.project_dir)
    gdf = fetch_from_config(cfg, force=args.force)
    print(gdf[["osm_id", "name", "ref", "bridge", "highway"]].to_string(index=False))


# ----------------------------------------------------------------- timeseries
def timeseries_main(argv=None):
    p = argparse.ArgumentParser(
        prog="bridgesar-timeseries",
        description="Run the BridgeSAR clearance time-series for a config.")
    p.add_argument("--config", required=True)
    p.add_argument("--out-csv", default=None, help="Where to write the per-date CSV.")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--reference", choices=["airgap", "none"], default="none",
                   help="Compare the filtered series against a NOAA reference.")
    _add_project_dir(p)
    args = p.parse_args(argv)

    from .clearance import ClearancePipeline
    cfg = _load_cfg(args.config, args.project_dir)
    pipe = ClearancePipeline(cfg).setup()
    df = pipe.run_timeseries(n_jobs=args.n_jobs, out_csv=args.out_csv)
    kept = pipe.filter_timeseries(df)
    print(f"Processed {len(df)} dates; kept {len(kept)} after filtering.")
    if args.reference == "airgap":
        ref = pipe.airgap_reference(kept)
        print(f"vs NOAA air gap {ref['station']}: "
              f"R={ref['R']:+.3f}  RMSE={ref['RMSE_m']:.2f} m  "
              f"bias={ref['bias_m']:+.2f} m  (N={ref['N']})")


if __name__ == "__main__":  # pragma: no cover
    import sys
    {"stream": stream_main, "osm": osm_main,
     "timeseries": timeseries_main}[sys.argv[1]](sys.argv[2:])
