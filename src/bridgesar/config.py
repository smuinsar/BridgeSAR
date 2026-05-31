"""Configuration objects for BridgeSAR.

Two dataclasses drive every run:

* :class:`HyperParams` holds the stripe-extraction tuning constants. They are
  shared across bridges; only a few are ever overridden per site (most commonly
  ``nominal_stripe_width_m`` and ``rot_delta_bound_deg``).
* :class:`BridgeConfig` holds everything that varies per bridge / track: the
  streaming ROI, deck geometry, temporal-averaging window, NOAA reference
  stations and the OpenStreetMap query.

Both load from / save to YAML so the public can apply BridgeSAR to new bridges
by writing a single config file. The shipped ``configs/*.yaml`` cover five
example bridges.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Stripe-extraction hyperparameters (cell 2 of the reference notebooks)
# ---------------------------------------------------------------------------
@dataclass
class HyperParams:
    """Tuning constants for the Method-E++ joint 3-peak stripe fit.

    Defaults reproduce the reference runs. Field names mirror the original
    notebook globals (lower-cased) so the algorithm is a faithful port.
    """

    sat_look_dir: str = "right"          # Sentinel-1 looks right
    length_buffer_frac: float = -0.1
    nominal_stripe_width_m: float = 25.0
    rot_delta_bound_deg: float = 0.0     # tilt lock on OUTPUT (matches OSM)
    width_factor_bounds: tuple[float, float] = (0.6, 2.0)
    spacing_rel_bound: float = 0.30
    lambda_width_prior: float = 0.15
    lambda_spacing_prior: float = 0.25
    lambda_symmetry_prior: float = 0.10
    sigma_factor: float = 2.0
    profile_pad_px: int = 100
    profile_clip_z: float = 6.0
    profile_smooth_sigma: float = 1.0
    peak_pick_method: str = "derivative_only"   # or 'hybrid'
    stripe_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)
    contrast_target: float = 2.5
    softmin_tau: float = 1.0

    # SNR uncertainty
    snr_floor: float = 1.0

    @property
    def rot_delta_fit_bound_deg(self) -> float:
        """Wider search basin than the output tilt lock (>= 2 deg)."""
        return max(self.rot_delta_bound_deg, 2.0)


# ---------------------------------------------------------------------------
# Per-bridge / per-track configuration
# ---------------------------------------------------------------------------
@dataclass
class OSMQuery:
    """OpenStreetMap Overpass query for the bridge centerline."""

    names: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # S, W, N, E
    utm_epsg: int = 4326
    # regex used to select the right way(s) from the fetched geojson 'name' column
    name_regex: str | None = None


@dataclass
class WLStation:
    id: str
    name: str
    datum: str = "MLLW"


@dataclass
class BridgeConfig:
    """Everything that varies per bridge / track."""

    bridge_name: str
    track: str                            # e.g. 'P115'

    # streaming ROI (WGS84 degrees) — from run_zarr_*.csh
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    # deck geometry
    deck_height_m: float
    bridge_thickness_m: float
    scatter_model: str = "mixed"          # 'symmetric'|'lower_triple'|'lower_single'|'mixed'

    # local temporal-averaging window for the per-date D+T fit
    k_before: int = 1
    k_after: int = 1

    # outlier filtering
    contrast_min_threshold: float = 1.30
    h_mean_mad_k: float | None = 3.0
    h_mean_bounds: tuple[float, float] = (float("-inf"), float("inf"))
    mask_towers: bool = False
    tower_k_mad: float = 4.0
    tower_expand: int = 1

    # NOAA reference matching
    wl_avg_halfwin_hours: float = 3.0
    wl_stations: list[WLStation] = field(default_factory=list)
    wl_primary_id: str | None = None
    airgap_station: str | None = None     # on-bridge NOAA air-gap sensor id

    # OpenStreetMap geometry
    osm: OSMQuery = field(default_factory=OSMQuery)

    # per-bridge hyperparameter overrides (keys = HyperParams field names)
    hyperparams: dict[str, Any] = field(default_factory=dict)

    # ---- paths (optional; defaults derived from a project root) ----------
    project_dir: str | None = None        # root holding {bridge}/{track}/...
    amp_dir: str | None = None
    zarr_path: str | None = None
    osm_geojson: str | None = None
    out_dir: str | None = None

    # ----------------------------------------------------------------- IO --
    @classmethod
    def from_yaml(cls, path: str | Path) -> "BridgeConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def named(cls, name: str) -> "BridgeConfig":
        """Load a bundled config by name, e.g. ``BridgeConfig.named('baybridge_p115')``."""
        from importlib import resources
        ref = resources.files("bridgesar.configs") / f"{name}.yaml"
        with resources.as_file(ref) as p:
            return cls.from_yaml(p)

    @staticmethod
    def list_bundled() -> list[str]:
        """Names of the bundled bridge configs."""
        from importlib import resources
        return sorted(p.name[:-5] for p in resources.files("bridgesar.configs").iterdir()
                      if p.name.endswith(".yaml"))

    @classmethod
    def from_dict(cls, d: dict) -> "BridgeConfig":
        d = dict(d)
        d.pop("hyperparams_defaults", None)
        if "wl_stations" in d and d["wl_stations"]:
            d["wl_stations"] = [
                s if isinstance(s, WLStation) else WLStation(**s)
                for s in d["wl_stations"]
            ]
        if "osm" in d and d["osm"] is not None and not isinstance(d["osm"], OSMQuery):
            osm = dict(d["osm"])
            if "bbox" in osm and osm["bbox"] is not None:
                osm["bbox"] = tuple(osm["bbox"])
            d["osm"] = OSMQuery(**osm)
        if "h_mean_bounds" in d and d["h_mean_bounds"] is not None:
            d["h_mean_bounds"] = tuple(d["h_mean_bounds"])
        return cls(**d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    # ---- derived helpers -------------------------------------------------
    @property
    def track_number(self) -> str:
        """'P115' -> '115' (zero-padded form used in zarr filenames)."""
        return self.track.lstrip("Pp")

    def hyper(self) -> HyperParams:
        """HyperParams with this bridge's overrides applied."""
        hp = HyperParams()
        for k, v in (self.hyperparams or {}).items():
            if hasattr(hp, k):
                if k in ("width_factor_bounds", "stripe_weights") and v is not None:
                    v = tuple(v)
                setattr(hp, k, v)
        return hp

    def resolve_paths(self) -> "BridgeConfig":
        """Fill amp_dir / zarr_path / osm_geojson / out_dir from project_dir
        when they are not explicitly set. Mirrors the notebook directory
        layout: ``{project}/{Bridge}/{TRACK}/...``."""
        if self.project_dir is None:
            return self
        root = Path(self.project_dir)
        folder = self.bridge_name.replace(" ", "")
        track_dir = root / folder / self.track
        if self.amp_dir is None:
            self.amp_dir = str(track_dir / "Amplitudes")
        if self.zarr_path is None:
            self.zarr_path = str(track_dir / f"OPERA_CSLC_S1_T{self.track_number}.zarr")
        if self.osm_geojson is None:
            self.osm_geojson = str(root / folder / "Inventory" /
                                   f"osm_{folder.lower()}.geojson")
        if self.out_dir is None:
            self.out_dir = str(root / folder / "output")
        return self
