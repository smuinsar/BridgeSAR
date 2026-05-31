"""Config round-trip tests across all shipped bridge YAMLs."""

import pytest

from bridgesar.config import BridgeConfig, HyperParams

CONFIG_NAMES = [
    "baybridge_p035", "baybridge_p042", "baybridge_p115",
    "goldengate_p035", "goldengate_p042", "goldengate_p115",
    "reedypoint_p106", "haleboggs_p165", "hueylong_p165",
]


def test_list_bundled_matches():
    assert sorted(BridgeConfig.list_bundled()) == sorted(CONFIG_NAMES)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_shipped_config_loads(name):
    cfg = BridgeConfig.named(name)
    assert cfg.bridge_name
    assert cfg.track.startswith("P")
    assert cfg.deck_height_m > 0
    assert cfg.lat_min < cfg.lat_max
    assert cfg.lon_min < cfg.lon_max
    assert len(cfg.wl_stations) >= 1
    assert cfg.osm.utm_epsg > 0
    # hyperparameter overrides apply cleanly
    hp = cfg.hyper()
    assert isinstance(hp, HyperParams)


def test_track_number():
    cfg = BridgeConfig.named("baybridge_p115")
    assert cfg.track_number == "115"


def test_resolve_paths_from_project_dir(tmp_path):
    cfg = BridgeConfig.named("baybridge_p115")
    cfg.project_dir = str(tmp_path)
    cfg.resolve_paths()
    assert cfg.amp_dir.endswith("BayBridge/P115/Amplitudes")
    assert cfg.zarr_path.endswith("OPERA_CSLC_S1_T115.zarr")
    assert cfg.osm_geojson.endswith("osm_baybridge.geojson")
