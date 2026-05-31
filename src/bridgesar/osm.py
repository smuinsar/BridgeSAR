"""Fetch bridge centerlines from OpenStreetMap via the Overpass API.

the bridge name(s), highway ref(s) and bounding box come from a :class:`~bridgesar.config.OSMQuery`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import requests
import geopandas as gpd
from shapely.geometry import LineString

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {"User-Agent": "BridgeSAR/0.1"}


def build_query(bbox: Sequence[float], names: Iterable[str],
                refs: Iterable[str], timeout: int) -> str:
    """Overpass QL query selecting bridge ways by name regex and/or highway ref."""
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    clauses = []
    for name in names:
        clauses.append(f'  way["bridge"]["name"~"{name}",i]({bbox_str});')
        clauses.append(f'  way["man_made"="bridge"]["name"~"{name}",i]({bbox_str});')
    for ref in refs:
        clauses.append(f'  way["bridge"]["ref"="{ref}"]({bbox_str});')
    if not clauses:
        clauses.append(f'  way["bridge"]({bbox_str});')
    body = "\n".join(clauses)
    return f"[out:json][timeout:{timeout}];\n(\n{body}\n);\nout geom;"


def fetch_bridge_osm(out, bbox, names=(), refs=(), *, timeout=180,
                     force=False) -> gpd.GeoDataFrame:
    """Fetch bridge ways from Overpass and write a GeoJSON of LineStrings.

    Returns the resulting GeoDataFrame (EPSG:4326). Cached unless ``force``.
    """
    out = Path(out)
    if out.exists() and not force:
        print(f"cache hit: {out}")
        return gpd.read_file(out)
    query = build_query(bbox, names, refs, timeout)
    r = requests.post(OVERPASS_URL, data={"data": query},
                      headers=OVERPASS_HEADERS, timeout=timeout)
    r.raise_for_status()
    feats = []
    for el in r.json().get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        feats.append({"geometry": LineString(coords),
                      "osm_id": el["id"],
                      "name": tags.get("name", ""),
                      "bridge": tags.get("bridge", ""),
                      "ref": tags.get("ref", ""),
                      "highway": tags.get("highway", "")})
    if not feats:
        raise RuntimeError("Overpass returned no ways — broaden the bbox or filters.")
    gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GeoJSON")
    print(f"wrote {out}  ({len(gdf)} ways)")
    return gdf


def fetch_from_config(cfg, *, force=False) -> gpd.GeoDataFrame:
    """Convenience: fetch using a :class:`BridgeConfig` (uses ``cfg.osm`` and
    ``cfg.osm_geojson``)."""
    cfg.resolve_paths()
    return fetch_bridge_osm(cfg.osm_geojson, bbox=cfg.osm.bbox,
                            names=cfg.osm.names, refs=cfg.osm.refs, force=force)
