"""Natural Earth state polygons and coastline geometry."""

from pathlib import Path

import geopandas as gpd
import requests

from migracoast_ml.config import RAW_DIR, STATES, ensure_dirs
from migracoast_ml.sources._io import atomic_write_bytes

STATES_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
COAST_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"


def _download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    ensure_dirs()
    response = requests.get(url, timeout=180, allow_redirects=True)
    response.raise_for_status()
    return atomic_write_bytes(destination, response.content)


def fetch_boundaries(force: bool = False) -> tuple[Path, Path]:
    """Download the state and coastline archives. Returns their paths."""
    states = _download(STATES_URL, RAW_DIR / "ne_10m_admin_1_states_provinces.zip", force)
    coast = _download(COAST_URL, RAW_DIR / "ne_10m_coastline.zip", force)
    return states, coast


def filter_states(gdf, required=STATES):
    """Keep only the US states named in `required`, and verify all are present."""
    us = gdf[gdf["admin"] == "United States of America"]
    out = us[us["postal"].isin(required)].copy()
    missing = set(required) - set(out["postal"])
    if missing:
        raise ValueError(f"State boundaries missing for: {sorted(missing)}")
    return out.reset_index(drop=True)


def clip_coastline(gdf, bounds):
    """Keep coastline geometries intersecting (min_lon, min_lat, max_lon, max_lat)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return gdf.cx[min_lon:max_lon, min_lat:max_lat].reset_index(drop=True)


def load_states(path: Path | None = None):
    path = path or (RAW_DIR / "ne_10m_admin_1_states_provinces.zip")
    return filter_states(gpd.read_file(f"zip://{path}"))


def load_coastline(path: Path | None = None, bounds=None):
    path = path or (RAW_DIR / "ne_10m_coastline.zip")
    gdf = gpd.read_file(f"zip://{path}")
    return clip_coastline(gdf, bounds) if bounds else gdf
