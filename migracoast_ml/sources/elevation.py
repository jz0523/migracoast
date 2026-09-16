"""ETOPO 2022 elevation, fetched as per-state subsets.

The global file is 1.6 GB. NOAA's THREDDS NetCDF Subset Service returns a
clipped box instead, a few megabytes per state.
"""

from pathlib import Path
from urllib.parse import urlencode

import requests

from migracoast_ml.config import HTTP_HEADERS, RAW_DIR, ensure_dirs
from migracoast_ml.sources._io import atomic_write_bytes

# Only the 30 arc-second dataset works; the 60 arc-second one returns HTTP 500.
ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/ncss/grid/global/ETOPO2022/30s/"
    "30s_surface_elev_netcdf/ETOPO_2022_v1_30s_N90W180_surface.nc"
)

# Padding so that cells on a state edge still get a complete window.
BOUNDS_PAD = 0.2


def pad_bounds(bounds, pad: float = BOUNDS_PAD) -> tuple[float, float, float, float]:
    """Expand (min_lon, min_lat, max_lon, max_lat) into (south, north, west, east)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return (
        round(min_lat - pad, 6),
        round(max_lat + pad, 6),
        round(min_lon - pad, 6),
        round(max_lon + pad, 6),
    )


def build_url(south: float, north: float, west: float, east: float) -> str:
    """Compose a NetCDF Subset Service request for one bounding box."""
    query = urlencode(
        {
            "var": "z",
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            # accept=netcdf4 is rejected with HTTP 400 by this service.
            "accept": "netcdf",
        }
    )
    return f"{ETOPO_URL}?{query}"


def fetch_elevation(states_gdf, force: bool = False) -> list[Path]:
    """Download one elevation subset per state. Returns the written paths."""
    ensure_dirs()
    out_dir = RAW_DIR / "elevation"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for _, row in states_gdf.iterrows():
        destination = out_dir / f"etopo_{row.postal}.nc"
        if destination.exists() and not force:
            written.append(destination)
            continue
        south, north, west, east = pad_bounds(row.geometry.bounds)
        response = requests.get(
            build_url(south, north, west, east), timeout=600, headers=HTTP_HEADERS
        )
        response.raise_for_status()
        atomic_write_bytes(destination, response.content)
        written.append(destination)
    return written
