"""Grid cell arithmetic shared by training and serving."""

import math

import pandas as pd

from migracoast_ml.config import BLOCK_DEG, CELL_SIZE


def snap_to_cell(lat: float, lon: float) -> tuple[float, float]:
    """Return the centre of the 0.1 degree cell containing this point.

    Rounding to one decimal place mirrors how FEMA redacts claim
    coordinates, so a claim and the cell it belongs to always agree.
    """
    return (round(round(lat / CELL_SIZE) * CELL_SIZE, 1),
            round(round(lon / CELL_SIZE) * CELL_SIZE, 1))


def block_id(lat: float, lon: float, deg: float = BLOCK_DEG) -> str:
    """Return the identifier of the cross-validation block holding this point."""
    return f"{math.floor(lat / deg)}_{math.floor(lon / deg)}"


def build_cell_universe(states_gdf) -> pd.DataFrame:
    """Enumerate every 0.1 degree cell whose centre lies inside a state.

    Returns a frame with cell_lat, cell_lon and state columns.
    """
    from shapely.geometry import Point

    rows = []
    for _, state_row in states_gdf.iterrows():
        geom = state_row.geometry
        min_lon, min_lat, max_lon, max_lat = geom.bounds
        lat = round(math.floor(min_lat / CELL_SIZE) * CELL_SIZE, 1)
        while lat <= max_lat + CELL_SIZE:
            lon = round(math.floor(min_lon / CELL_SIZE) * CELL_SIZE, 1)
            while lon <= max_lon + CELL_SIZE:
                if geom.contains(Point(lon, lat)):
                    rows.append((lat, lon, state_row.postal))
                lon = round(lon + CELL_SIZE, 1)
            lat = round(lat + CELL_SIZE, 1)

    frame = pd.DataFrame(rows, columns=["cell_lat", "cell_lon", "state"])
    return frame.drop_duplicates(subset=["cell_lat", "cell_lon"]).reset_index(drop=True)
