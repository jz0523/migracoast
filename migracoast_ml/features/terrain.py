"""Terrain statistics from ETOPO elevation subsets."""

from pathlib import Path

import numpy as np

from migracoast_ml.config import CELL_HALF

NAN = float("nan")


def terrain_features(window) -> dict:
    """Summarise an elevation window. Land is any pixel above zero metres."""
    values = np.ma.masked_invalid(np.ma.asarray(window)).compressed()
    values = values[values > -30000]  # drop ETOPO fill values

    if values.size == 0:
        return {
            "elev_mean": NAN,
            "elev_min": NAN,
            "elev_p10": NAN,
            "frac_below_2m": NAN,
            "frac_below_5m": NAN,
            "land_fraction": NAN,
        }

    land = values[values > 0]
    land_fraction = float(land.size) / float(values.size)

    if land.size == 0:
        return {
            "elev_mean": NAN,
            "elev_min": NAN,
            "elev_p10": NAN,
            "frac_below_2m": NAN,
            "frac_below_5m": NAN,
            "land_fraction": land_fraction,
        }

    return {
        "elev_mean": float(np.mean(land)),
        "elev_min": float(np.min(land)),
        "elev_p10": float(np.percentile(land, 10)),
        "frac_below_2m": float(np.mean(land < 2.0)),
        "frac_below_5m": float(np.mean(land < 5.0)),
        "land_fraction": land_fraction,
    }


class ElevationGrid:
    """One or more ETOPO boxes, queried by geographic window."""

    def __init__(self, boxes):
        # Each box is (lat_array, lon_array, z_array).
        self._boxes = boxes

    @classmethod
    def from_dir(cls, directory) -> "ElevationGrid":
        import netCDF4

        boxes = []
        for path in sorted(Path(directory).glob("*.nc")):
            with netCDF4.Dataset(path) as ds:
                boxes.append(
                    (
                        np.array(ds.variables["lat"][:], dtype=float),
                        np.array(ds.variables["lon"][:], dtype=float),
                        np.ma.asarray(ds.variables["z"][:]),
                    )
                )
        if not boxes:
            raise ValueError(f"No elevation NetCDF files found in {directory}")
        return cls(boxes)

    def window(self, lat: float, lon: float, half: float = CELL_HALF) -> np.ndarray:
        """Return each distinct pixel within `half` degrees of the point.

        Padded state boxes overlap and all sample one global grid, so a pixel
        can appear in several boxes with identical coordinates. Pixels are
        deduplicated by coordinate so that overlap never over-weights part of
        a window. Masked pixels are returned as NaN.
        """
        keys = []
        values = []
        for lats, lons, z in self._boxes:
            rows = np.where((lats >= lat - half) & (lats <= lat + half))[0]
            cols = np.where((lons >= lon - half) & (lons <= lon + half))[0]
            if not (rows.size and cols.size):
                continue
            lat_grid, lon_grid = np.meshgrid(lats[rows], lons[cols], indexing="ij")
            keys.append(
                np.column_stack([np.round(lat_grid.ravel(), 6), np.round(lon_grid.ravel(), 6)])
            )
            block = np.ma.asarray(z[np.ix_(rows, cols)]).astype(float)
            values.append(np.ma.filled(block, np.nan).ravel())
        if not keys:
            return np.array([])
        keys = np.vstack(keys)
        values = np.concatenate(values)
        _, first = np.unique(keys, axis=0, return_index=True)
        return values[np.sort(first)]

    def features(self, lat: float, lon: float, half: float = CELL_HALF) -> dict:
        return terrain_features(self.window(lat, lon, half))
