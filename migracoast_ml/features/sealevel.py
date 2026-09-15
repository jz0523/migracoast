"""Nearest-site lookup of projected sea level rise."""

import numpy as np
import pandas as pd


class SeaLevelIndex:
    """Nearest-neighbour lookup over NOAA sea level projection sites."""

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Sea level projection table is empty")
        self._lat = df["lat"].to_numpy(dtype=float)
        self._lon = df["lon"].to_numpy(dtype=float)
        self._slr_2050 = df["slr_2050_ft"].to_numpy(dtype=float)
        self._slr_2100 = df["slr_2100_ft"].to_numpy(dtype=float)

    def _nearest(self, lat: float, lon: float) -> int:
        # Equirectangular approximation is sufficient for picking a nearest
        # site; the candidate sites are far apart relative to the error.
        dlat = self._lat - lat
        dlon = (self._lon - lon) * np.cos(np.radians(lat))
        return int(np.argmin(dlat * dlat + dlon * dlon))

    def features(self, lat: float, lon: float) -> dict:
        i = self._nearest(lat, lon)
        return {
            "slr_2050_ft": float(self._slr_2050[i]),
            "slr_2100_ft": float(self._slr_2100[i]),
        }
