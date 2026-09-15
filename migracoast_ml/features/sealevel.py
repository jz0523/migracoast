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
        # True great-circle distance. An equirectangular shortcut picked the
        # wrong site for about 2% of real grid cells.
        phi1 = np.radians(lat)
        phi2 = np.radians(self._lat)
        dphi = phi2 - phi1
        dlambda = np.radians(self._lon - lon)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return int(np.argmin(a))

    def features(self, lat: float, lon: float) -> dict:
        i = self._nearest(lat, lon)
        return {
            "slr_2050_ft": float(self._slr_2050[i]),
            "slr_2100_ft": float(self._slr_2100[i]),
        }
