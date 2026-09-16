"""Historical flood and storm event rates per grid cell."""

import pandas as pd

from migracoast_ml.features.grid import snap_to_cell


class HazardIndex:
    """Event counts binned to 0.1 degree cells, exposed as annual rates."""

    def __init__(self, events: pd.DataFrame, n_years: int):
        if not n_years:
            raise ValueError("n_years must be a positive number of years")
        self._n_years = float(n_years)
        self._counts: dict[tuple[float, float], dict[str, int]] = {}

        if events.empty:
            return

        snapped = [snap_to_cell(a, o) for a, o in zip(events["lat"], events["lon"])]
        binned = pd.DataFrame(
            {
                "cell_lat": [s[0] for s in snapped],
                "cell_lon": [s[1] for s in snapped],
                "kind": list(events["kind"]),
            }
        )
        grouped = binned.groupby(["cell_lat", "cell_lon", "kind"]).size()
        for (cell_lat, cell_lon, kind), count in grouped.items():
            self._counts.setdefault((cell_lat, cell_lon), {})[kind] = int(count)

    def features(self, lat: float, lon: float) -> dict:
        """Annual flood and storm event rates for the cell containing this point."""
        cell = snap_to_cell(lat, lon)
        counts = self._counts.get(cell, {})
        return {
            "flood_events_per_year": counts.get("flood", 0) / self._n_years,
            "storm_events_per_year": counts.get("storm", 0) / self._n_years,
        }
