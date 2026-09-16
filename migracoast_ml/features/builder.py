"""The single feature code path shared by training and serving."""

import pandas as pd

from migracoast_ml.config import CELL_HALF, FEATURE_NAMES, RAW_DIR, SLR_SCENARIO
from migracoast_ml.features.grid import snap_to_cell


class FeatureBuilder:
    """Assembles the model's feature vector for any coordinate.

    Every request is snapped to its 0.1 degree cell centre first. Training
    rows and live queries therefore run identical arithmetic, which is what
    makes training and serving skew impossible by construction.
    """

    def __init__(self, elevation, coast, sealevel, hazards):
        self._elevation = elevation
        self._coast = coast
        self._sealevel = sealevel
        self._hazards = hazards

    @classmethod
    def load(cls, raw_dir=None) -> "FeatureBuilder":
        """Build from the files written by the fetch stage."""
        from migracoast_ml.features.coast import CoastlineIndex
        from migracoast_ml.features.hazards import HazardIndex
        from migracoast_ml.features.sealevel import SeaLevelIndex
        from migracoast_ml.features.terrain import ElevationGrid
        from migracoast_ml.sources import boundaries, sealevel as slr_source
        from migracoast_ml.sources.storms import classify_events
        from migracoast_ml.config import STORM_YEARS

        raw_dir = raw_dir or RAW_DIR

        required = {
            "elevation": raw_dir / "elevation",
            "coastline": raw_dir / "ne_10m_coastline.zip",
            "sea level": raw_dir / "slr_projections.csv",
            "storm events": raw_dir / "storm_events.csv",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing source data for: {', '.join(missing)}. "
                "Run `python -m migracoast_ml fetch` first."
            )

        elevation = ElevationGrid.from_dir(raw_dir / "elevation")
        coast = CoastlineIndex(boundaries.load_coastline(raw_dir / "ne_10m_coastline.zip"))
        projections = slr_source.parse_projections(
            raw_dir / "slr_projections.csv", scenario=SLR_SCENARIO
        )
        events = classify_events(pd.read_csv(raw_dir / "storm_events.csv", low_memory=False))
        hazards = HazardIndex(events, n_years=len(STORM_YEARS))
        return cls(elevation, coast, SeaLevelIndex(projections), hazards)

    def build(self, lat: float, lon: float) -> dict:
        """Return the feature vector for the cell containing this point."""
        cell_lat, cell_lon = snap_to_cell(lat, lon)

        features = {}
        features.update(self._elevation.features(cell_lat, cell_lon, CELL_HALF))
        features["coast_km"] = self._coast.distance_km(cell_lat, cell_lon)
        features.update(self._sealevel.features(cell_lat, cell_lon))
        features.update(self._hazards.features(cell_lat, cell_lon))

        return {name: features[name] for name in FEATURE_NAMES}

    def build_frame(self, points) -> pd.DataFrame:
        """Vectorised build over an iterable of (lat, lon) pairs."""
        rows = [self.build(lat, lon) for lat, lon in points]
        return pd.DataFrame(rows, columns=list(FEATURE_NAMES))
