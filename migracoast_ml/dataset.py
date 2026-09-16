"""Assemble the training table: one row per grid cell."""

import numpy as np
import pandas as pd

from migracoast_ml.config import (
    COASTAL_KM,
    FEATURE_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
)
from migracoast_ml.features.grid import block_id

TABLE_PATH = PROCESSED_DIR / "training_table.parquet"


def target_from_counts(counts) -> np.ndarray:
    """Model the log of one plus the claim count, which tames the long tail."""
    return np.log1p(np.asarray(counts, dtype=float))


def coastal_mask(frame: pd.DataFrame) -> pd.Series:
    """True for cells within the coastal threshold of the shoreline."""
    return frame["coast_km"] < COASTAL_KM


def assemble(cells: pd.DataFrame, counts: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join the cell universe, claim counts and features into one table."""
    out = cells.reset_index(drop=True).copy()
    for name in FEATURE_NAMES:
        out[name] = features[name].to_numpy()

    if len(counts):
        out = out.merge(counts, on=["cell_lat", "cell_lon"], how="left")
    else:
        out["claim_count"] = 0

    out["claim_count"] = out["claim_count"].fillna(0).astype(int)
    out["target"] = target_from_counts(out["claim_count"])
    out["block"] = [block_id(a, o) for a, o in zip(out["cell_lat"], out["cell_lon"])]
    return out


def build_training_table(raw_dir=None, save: bool = True) -> pd.DataFrame:
    """Run the full build: cell universe, features, claim counts, join."""
    from migracoast_ml.features.builder import FeatureBuilder
    from migracoast_ml.features.grid import build_cell_universe
    from migracoast_ml.sources import boundaries, fema

    raw_dir = raw_dir or RAW_DIR
    ensure_dirs()

    states = boundaries.load_states(raw_dir / "ne_10m_admin_1_states_provinces.zip")
    cells = build_cell_universe(states)

    builder = FeatureBuilder.load(raw_dir)
    features = builder.build_frame(zip(cells["cell_lat"], cells["cell_lon"]))

    claims = pd.read_csv(raw_dir / "nfip_claims.csv", low_memory=False)
    counts = fema.count_claims_per_cell(claims)

    table = assemble(cells, counts, features)
    if save:
        table.to_parquet(TABLE_PATH, index=False)
    return table


def load_training_table(path=None) -> pd.DataFrame:
    return pd.read_parquet(path or TABLE_PATH)
