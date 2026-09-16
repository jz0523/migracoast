"""Fit the ladder, choose a winner, and serialise it for serving."""

import json
from datetime import date

import numpy as np
import pandas as pd

from migracoast_ml.config import (
    FEATURE_NAMES,
    MODELS_DIR,
    SLR_SCENARIO,
    STATES,
    ensure_dirs,
)
from migracoast_ml.dataset import coastal_mask, load_training_table
from migracoast_ml.models.evaluate import spatial_cv
from migracoast_ml.models.ladder import build_models
from migracoast_ml.models.scoring import RiskScorer

MODEL_PATH = MODELS_DIR / "risk_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def select_winner(metrics: pd.DataFrame) -> str:
    """Lowest coastal RMSE wins, because coastal cells are the use case."""
    usable = metrics.dropna(subset=["coastal_rmse_mean"])
    if usable.empty:
        usable = metrics.dropna(subset=["rmse_mean"])
        return str(usable.loc[usable["rmse_mean"].idxmin(), "model"])
    return str(usable.loc[usable["coastal_rmse_mean"].idxmin(), "model"])


def train_from_table(table: pd.DataFrame, out_dir=None) -> dict:
    """Cross-validate every model, refit the winner, and write the artifacts."""
    out_dir = out_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    X = table[list(FEATURE_NAMES)]
    y = table["target"]
    groups = table["block"]
    coastal = coastal_mask(table)

    models = build_models()
    metrics = spatial_cv(X, y, groups, coastal, models)
    winner = select_winner(metrics)

    fitted = models[winner].fit(X, y)

    # Percentiles are ranked against coastal cells only; ranking against
    # inland zeros would push every coastal answer to the top of the scale.
    reference_rows = X[coastal.to_numpy()]
    reference = fitted.predict(reference_rows if len(reference_rows) else X)

    metadata = {
        "winner": winner,
        "trained_on": date.today().isoformat(),
        "n_cells": int(len(table)),
        "n_coastal_cells": int(coastal.sum()),
        "states": list(STATES),
        "scenario": SLR_SCENARIO,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics.to_dict(orient="records"),
    }

    RiskScorer(fitted, reference, FEATURE_NAMES, metadata).save(out_dir / "risk_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))

    return {"winner": winner, "metrics": metrics, "metadata": metadata}


def train_and_save() -> dict:
    ensure_dirs()
    return train_from_table(load_training_table())


def print_metrics() -> None:
    """Print the saved cross-validation table."""
    metadata = json.loads(METRICS_PATH.read_text())
    print(f"Winner: {metadata['winner']}  (trained {metadata['trained_on']})")
    print(f"Cells: {metadata['n_cells']}, coastal: {metadata['n_coastal_cells']}")
    print(pd.DataFrame(metadata["metrics"]).to_string(index=False))
