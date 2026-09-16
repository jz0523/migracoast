import numpy as np
import pandas as pd
import pytest

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.train import select_winner, train_from_table


@pytest.fixture
def table():
    rng = np.random.default_rng(3)
    n = 300
    data = {name: rng.normal(size=n) for name in FEATURE_NAMES}
    frame = pd.DataFrame(data)
    frame["target"] = frame["elev_mean"] * -1.2 + rng.normal(scale=0.2, size=n)
    frame["block"] = [f"b{i % 12}" for i in range(n)]
    frame["coast_km"] = np.abs(frame["coast_km"]) * 10.0
    frame["cell_lat"] = 42.0 + rng.normal(scale=0.5, size=n)
    frame["cell_lon"] = -71.0 + rng.normal(scale=0.5, size=n)
    frame["claim_count"] = rng.integers(0, 5, size=n)
    frame["state"] = "MA"
    return frame


def test_winner_is_chosen_on_coastal_rmse():
    metrics = pd.DataFrame(
        {
            "model": ["a", "b"],
            "rmse_mean": [0.1, 0.9],          # 'a' wins overall
            "coastal_rmse_mean": [0.8, 0.2],  # 'b' wins where it matters
        }
    )
    assert select_winner(metrics) == "b"


def test_winner_ignores_nan_rows():
    metrics = pd.DataFrame(
        {"model": ["a", "b"], "rmse_mean": [0.5, 0.5], "coastal_rmse_mean": [np.nan, 0.4]}
    )
    assert select_winner(metrics) == "b"


def test_training_returns_a_scorer_and_metrics(table, tmp_path):
    result = train_from_table(table, out_dir=tmp_path)
    assert result["winner"] in {"ridge", "gradient_boosting", "neural_net"}
    assert len(result["metrics"]) == 3


def test_training_writes_the_expected_artifacts(table, tmp_path):
    train_from_table(table, out_dir=tmp_path)
    assert (tmp_path / "risk_model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_saved_scorer_predicts_on_a_feature_dict(table, tmp_path):
    from migracoast_ml.models.scoring import RiskScorer

    train_from_table(table, out_dir=tmp_path)
    scorer = RiskScorer.load(tmp_path / "risk_model.joblib")
    out = scorer.score_features({name: 0.0 for name in FEATURE_NAMES})
    assert 0.0 <= out["risk_percentile"] <= 100.0


def test_reference_distribution_comes_from_coastal_cells(table, tmp_path):
    from migracoast_ml.models.scoring import RiskScorer

    table = table.copy()
    # Exactly 100 of the 300 cells are coastal (under 30 km).
    table["coast_km"] = [5.0 if i % 3 == 0 else 100.0 for i in range(len(table))]
    train_from_table(table, out_dir=tmp_path)
    scorer = RiskScorer.load(tmp_path / "risk_model.joblib")
    assert scorer.reference.size == 100
