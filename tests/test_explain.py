import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.explain import permutation_importance_table, top_contributors


@pytest.fixture
def fitted():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(150, 11)), columns=list(FEATURE_NAMES))
    # Only elev_mean carries signal.
    y = pd.Series(X["elev_mean"] * 3.0 + rng.normal(scale=0.1, size=150))
    model = Pipeline([("m", Ridge())]).fit(X, y)
    return model, X, y


def test_importance_table_covers_every_feature(fitted):
    table = permutation_importance_table(*fitted)
    assert len(table) == len(FEATURE_NAMES)
    assert set(table["feature"]) == set(FEATURE_NAMES)


def test_importance_is_sorted_with_the_strongest_first(fitted):
    table = permutation_importance_table(*fitted)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["feature"].iloc[0] == "elev_mean"


def test_top_contributors_returns_the_requested_count(fitted):
    model, X, _ = fitted
    out = top_contributors(model, X.iloc[0].to_dict(), FEATURE_NAMES, k=3)
    assert len(out) == 3


def test_contributor_entries_carry_a_name_and_value(fitted):
    model, X, _ = fitted
    entry = top_contributors(model, X.iloc[0].to_dict(), FEATURE_NAMES, k=1)[0]
    assert "feature" in entry
    assert "value" in entry


def test_shap_absence_is_reported_not_raised(fitted, monkeypatch):
    import migracoast_ml.models.explain as explain

    monkeypatch.setattr(explain, "_import_shap", lambda: None)
    model, X, _ = fitted
    assert explain.shap_contributions(model, X.iloc[0].to_dict(), FEATURE_NAMES) is None
