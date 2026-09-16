import numpy as np
import pandas as pd
import pytest

from migracoast_ml.models.evaluate import oof_predictions, spatial_cv


@pytest.fixture
def toy():
    rng = np.random.default_rng(7)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 11)), columns=[f"f{i}" for i in range(11)])
    y = pd.Series(X["f0"] * 1.5 + rng.normal(scale=0.2, size=n))
    groups = pd.Series([f"b{i % 10}" for i in range(n)])
    coastal = pd.Series([i % 2 == 0 for i in range(n)])
    return X, y, groups, coastal


def test_returns_one_row_per_model(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    models = {
        "a": Pipeline([("m", Ridge())]),
        "b": Pipeline([("m", Ridge(alpha=10.0))]),
    }
    table = spatial_cv(*toy, models=models, n_splits=3)
    assert len(table) == 2
    assert set(table["model"]) == {"a", "b"}


def test_reports_overall_and_coastal_metrics(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    table = spatial_cv(*toy, models={"a": Pipeline([("m", Ridge())])}, n_splits=3)
    for column in (
        "rmse_mean", "rmse_std", "r2_mean", "spearman_mean",
        "coastal_rmse_mean", "coastal_rmse_std", "coastal_spearman_mean",
    ):
        assert column in table.columns


def test_no_block_appears_in_both_train_and_test(toy, monkeypatch):
    """The guarantee that makes the score honest, checked on our own code path."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    import migracoast_ml.models.evaluate as evaluate

    X, y, groups, _ = toy
    seen = []

    class SpyGroupKFold(evaluate.GroupKFold):
        def split(self, X, y=None, groups=None):
            for train_idx, test_idx in super().split(X, y, groups):
                seen.append((train_idx, test_idx))
                yield train_idx, test_idx

    monkeypatch.setattr(evaluate, "GroupKFold", SpyGroupKFold)
    evaluate.oof_predictions(X, y, groups, Pipeline([("m", Ridge())]), n_splits=3)

    assert len(seen) == 3
    for train_idx, test_idx in seen:
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


def test_oof_predictions_cover_every_row_exactly_once(toy):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    X, y, groups, _ = toy
    preds = oof_predictions(X, y, groups, Pipeline([("m", Ridge())]), n_splits=3)
    assert preds.shape == (len(X),)
    assert not np.isnan(preds).any()


def test_a_useful_model_beats_a_constant_one(toy):
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    table = spatial_cv(
        *toy,
        models={
            "real": Pipeline([("m", Ridge())]),
            "constant": Pipeline([("m", DummyRegressor(strategy="mean"))]),
        },
        n_splits=3,
    )
    real = table[table.model == "real"].rmse_mean.iloc[0]
    constant = table[table.model == "constant"].rmse_mean.iloc[0]
    assert real < constant
