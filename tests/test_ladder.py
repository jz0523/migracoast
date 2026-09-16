import numpy as np
from sklearn.pipeline import Pipeline

from migracoast_ml.models.ladder import build_models


def test_ladder_has_three_rungs():
    models = build_models()
    assert set(models) == {"ridge", "gradient_boosting", "neural_net"}


def test_every_rung_is_a_pipeline():
    for model in build_models().values():
        assert isinstance(model, Pipeline)


def test_every_rung_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 11))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=80)
    for name, model in build_models().items():
        model.fit(X, y)
        assert model.predict(X[:5]).shape == (5,), name


def test_pipelines_tolerate_missing_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 11))
    X[0, 0] = np.nan
    y = rng.normal(size=60)
    for name, model in build_models().items():
        model.fit(X, y)  # must not raise
