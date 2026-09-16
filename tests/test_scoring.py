import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from migracoast_ml.config import FEATURE_NAMES
from migracoast_ml.models.scoring import RiskScorer


class Passthrough:
    """Predicts the first feature, so expected scores are easy to reason about."""

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(float)


@pytest.fixture
def scorer():
    return RiskScorer(
        model=Passthrough(),
        reference=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        feature_names=FEATURE_NAMES,
        metadata={"winner": "test"},
    )


def _features(first_value):
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats[FEATURE_NAMES[0]] = first_value
    return feats


def test_score_returns_the_three_public_numbers(scorer):
    out = scorer.score_features(_features(2.0))
    assert set(out) >= {"predicted_claim_intensity", "risk_percentile", "safety_score"}


def test_percentile_is_zero_below_the_reference_range(scorer):
    assert scorer.score_features(_features(-10.0))["risk_percentile"] == pytest.approx(0.0)


def test_percentile_is_one_hundred_above_the_reference_range(scorer):
    assert scorer.score_features(_features(99.0))["risk_percentile"] == pytest.approx(100.0)


def test_percentile_is_in_the_middle_for_a_median_value(scorer):
    assert scorer.score_features(_features(2.0))["risk_percentile"] == pytest.approx(50.0, abs=15)


def test_safety_is_the_complement_of_risk(scorer):
    out = scorer.score_features(_features(1.0))
    assert out["safety_score"] == pytest.approx(100.0 - out["risk_percentile"])


def test_feature_order_is_enforced_not_dict_order(scorer):
    shuffled = dict(reversed(list(_features(3.0).items())))
    assert scorer.score_features(shuffled) == scorer.score_features(_features(3.0))


def test_missing_feature_raises(scorer):
    feats = _features(1.0)
    del feats[FEATURE_NAMES[-1]]
    with pytest.raises(ValueError, match="missing"):
        scorer.score_features(feats)


def test_round_trips_through_disk(tmp_path):
    model = Pipeline([("m", DummyRegressor(strategy="constant", constant=1.0))])
    model.fit(np.zeros((4, len(FEATURE_NAMES))), np.zeros(4))
    original = RiskScorer(model, np.array([0.0, 1.0, 2.0]), FEATURE_NAMES, {"winner": "dummy"})

    path = tmp_path / "model.joblib"
    original.save(path)
    restored = RiskScorer.load(path)

    assert restored.metadata["winner"] == "dummy"
    assert restored.feature_names == tuple(FEATURE_NAMES)
