import numpy as np
import pytest
from flask import Flask

from migracoast_ml.api import MAX_GRID_CELLS, create_blueprint
from migracoast_ml.config import FEATURE_NAMES


class StubBuilder:
    def build(self, lat, lon):
        return {
            name: (float("nan") if name == "elev_mean" else 1.0)
            for name in FEATURE_NAMES
        }


class StubScorer:
    feature_names = tuple(FEATURE_NAMES)
    metadata = {"winner": "stub", "trained_on": "2026-09-15", "metrics": []}
    model = None
    reference = np.array([0.0, 1.0, 2.0])

    def score_features(self, features):
        return {
            "predicted_claim_intensity": 1.0,
            "risk_percentile": 50.0,
            "safety_score": 50.0,
        }


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(
        create_blueprint(StubScorer(), StubBuilder(), covered_states=("MA", "FL"))
    )
    return app.test_client()


def test_risk_returns_the_score_payload(client):
    response = client.post("/ml/risk", json={"latitude": 42.36, "longitude": -71.06})
    assert response.status_code == 200
    body = response.get_json()
    assert body["safety_score"] == 50.0
    assert body["risk_percentile"] == 50.0
    assert "features" in body
    assert "top_contributors" in body


def test_risk_rejects_missing_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": 42.36})
    assert response.status_code == 400
    assert "longitude" in response.get_json()["error"].lower()


def test_risk_rejects_non_numeric_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": "north", "longitude": -71.0})
    assert response.status_code == 400


def test_risk_rejects_out_of_range_coordinates(client):
    response = client.post("/ml/risk", json={"latitude": 999.0, "longitude": -71.0})
    assert response.status_code == 400


def test_heatmap_returns_one_entry_per_cell(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.4, "south": 42.2, "east": -71.0, "west": -71.2, "resolution": 0.1},
    )
    assert response.status_code == 200
    cells = response.get_json()["cells"]
    assert len(cells) >= 1
    assert {"lat", "lon", "risk_percentile", "safety_score"} <= set(cells[0])


def test_heatmap_rejects_an_oversized_grid(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 49.0, "south": 25.0, "east": -67.0, "west": -125.0, "resolution": 0.1},
    )
    assert response.status_code == 400
    assert str(MAX_GRID_CELLS) in response.get_json()["error"]


def test_heatmap_rejects_an_inverted_box(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.0, "south": 43.0, "east": -71.0, "west": -70.0, "resolution": 0.1},
    )
    assert response.status_code == 400


def test_model_info_reports_the_winner(client):
    body = client.get("/ml/model-info").get_json()
    assert body["winner"] == "stub"
    assert body["feature_names"] == list(FEATURE_NAMES)


def test_errors_are_json_not_html(client):
    response = client.post("/ml/risk", json={})
    assert response.status_code == 400
    assert response.is_json


def test_non_finite_features_are_nulled_not_nan(client):
    """Flask renders NaN as a bare token that strict JSON parsers reject."""
    import json as jsonlib

    response = client.post("/ml/risk", json={"latitude": 42.4, "longitude": -71.1})
    body = response.get_data(as_text=True)
    assert "NaN" not in body
    jsonlib.loads(body)


def test_null_resolution_falls_back_to_the_default(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.4, "south": 42.2, "east": -71.0, "west": -71.2, "resolution": None},
    )
    assert response.status_code == 200


def test_non_numeric_resolution_is_a_clean_400(client):
    response = client.post(
        "/ml/heatmap",
        json={"north": 42.4, "south": 42.2, "east": -71.0, "west": -71.2, "resolution": "coarse"},
    )
    assert response.status_code == 400
    assert response.is_json
