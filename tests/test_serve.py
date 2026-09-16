import re
from pathlib import Path

import pytest

from migracoast_ml import serve

APP_PY = Path(__file__).parent.parent / "app.py"


def test_create_app_registers_the_ml_blueprint(monkeypatch):
    class StubScorer:
        feature_names = ("a",)
        metadata = {}
        model = None

    monkeypatch.setattr(serve, "_load_scorer", lambda: StubScorer())
    monkeypatch.setattr(serve, "_load_builder", lambda: object())

    app = serve.create_app()
    rules = {str(rule) for rule in app.url_map.iter_rules()}
    assert "/ml/risk" in rules
    assert "/ml/heatmap" in rules
    assert "/ml/model-info" in rules


def test_health_endpoint_responds(monkeypatch):
    class StubScorer:
        feature_names = ("a",)
        metadata = {}
        model = None

    monkeypatch.setattr(serve, "_load_scorer", lambda: StubScorer())
    monkeypatch.setattr(serve, "_load_builder", lambda: object())

    client = serve.create_app().test_client()
    assert client.get("/health").status_code == 200


def test_app_py_has_no_hardcoded_password():
    """The committed credential is replaced by an environment lookup."""
    source = APP_PY.read_text(encoding="utf-8", errors="replace")
    assert "St5SfZXpxRPS4XZy" not in source
    assert "os.environ" in source


def test_app_py_reads_every_db_field_from_the_environment():
    source = APP_PY.read_text(encoding="utf-8", errors="replace")
    block = re.search(r"db_config\s*=\s*\{.*?\}", source, re.S)
    assert block is not None
    for field in ("MIGRACOAST_DB_HOST", "MIGRACOAST_DB_USER",
                  "MIGRACOAST_DB_PASSWORD", "MIGRACOAST_DB_NAME"):
        assert field in block.group(0)
