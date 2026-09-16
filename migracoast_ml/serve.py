"""Standalone Flask server for the ML API.

The rule engine in app.py cannot be imported without its raster and NetCDF
data files, so the ML API must be able to run entirely on its own.
"""

from flask import Flask, jsonify
from flask_cors import CORS

from migracoast_ml.api import create_blueprint
from migracoast_ml.models.train import MODEL_PATH


def _load_scorer():
    from migracoast_ml.models.scoring import RiskScorer

    return RiskScorer.load(MODEL_PATH)


def _load_builder():
    from migracoast_ml.features.builder import FeatureBuilder

    return FeatureBuilder.load()


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Loaded once at startup and held in memory, so a point query is
    # a few milliseconds rather than a few seconds.
    app.register_blueprint(create_blueprint(_load_scorer(), _load_builder()))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def run(host: str = "127.0.0.1", port: int = 5001) -> None:
    create_app().run(host=host, port=port, debug=False)
