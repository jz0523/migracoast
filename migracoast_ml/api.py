"""Flask blueprint exposing the learned risk model."""

from flask import Blueprint, jsonify, request

from migracoast_ml.config import CELL_SIZE, STATES
from migracoast_ml.features.grid import snap_to_cell

# A heatmap request larger than this is rejected rather than served slowly.
MAX_GRID_CELLS = 2500


def _read_number(payload, key):
    if key not in payload or payload[key] is None:
        raise ValueError(f"Missing required field: {key}")
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Field {key} must be a number")


def _validate_point(lat, lon):
    if not -90.0 <= lat <= 90.0:
        raise ValueError("latitude must be between -90 and 90")
    if not -180.0 <= lon <= 180.0:
        raise ValueError("longitude must be between -180 and 180")


def create_blueprint(scorer, builder, covered_states=STATES) -> Blueprint:
    """Build the /ml blueprint around an already-loaded scorer and builder."""
    blueprint = Blueprint("migracoast_ml", __name__, url_prefix="/ml")

    @blueprint.post("/risk")
    def risk():
        payload = request.get_json(silent=True) or {}
        try:
            lat = _read_number(payload, "latitude")
            lon = _read_number(payload, "longitude")
            _validate_point(lat, lon)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        features = builder.build(lat, lon)
        result = scorer.score_features(features)

        contributors = []
        if getattr(scorer, "model", None) is not None:
            from migracoast_ml.models.explain import top_contributors

            contributors = top_contributors(scorer.model, features, scorer.feature_names)

        cell_lat, cell_lon = snap_to_cell(lat, lon)
        return jsonify(
            {
                **result,
                "latitude": lat,
                "longitude": lon,
                "cell": {"lat": cell_lat, "lon": cell_lon},
                "covered_states": list(covered_states),
                "features": features,
                "top_contributors": contributors,
            }
        )

    @blueprint.post("/heatmap")
    def heatmap():
        payload = request.get_json(silent=True) or {}
        try:
            north = _read_number(payload, "north")
            south = _read_number(payload, "south")
            east = _read_number(payload, "east")
            west = _read_number(payload, "west")
            resolution = float(payload.get("resolution", CELL_SIZE))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if north <= south or east <= west:
            return jsonify({"error": "north must exceed south and east must exceed west"}), 400
        if resolution <= 0:
            return jsonify({"error": "resolution must be positive"}), 400

        rows = int((north - south) / resolution) + 1
        cols = int((east - west) / resolution) + 1
        if rows * cols > MAX_GRID_CELLS:
            return jsonify(
                {
                    "error": f"Grid of {rows * cols} cells exceeds the limit of "
                    f"{MAX_GRID_CELLS}. Use a coarser resolution or a smaller box."
                }
            ), 400

        cells = []
        for r in range(rows):
            lat = south + r * resolution
            for c in range(cols):
                lon = west + c * resolution
                scored = scorer.score_features(builder.build(lat, lon))
                cells.append(
                    {
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "risk_percentile": scored["risk_percentile"],
                        "safety_score": scored["safety_score"],
                    }
                )

        return jsonify({"count": len(cells), "resolution": resolution, "cells": cells})

    @blueprint.get("/model-info")
    def model_info():
        metadata = dict(getattr(scorer, "metadata", {}))
        return jsonify(
            {
                "winner": metadata.get("winner"),
                "trained_on": metadata.get("trained_on"),
                "states": metadata.get("states", list(covered_states)),
                "scenario": metadata.get("scenario"),
                "feature_names": list(scorer.feature_names),
                "metrics": metadata.get("metrics", []),
                "n_cells": metadata.get("n_cells"),
            }
        )

    return blueprint
