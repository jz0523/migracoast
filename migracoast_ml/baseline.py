"""Compare the hand-tuned rule engine against the learned model.

The rule engine's 480 saved scores carry no coordinates. They are recovered
from the notebook's point lists, whose row counts reconcile exactly with the
CSV. That reconciliation is strong evidence but not proof, so
`alignment_confidence` checks the mapping against elevation before any
comparison is reported.
"""

import ast
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Concatenation order inferred from row counts. BOS absorbs MA_points_new
# because those twelve coordinates are downtown Boston.
REGION_LISTS = {
    "MA": ("MA_points",),
    "NY": ("NY_points",),
    "PA": ("PA_points", "PA_points_new"),
    "FL": ("FL_points", "FL_points_new"),
    "CA": ("CA_points", "CA_points_new"),
    "BOS": ("BOS_points", "BOS_points_new", "MA_points_new"),
}

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\[[^\]]*\])", re.M | re.S)


def extract_point_lists(notebook_path) -> dict[str, list[tuple[float, float]]]:
    """Pull every coordinate list out of the notebook source."""
    notebook = json.loads(open(notebook_path, encoding="utf-8").read())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    found = {}
    for match in _ASSIGNMENT.finditer(source):
        name, body = match.group(1), match.group(2)
        if "points" not in name.lower():
            continue
        try:
            value = ast.literal_eval(body)
        except (ValueError, SyntaxError):
            continue
        pairs = [t for t in value if isinstance(t, tuple) and len(t) == 2]
        if pairs:
            found[name] = pairs
    return found


def reconstruct_coordinates(notebook_path, scores_path) -> pd.DataFrame:
    """Attach coordinates to every row of the saved rule-engine scores."""
    lists = extract_point_lists(notebook_path)
    scores = pd.read_csv(scores_path)

    frames = []
    for scenario in sorted(scores["Scenario"].unique()):
        for region, list_names in REGION_LISTS.items():
            block = scores[
                (scores["State"] == region) & (scores["Scenario"] == scenario)
            ].reset_index(drop=True)
            if block.empty:
                continue

            points = []
            for name in list_names:
                points.extend(lists.get(name, []))

            if len(points) != len(block):
                raise ValueError(
                    f"Coordinate count mismatch for {region}/{scenario}: "
                    f"{len(points)} points against {len(block)} rows"
                )

            block = block.copy()
            block["lat"] = [p[0] for p in points]
            block["lon"] = [p[1] for p in points]
            frames.append(block)

    return pd.concat(frames, ignore_index=True)


def alignment_confidence(frame: pd.DataFrame) -> float:
    """Spearman correlation between the rule score and elevation.

    The rule score is oriented so that higher means safer, so a correct
    coordinate alignment must produce a clearly positive correlation. A value
    near zero or negative means the mapping is wrong.
    """
    usable = frame.dropna(subset=["rule_score", "elev_mean"])
    if len(usable) < 10:
        return float("nan")
    return float(spearmanr(usable["rule_score"], usable["elev_mean"]).statistic)


def run_comparison(
    notebook_path="Check_inundation.ipynb",
    scores_path="score_time_series.csv",
    min_confidence: float = 0.2,
) -> str:
    """Produce the comparison, or explain why it cannot be trusted."""
    from migracoast_ml.dataset import load_training_table
    from migracoast_ml.features.grid import snap_to_cell
    from migracoast_ml.models.evaluate import oof_predictions
    from migracoast_ml.models.ladder import build_models
    from migracoast_ml.models.train import METRICS_PATH
    from migracoast_ml.config import FEATURE_NAMES

    coords = reconstruct_coordinates(notebook_path, scores_path)
    # 2020 is the first scored year and the one closest to the claim record.
    coords["rule_score"] = coords["2020"].astype(float)

    snapped = [snap_to_cell(a, o) for a, o in zip(coords["lat"], coords["lon"])]
    coords["cell_lat"] = [s[0] for s in snapped]
    coords["cell_lon"] = [s[1] for s in snapped]

    table = load_training_table()
    winner = json.loads(METRICS_PATH.read_text())["winner"]
    table["oof"] = oof_predictions(
        table[list(FEATURE_NAMES)], table["target"], table["block"], build_models()[winner]
    )

    merged = coords.merge(
        table[["cell_lat", "cell_lon", "claim_count", "oof", "elev_mean"]],
        on=["cell_lat", "cell_lon"],
        how="inner",
    )
    # One row per cell; the same cell repeats across scenarios and years.
    merged = merged.drop_duplicates(subset=["cell_lat", "cell_lon"])

    confidence = alignment_confidence(merged)
    lines = [
        f"Matched cells: {len(merged)} of {len(coords)} scored rows",
        f"Alignment confidence (rule score vs elevation): {confidence:.3f}",
    ]

    if not np.isfinite(confidence) or confidence < min_confidence:
        lines.append(
            "ALIGNMENT REJECTED. The reconstructed coordinates do not track "
            "elevation as a higher-is-safer score must. The comparison is not "
            "reported, because publishing it would mean trusting a mapping the "
            "data contradicts."
        )
        return "\n".join(lines)

    if len(merged) < 10:
        lines.append("Too few matched cells to compare.")
        return "\n".join(lines)

    rule_rho = float(spearmanr(merged["claim_count"], merged["rule_score"]).statistic)
    model_rho = float(spearmanr(merged["claim_count"], merged["oof"]).statistic)

    lines += [
        "",
        "Spearman correlation with observed claim counts:",
        f"  rule engine score (higher = safer): {rule_rho:+.3f}",
        f"  learned model, out-of-fold:         {model_rho:+.3f}",
        "",
        "The rule score is expected to correlate negatively, since higher means",
        "safer and more claims mean more risk. Compare absolute magnitudes.",
    ]

    if abs(model_rho) > abs(rule_rho):
        lines.append("RESULT: the learned model ranks risk more accurately.")
    else:
        lines.append(
            "RESULT: the rule engine ranks risk at least as well on these cells. "
            "Reported as found, without re-tuning the model to overturn it."
        )
    return "\n".join(lines)
