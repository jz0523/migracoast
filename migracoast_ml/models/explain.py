"""Why the model said what it said."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model, X, y, n_repeats: int = 10, random_state: int = 42):
    """Global feature importance, strongest first."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="r2"
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def top_contributors(model, features: dict, feature_names, k: int = 5) -> list[dict]:
    """Per-prediction attribution by single-feature ablation.

    Each feature is replaced by the reference value of zero in turn; the shift
    in prediction is that feature's contribution. This always works, unlike
    SHAP which is an optional dependency.
    """
    names = list(feature_names)
    base_row = np.array([[float(features[name]) for name in names]])
    base_prediction = float(model.predict(base_row)[0])

    contributions = []
    for i, name in enumerate(names):
        ablated = base_row.copy()
        ablated[0, i] = 0.0
        shift = base_prediction - float(model.predict(ablated)[0])
        contributions.append(
            {"feature": name, "value": float(features[name]), "contribution": shift}
        )

    contributions.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return contributions[:k]


def _import_shap():
    """Isolated so tests can simulate SHAP being absent."""
    try:
        import shap

        return shap
    except ImportError:
        return None


def shap_contributions(model, features: dict, feature_names) -> dict | None:
    """SHAP values when the optional package is installed, otherwise None."""
    shap = _import_shap()
    if shap is None:
        return None

    names = list(feature_names)
    row = np.array([[float(features[name]) for name in names]])
    try:
        explainer = shap.Explainer(model.predict, row)
        values = explainer(row).values[0]
    except Exception:
        return None
    return {name: float(value) for name, value in zip(names, values)}
