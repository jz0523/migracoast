"""The three competing models, fit on identical features.

No PyTorch, LightGBM or XGBoost. On roughly ten thousand rows and eleven
features they add installation friction without adding accuracy, and being
able to explain that choice is worth more than the library name.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    """Return the model ladder, each rung a self-contained pipeline.

    Imputation and scaling live inside the pipeline so that no preprocessing
    statistic can leak across a cross-validation fold.
    """
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                # HistGradientBoosting handles NaN natively, but imputing keeps
                # every rung on identical inputs so the comparison is fair.
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.06,
                        max_depth=6,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "neural_net": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-3,
                        learning_rate_init=1e-3,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
