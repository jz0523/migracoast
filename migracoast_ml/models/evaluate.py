"""Spatially blocked cross-validation.

Neighbouring cells are strongly correlated. A random split would place a
cell's neighbour in the test fold and report a score that cannot be
reproduced on genuinely unseen ground, so folds are split by 1 degree block.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def _safe_spearman(truth, predicted) -> float:
    # An exact zero test misses floating point noise: a constant predictor
    # can measure a standard deviation around 1e-18 and still be constant.
    if len(truth) < 3 or np.std(predicted) < 1e-12:
        return float("nan")
    return float(spearmanr(truth, predicted).statistic)


def spatial_cv(X, y, groups, coastal, models, n_splits: int = 5) -> pd.DataFrame:
    """Score every model under blocked cross-validation.

    Metrics are reported twice: over all cells, and over coastal cells only,
    because coastal locations are the product's actual use case.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    coastal = pd.Series(coastal).reset_index(drop=True)

    splitter = GroupKFold(n_splits=n_splits)
    rows = []

    for name, model in models.items():
        fold = {"rmse": [], "r2": [], "spearman": [],
                "coastal_rmse": [], "coastal_spearman": []}

        for train_idx, test_idx in splitter.split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = fitted.predict(X.iloc[test_idx])
            truth = y.iloc[test_idx].to_numpy()

            fold["rmse"].append(float(np.sqrt(mean_squared_error(truth, predicted))))
            fold["r2"].append(float(r2_score(truth, predicted)))
            fold["spearman"].append(_safe_spearman(truth, predicted))

            mask = coastal.iloc[test_idx].to_numpy()
            if mask.sum() >= 3:
                fold["coastal_rmse"].append(
                    float(np.sqrt(mean_squared_error(truth[mask], predicted[mask])))
                )
                fold["coastal_spearman"].append(_safe_spearman(truth[mask], predicted[mask]))

        rows.append(
            {
                "model": name,
                "rmse_mean": float(np.nanmean(fold["rmse"])),
                "rmse_std": float(np.nanstd(fold["rmse"])),
                "r2_mean": float(np.nanmean(fold["r2"])),
                "r2_std": float(np.nanstd(fold["r2"])),
                "spearman_mean": float(np.nanmean(fold["spearman"])),
                "coastal_rmse_mean": float(np.nanmean(fold["coastal_rmse"])),
                "coastal_rmse_std": float(np.nanstd(fold["coastal_rmse"])),
                "coastal_spearman_mean": float(np.nanmean(fold["coastal_spearman"])),
                # How many folds actually had enough coastal cells to score.
                # Fewer than n_splits means the coastal mean is a partial average.
                "coastal_n_folds": int(len(fold["coastal_rmse"])),
            }
        )

    return pd.DataFrame(rows)


def oof_predictions(X, y, groups, model, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions, the only honest input to the baseline comparison."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    out = np.full(len(X), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        out[test_idx] = fitted.predict(X.iloc[test_idx])
    return out
