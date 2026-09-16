"""Turn a raw model prediction into the scores the product exposes."""

import numpy as np
import joblib


class RiskScorer:
    """Wraps the fitted model and the reference distribution used for ranking."""

    def __init__(self, model, reference, feature_names, metadata=None):
        self.model = model
        self.reference = np.sort(np.asarray(reference, dtype=float))
        self.feature_names = tuple(feature_names)
        self.metadata = dict(metadata or {})

    def _vector(self, features: dict) -> np.ndarray:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Feature vector missing: {missing}")
        # Order by the configured feature list, never by dict insertion order.
        return np.array([[float(features[name]) for name in self.feature_names]])

    def percentile(self, prediction: float) -> float:
        """Where this prediction falls among the reference cells, 0 to 100."""
        if self.reference.size == 0:
            return float("nan")
        position = float(np.searchsorted(self.reference, prediction, side="right"))
        return float(np.clip(100.0 * position / self.reference.size, 0.0, 100.0))

    def score_features(self, features: dict) -> dict:
        """Score one already-built feature vector."""
        prediction = float(self.model.predict(self._vector(features))[0])
        risk = self.percentile(prediction)
        return {
            "predicted_claim_intensity": prediction,
            "risk_percentile": round(risk, 1),
            # The site's existing convention is higher means safer.
            "safety_score": round(100.0 - risk, 1),
        }

    def save(self, path) -> None:
        joblib.dump(
            {
                "model": self.model,
                "reference": self.reference,
                "feature_names": self.feature_names,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "RiskScorer":
        blob = joblib.load(path)
        return cls(
            blob["model"], blob["reference"], blob["feature_names"], blob["metadata"]
        )
