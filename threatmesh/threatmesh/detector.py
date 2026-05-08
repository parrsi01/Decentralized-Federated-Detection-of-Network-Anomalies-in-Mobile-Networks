from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - exercised only when xgboost is unavailable
    XGBClassifier = None  # type: ignore[assignment]


class LocalDetector:
    """Local anomaly detector with XGBoost preferred and RandomForest fallback."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = self._new_model()
        self.quality = 0.0
        self.trained_samples = 0
        self.feature_importance: list[float] = []

    def _new_model(self):
        if XGBClassifier is not None:
            return XGBClassifier(
                eval_metric="logloss",
                tree_method="hist",
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
            )
        return RandomForestClassifier(n_estimators=120, max_depth=8, random_state=self.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0 or len(np.unique(y)) < 2:
            self.quality = 0.0
            self.trained_samples = int(len(x))
            return self.quality

        stratify = y if min(np.bincount(y)) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=self.random_state, stratify=stratify
        )
        self.model = self._new_model()
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        self.quality = float(f1_score(y_test, predictions, zero_division=0))
        self.trained_samples = int(len(x))
        importance = getattr(self.model, "feature_importances_", np.zeros(x.shape[1]))
        total = float(np.sum(importance))
        self.feature_importance = (importance / total).tolist() if total else [0.0] * x.shape[1]
        return self.quality

    def predict_score(self, x: np.ndarray) -> float:
        if not hasattr(self.model, "classes_"):
            return 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x.reshape(1, -1))[0]
            classes = list(self.model.classes_)
            if 1 in classes:
                return float(proba[classes.index(1)])
            return 0.0
        return float(self.model.predict(x.reshape(1, -1))[0])
