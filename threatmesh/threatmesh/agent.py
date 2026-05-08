from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from threatmesh.detector import LocalDetector
from threatmesh.features import build_training_matrix, extract_window_features
from threatmesh.models import AgentSnapshot, Event, Finding, PeerSignal
from threatmesh.policy import FeaturePolicy
from threatmesh.trust import TrustEngine


class MeshAgent:
    """A decentralized security agent that trains locally and exchanges metadata signals."""

    def __init__(self, agent_id: str, feature_count: int = 10) -> None:
        self.agent_id = agent_id
        self.detector = LocalDetector()
        self.trust = TrustEngine()
        self.policy = FeaturePolicy(feature_count=feature_count)
        self.selected_features = list(range(feature_count))
        self.findings: list[Finding] = []
        self._training_x: np.ndarray | None = None
        self._training_y: np.ndarray | None = None

    def train(self, windows: list[list[Event]]) -> float:
        x, y = build_training_matrix(windows)
        self._training_x = x
        self._training_y = y
        self.selected_features = self.policy.select()
        quality = self.detector.fit(x[:, self.selected_features], y)
        self.policy.reward(self.selected_features, quality)
        return quality

    def inspect(self, window: list[Event], threshold: float = 0.65) -> Finding:
        features = extract_window_features(window)
        vector = np.asarray(features.as_list(), dtype=float)
        score = self.detector.predict_score(vector[self.selected_features])
        severity = "high" if score >= 0.85 else "medium" if score >= threshold else "low"
        label = "anomalous" if score >= threshold else "normal"
        finding = Finding(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            score=score,
            label=label,
            severity=severity,
            features=features,
        )
        self.findings.append(finding)
        return finding

    def signal(self, poisoned: bool = False) -> PeerSignal:
        importance = self.detector.feature_importance or [0.0] * len(self.selected_features)
        full_importance = [0.0] * self.policy.feature_count
        for local_index, feature_index in enumerate(self.selected_features):
            if local_index < len(importance):
                full_importance[feature_index] = float(importance[local_index])
        anomaly_rate = (
            float(np.mean(self._training_y)) if self._training_y is not None and len(self._training_y) else 0.0
        )
        quality = max(0.0, min(1.0, self.detector.quality))
        if poisoned:
            quality = 1.0 - quality
            full_importance = list(reversed(full_importance))
        return PeerSignal(
            peer_id=self.agent_id,
            model_quality=quality,
            anomaly_rate=anomaly_rate,
            feature_importance=full_importance,
            sample_count=self.detector.trained_samples,
            poisoned=poisoned,
        )

    def receive(self, signal: PeerSignal) -> float:
        return self.trust.update(signal, self.detector.quality)

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            agent_id=self.agent_id,
            trust=self.trust.scores,
            findings=self.findings[-25:],
            trained_samples=self.detector.trained_samples,
            selected_features=self.selected_features,
        )
