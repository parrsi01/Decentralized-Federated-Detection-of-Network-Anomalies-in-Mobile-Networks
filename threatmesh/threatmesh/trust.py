from __future__ import annotations

import math

import numpy as np

from threatmesh.models import PeerSignal


class TrustEngine:
    """Maintains fail-closed peer trust scores for decentralized signal exchange."""

    def __init__(self, initial_trust: float = 0.5, decay: float = 0.9) -> None:
        self.initial_trust = initial_trust
        self.decay = decay
        self.scores: dict[str, float] = {}

    def get(self, peer_id: str) -> float:
        return self.scores.get(peer_id, self.initial_trust)

    def update(self, signal: PeerSignal, local_quality: float) -> float:
        previous = self.get(signal.peer_id)
        quality_delta = abs(signal.model_quality - local_quality)
        quality_score = max(0.0, 1.0 - quality_delta)
        volume_score = min(1.0, math.log10(signal.sample_count + 1) / 4.0)
        importance_score = self._importance_sanity(signal.feature_importance)
        poison_penalty = 0.0 if signal.poisoned else 1.0
        observed = 0.45 * quality_score + 0.2 * volume_score + 0.2 * importance_score + 0.15 * poison_penalty
        updated = self.decay * previous + (1.0 - self.decay) * observed
        updated = min(1.0, max(0.0, updated))
        self.scores[signal.peer_id] = updated
        return updated

    @staticmethod
    def _importance_sanity(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            return 0.0
        total = float(arr.sum())
        if total <= 0:
            return 0.2
        normalized = arr / total
        concentration = float(np.max(normalized))
        return max(0.0, 1.0 - concentration)
