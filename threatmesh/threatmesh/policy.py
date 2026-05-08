from __future__ import annotations

import random

import numpy as np


class FeaturePolicy:
    """Small epsilon-greedy policy for feature subset selection.

    This is intentionally a lightweight MARL-compatible policy layer: each agent uses local reward
    feedback to bias future feature selection without requiring a central coordinator.
    """

    def __init__(self, feature_count: int, epsilon: float = 0.25, random_state: int = 42) -> None:
        self.feature_count = feature_count
        self.epsilon = epsilon
        self.random = random.Random(random_state)
        self.values = np.ones(feature_count, dtype=float)

    def select(self, min_features: int = 4) -> list[int]:
        size = self.random.randint(min(min_features, self.feature_count), self.feature_count)
        if self.random.random() < self.epsilon:
            return sorted(self.random.sample(range(self.feature_count), size))
        ranked = np.argsort(self.values)[::-1][:size]
        return sorted(int(i) for i in ranked)

    def reward(self, selected: list[int], value: float) -> None:
        for index in selected:
            self.values[index] = 0.85 * self.values[index] + 0.15 * value
        self.epsilon = max(0.05, self.epsilon * 0.98)
