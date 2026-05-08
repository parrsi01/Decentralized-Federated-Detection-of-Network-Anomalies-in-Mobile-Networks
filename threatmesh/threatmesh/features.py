from __future__ import annotations

import math
from collections import Counter
from datetime import datetime

import numpy as np

from threatmesh.models import Event, FeatureVector


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def _safe_ratio(left: float, right: float) -> float:
    return float(left / right) if right else float(left)


def extract_window_features(events: list[Event]) -> FeatureVector:
    """Convert a telemetry window into deterministic numeric features."""

    if not events:
        return FeatureVector(
            event_count=0.0,
            unique_src_ips=0.0,
            failed_auth_count=0.0,
            http_error_count=0.0,
            unique_paths=0.0,
            path_entropy=0.0,
            port_diversity=0.0,
            user_agent_rarity=0.0,
            request_rate_per_minute=0.0,
            bytes_ratio=0.0,
        )

    timestamps = [event.timestamp for event in events if isinstance(event.timestamp, datetime)]
    duration_seconds = max((max(timestamps) - min(timestamps)).total_seconds(), 1.0) if timestamps else 60.0
    paths = [event.path or "" for event in events if event.path]
    user_agents = [event.user_agent or "" for event in events if event.user_agent]
    user_agent_counts = Counter(user_agents)
    rare_user_agents = sum(1 for ua in user_agents if user_agent_counts[ua] == 1)
    bytes_in = sum(event.bytes_in for event in events)
    bytes_out = sum(event.bytes_out for event in events)

    return FeatureVector(
        event_count=float(len(events)),
        unique_src_ips=float(len({event.src_ip for event in events})),
        failed_auth_count=float(
            sum(1 for event in events if event.action.lower() in {"failed_login", "auth_fail"})
        ),
        http_error_count=float(sum(1 for event in events if (event.status_code or 0) >= 400)),
        unique_paths=float(len(set(paths))),
        path_entropy=_entropy(paths),
        port_diversity=float(len({event.dst_port for event in events})),
        user_agent_rarity=_safe_ratio(rare_user_agents, len(user_agents)),
        request_rate_per_minute=float(len(events) / (duration_seconds / 60.0)),
        bytes_ratio=_safe_ratio(bytes_out, max(bytes_in, 1)),
    )


def build_training_matrix(windows: list[list[Event]]) -> tuple[np.ndarray, np.ndarray]:
    vectors = [extract_window_features(window).as_list() for window in windows]
    labels = [1 if any(event.label == 1 for event in window) else 0 for window in windows]
    return np.asarray(vectors, dtype=float), np.asarray(labels, dtype=int)
