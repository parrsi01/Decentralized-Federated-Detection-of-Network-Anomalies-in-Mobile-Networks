from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventSource(str, Enum):
    SSH = "ssh"
    HTTP = "http"
    HONEYPOT = "honeypot"
    DNS = "dns"
    FLOW = "flow"


class Event(BaseModel):
    """Normalized security telemetry event.

    Events are intentionally metadata-only. Raw packet payloads, credentials, and secrets should
    not be stored or exchanged by agents.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: EventSource
    src_ip: str
    dst_port: int = Field(ge=0, le=65535)
    action: str
    status_code: int | None = None
    path: str | None = None
    user_agent: str | None = None
    username: str | None = None
    bytes_in: int = Field(default=0, ge=0)
    bytes_out: int = Field(default=0, ge=0)
    label: int | None = Field(default=None, description="0 benign, 1 suspicious/anomalous")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureVector(BaseModel):
    event_count: float
    unique_src_ips: float
    failed_auth_count: float
    http_error_count: float
    unique_paths: float
    path_entropy: float
    port_diversity: float
    user_agent_rarity: float
    request_rate_per_minute: float
    bytes_ratio: float

    def as_list(self) -> list[float]:
        return [
            self.event_count,
            self.unique_src_ips,
            self.failed_auth_count,
            self.http_error_count,
            self.unique_paths,
            self.path_entropy,
            self.port_diversity,
            self.user_agent_rarity,
            self.request_rate_per_minute,
            self.bytes_ratio,
        ]


class Finding(BaseModel):
    agent_id: str
    timestamp: datetime
    score: float = Field(ge=0.0, le=1.0)
    label: str
    severity: str
    features: FeatureVector


class PeerSignal(BaseModel):
    peer_id: str
    model_quality: float = Field(ge=0.0, le=1.0)
    anomaly_rate: float = Field(ge=0.0, le=1.0)
    feature_importance: list[float]
    sample_count: int = Field(ge=0)
    poisoned: bool = False


class AgentSnapshot(BaseModel):
    agent_id: str
    trust: dict[str, float]
    findings: list[Finding]
    trained_samples: int
    selected_features: list[int]
