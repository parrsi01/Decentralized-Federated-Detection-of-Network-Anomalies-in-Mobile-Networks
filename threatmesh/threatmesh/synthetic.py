from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from threatmesh.models import Event, EventSource


BENIGN_PATHS = ["/", "/health", "/login", "/static/app.css", "/api/status"]
ATTACK_PATHS = ["/wp-admin", "/.env", "/phpmyadmin", "/admin", "/cgi-bin/test", "/api/debug"]
USER_AGENTS = ["Mozilla/5.0", "curl/8.1", "python-requests/2.31", "masscan", "sqlmap"]


def generate_window(
    agent_index: int,
    suspicious: bool,
    size: int | None = None,
    start: datetime | None = None,
) -> list[Event]:
    rng = random.Random((agent_index + 1) * (17 if suspicious else 7) + (size or 0))
    start = start or datetime.now(timezone.utc)
    count = size or rng.randint(18, 65 if suspicious else 35)
    src_pool = rng.randint(1, 4 if suspicious else 12)
    events: list[Event] = []

    for offset in range(count):
        source = rng.choice([EventSource.HTTP, EventSource.SSH, EventSource.HONEYPOT])
        is_http = source in {EventSource.HTTP, EventSource.HONEYPOT}
        status_code = rng.choice([200, 204, 301, 404, 403, 500] if suspicious else [200, 200, 204, 301, 404])
        action = "request"
        if source == EventSource.SSH:
            action = rng.choice(["failed_login", "failed_login", "auth_fail"] if suspicious else ["login", "failed_login"])
        path = rng.choice(ATTACK_PATHS if suspicious else BENIGN_PATHS) if is_http else None
        user_agent = rng.choice(USER_AGENTS if suspicious else USER_AGENTS[:3]) if is_http else None
        src_octet = rng.randint(1, src_pool)
        events.append(
            Event(
                timestamp=start + timedelta(seconds=offset * rng.randint(1, 4 if suspicious else 15)),
                source=source,
                src_ip=f"198.51.100.{agent_index * 20 + src_octet}",
                dst_port=rng.choice([22, 80, 443, 8080, 8443] if suspicious else [22, 80, 443]),
                action=action,
                status_code=status_code if is_http else None,
                path=path,
                user_agent=user_agent,
                username=rng.choice(["root", "admin", "test", "ubuntu"]) if source == EventSource.SSH else None,
                bytes_in=rng.randint(40, 4000),
                bytes_out=rng.randint(40, 12000 if suspicious else 5000),
                label=1 if suspicious else 0,
            )
        )
    return events


def generate_training_windows(agent_index: int, windows: int = 80) -> list[list[Event]]:
    return [generate_window(agent_index, suspicious=(i % 4 == 0), size=None) for i in range(windows)]
