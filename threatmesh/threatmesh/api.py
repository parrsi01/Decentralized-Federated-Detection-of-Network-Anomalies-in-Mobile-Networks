from __future__ import annotations

from fastapi import FastAPI

from threatmesh.mesh import ThreatMesh

app = FastAPI(
    title="ThreatMesh",
    description="Decentralized honeypot and anomaly intelligence mesh API.",
    version="0.1.0",
)

mesh = ThreatMesh(agent_count=3)
mesh.bootstrap(windows_per_agent=60)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents")
def agents():
    return mesh.snapshots()


@app.post("/simulate")
def simulate(poisoned_agent: str | None = None):
    return mesh.inspect_synthetic_round(poisoned_agent=poisoned_agent)
