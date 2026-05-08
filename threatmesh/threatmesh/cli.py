from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from threatmesh.mesh import ThreatMesh

app = typer.Typer(help="ThreatMesh decentralized honeypot/anomaly intelligence toolkit.")


@app.command()
def simulate(
    agents: int = typer.Option(3, min=1, help="Number of mesh agents."),
    rounds: int = typer.Option(3, min=1, help="Synthetic inspection rounds."),
    poisoned_agent: str | None = typer.Option(None, help="Optional peer ID to mark as poisoned."),
    output: Path | None = typer.Option(None, help="Write JSON results to this file."),
) -> None:
    """Run a local decentralized anomaly-detection simulation."""

    mesh = ThreatMesh(agent_count=agents)
    mesh.bootstrap()
    results = []
    for _ in range(rounds):
        result = mesh.inspect_synthetic_round(poisoned_agent=poisoned_agent)
        results.append(
            {
                "findings": [finding.model_dump(mode="json") for finding in result.findings],
                "agents": [snapshot.model_dump(mode="json") for snapshot in result.snapshots],
            }
        )
    payload = {"rounds": results}
    rendered = json.dumps(payload, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(rendered)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8088, help="Bind port."),
) -> None:
    """Start the local API server."""

    uvicorn.run("threatmesh.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
