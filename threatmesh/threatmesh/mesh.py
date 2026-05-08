from __future__ import annotations

from dataclasses import dataclass

from threatmesh.agent import MeshAgent
from threatmesh.models import AgentSnapshot, Finding
from threatmesh.synthetic import generate_training_windows, generate_window


def ring_topology(agent_ids: list[str]) -> dict[str, list[str]]:
    if len(agent_ids) <= 1:
        return {agent_id: [] for agent_id in agent_ids}
    return {
        agent_id: [agent_ids[(index - 1) % len(agent_ids)], agent_ids[(index + 1) % len(agent_ids)]]
        for index, agent_id in enumerate(agent_ids)
    }


@dataclass
class MeshRunResult:
    findings: list[Finding]
    snapshots: list[AgentSnapshot]


class ThreatMesh:
    """In-process decentralized mesh simulator."""

    def __init__(self, agent_count: int = 3) -> None:
        self.agents = {f"agent-{i + 1}": MeshAgent(f"agent-{i + 1}") for i in range(agent_count)}
        self.topology = ring_topology(list(self.agents.keys()))

    def bootstrap(self, windows_per_agent: int = 80) -> None:
        for index, agent in enumerate(self.agents.values()):
            agent.train(generate_training_windows(index, windows=windows_per_agent))

    def exchange_signals(self, poisoned_agent: str | None = None) -> None:
        signals = {
            agent_id: agent.signal(poisoned=(agent_id == poisoned_agent))
            for agent_id, agent in self.agents.items()
        }
        for agent_id, neighbors in self.topology.items():
            for neighbor_id in neighbors:
                self.agents[agent_id].receive(signals[neighbor_id])

    def inspect_synthetic_round(self, poisoned_agent: str | None = None) -> MeshRunResult:
        self.exchange_signals(poisoned_agent=poisoned_agent)
        findings: list[Finding] = []
        for index, agent in enumerate(self.agents.values()):
            suspicious = index % 2 == 0
            findings.append(agent.inspect(generate_window(index, suspicious=suspicious)))
        return MeshRunResult(findings=findings, snapshots=self.snapshots())

    def snapshots(self) -> list[AgentSnapshot]:
        return [agent.snapshot() for agent in self.agents.values()]
