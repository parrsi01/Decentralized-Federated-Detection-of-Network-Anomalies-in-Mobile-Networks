from threatmesh.mesh import ThreatMesh


def test_mesh_bootstrap_and_signal_exchange() -> None:
    mesh = ThreatMesh(agent_count=3)
    mesh.bootstrap(windows_per_agent=20)
    result = mesh.inspect_synthetic_round(poisoned_agent="agent-2")

    assert len(result.findings) == 3
    assert len(result.snapshots) == 3
    assert any(snapshot.trust for snapshot in result.snapshots)
