# ThreatMesh

ThreatMesh is a defensive cybersecurity prototype for decentralized honeypot and anomaly intelligence. It turns the MARL-XGBoost network-anomaly research idea into a runnable tool: lightweight agents train on local telemetry windows, inspect new activity, and exchange only trust-weighted metadata signals with neighboring peers.

The project is designed for lawful defensive research, lab use, portfolio demonstrations, and authorized security monitoring. It does not exploit systems, bypass controls, or claim anonymity.

## What It Does

- Trains local anomaly detectors from metadata-only security events.
- Extracts features from SSH, HTTP, DNS, flow, and honeypot-style telemetry.
- Uses XGBoost by default, with a scikit-learn fallback if XGBoost is unavailable.
- Simulates a decentralized ring topology of peer agents.
- Exchanges peer quality, anomaly-rate, sample-count, and feature-importance signals instead of raw logs.
- Maintains fail-closed trust scores for peers, including poisoned-agent simulation.
- Exposes both a CLI and FastAPI API for demos and integration.

## Architecture

```text
Security Events -> Feature Windows -> Local Detector -> Finding
                                      |
                                      v
Peer Signal <- Trust Engine <- Neighbor Agents
```

Each agent has:

- a local detector
- an epsilon-greedy feature policy
- a trust engine
- recent findings
- peer-signal ingestion

## Install

```bash
cd threatmesh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run a Simulation

```bash
threatmesh simulate --agents 3 --rounds 2
```

Write JSON output:

```bash
threatmesh simulate --agents 4 --rounds 5 --poisoned-agent agent-2 --output output/simulation.json
```

## Run the API

```bash
threatmesh serve --host 127.0.0.1 --port 8088
```

Endpoints:

- `GET /health`
- `GET /agents`
- `POST /simulate`
- `POST /simulate?poisoned_agent=agent-2`

Interactive OpenAPI docs are available at:

```text
http://127.0.0.1:8088/docs
```

## Event Model

ThreatMesh uses normalized metadata events. Do not collect or exchange payloads, credentials, secrets, or packet contents.

Example:

```json
{
  "source": "http",
  "src_ip": "198.51.100.10",
  "dst_port": 443,
  "action": "request",
  "status_code": 404,
  "path": "/.env",
  "user_agent": "curl/8.1",
  "bytes_in": 120,
  "bytes_out": 512,
  "label": 1
}
```

## Security and Legal Use

ThreatMesh is for:

- defensive monitoring
- authorized lab environments
- honeypot research
- anomaly-detection experiments
- federated/decentralized ML research

Do not use it to attack systems, evade detection, bypass rate limits, or test third-party infrastructure without explicit authorization.

## Development

```bash
pytest
ruff check .
```

## Roadmap

- Real Zeek, Suricata, OpenSSH, and nginx log parsers.
- Persistent SQLite event store.
- Web dashboard for topology, trust, and findings.
- Secure peer transport with mTLS.
- Differential privacy and secure aggregation experiments.
- Adversarial peer benchmarks for label-flip, random-update, and free-rider behavior.

## Repository Description

Decentralized honeypot and anomaly intelligence mesh using local XGBoost agents, trust-weighted peer signaling, and MARL-style feature selection for defensive cybersecurity research.
