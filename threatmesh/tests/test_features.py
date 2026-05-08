from threatmesh.features import extract_window_features
from threatmesh.synthetic import generate_window


def test_extract_window_features_distinguishes_activity() -> None:
    benign = extract_window_features(generate_window(0, suspicious=False))
    suspicious = extract_window_features(generate_window(0, suspicious=True))

    assert suspicious.failed_auth_count >= benign.failed_auth_count
    assert suspicious.event_count >= benign.event_count
    assert suspicious.port_diversity >= 1
