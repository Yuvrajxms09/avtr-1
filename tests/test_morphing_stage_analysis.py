import numpy as np
import pytest

from scripts.analyze_morphing_stages import _compare, _geometry_metrics


def test_geometry_metrics_remove_rigid_similarity_motion() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(21, 3))
    angle = np.linspace(-0.2, 0.2, 8)
    frames = []
    for index, value in enumerate(angle):
        rotation = np.asarray(
            [
                [np.cos(value), -np.sin(value), 0.0],
                [np.sin(value), np.cos(value), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        frames.append((1.0 + index * 0.01) * (source @ rotation) + index * 0.03)
    metrics = _geometry_metrics(
        source,
        np.asarray(frames),
        np.arange(21),
        np.arange(21),
    )
    assert metrics["nonrigid_residual_rms"]["max"] < 1e-10
    assert metrics["temporal_step_rms"]["max"] < 1e-10


def test_comparison_rejects_different_motion_fingerprints() -> None:
    with pytest.raises(ValueError, match="different raw-motion fingerprints"):
        _compare(
            {"metadata": {"motion_fingerprint_sha256": "one"}},
            {"metadata": {"motion_fingerprint_sha256": "two"}},
        )
