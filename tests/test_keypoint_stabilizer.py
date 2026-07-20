import pytest
import torch

from avtr1_renderer.keypoint_stabilizer import (
    source_locked_keypoint_indices,
    stabilize_keypoints,
)
from avtr1_renderer.types import GeometryStabilizationOptions


def _inputs(frames: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    source = torch.randn(1, 21, 3) * 0.1
    raw = source.expand(frames, -1, -1).clone()
    network = raw + torch.randn_like(raw) * 0.005
    return source, raw, network


def test_geometry_defaults_preserve_network_output() -> None:
    source, raw, network = _inputs()
    result = stabilize_keypoints(
        source,
        raw,
        network,
        options=GeometryStabilizationOptions(),
        state=None,
    )
    assert torch.equal(result.final, network)
    assert result.state is None


def test_stitch_strength_blends_network_correction() -> None:
    source, raw, network = _inputs()
    result = stabilize_keypoints(
        source,
        raw,
        network,
        options=GeometryStabilizationOptions(stitch_strength=0.25),
        state=None,
    )
    torch.testing.assert_close(result.final, raw + 0.25 * (network - raw))


def test_post_stitch_filter_never_changes_unselected_keypoints() -> None:
    source, raw, network = _inputs(frames=8)
    selected = source_locked_keypoint_indices(21)
    result = stabilize_keypoints(
        source,
        raw,
        network,
        options=GeometryStabilizationOptions(
            post_stitch_enabled=True,
            post_stitch_keypoint_indices=selected,
            post_stitch_max_correction=0.002,
        ),
        state=None,
    )
    unselected = [index for index in range(21) if index not in selected]
    assert torch.equal(result.final[:, unselected], result.blended[:, unselected])
    correction = torch.linalg.vector_norm(result.post_stitch_correction, dim=-1)
    assert float(correction.max()) <= 0.002001


def test_post_stitch_filter_rejects_lipsync_keypoints() -> None:
    source, raw, network = _inputs()
    with pytest.raises(ValueError, match="cannot target lipsync keypoints"):
        stabilize_keypoints(
            source,
            raw,
            network,
            options=GeometryStabilizationOptions(
                post_stitch_enabled=True,
                post_stitch_keypoint_indices=(0, 1, 3),
            ),
            state=None,
        )


def test_geometry_filter_state_continues_across_chunks() -> None:
    source, raw, network = _inputs(frames=10)
    options = GeometryStabilizationOptions(
        stitch_temporal_filter="one_euro",
        stitch_temporal_max_correction=0.01,
    )
    full = stabilize_keypoints(source, raw, network, options=options, state=None)
    first = stabilize_keypoints(source, raw[:5], network[:5], options=options, state=None)
    second = stabilize_keypoints(
        source,
        raw[5:],
        network[5:],
        options=options,
        state=first.state,
    )
    torch.testing.assert_close(torch.cat([first.final, second.final]), full.final)
