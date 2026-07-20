from pathlib import Path

import pytest
import torch

from avtr1_renderer.motion_trajectory import MotionTrajectorySession, intercept_raw_motion

_SO3_OFFSET = torch.zeros(3)
_SO3_SCALE = torch.ones(3)


def _intercept(values: torch.Tensor) -> torch.Tensor:
    return intercept_raw_motion(
        values,
        so3_offset=_SO3_OFFSET,
        so3_scale=_SO3_SCALE,
    )


def _capture(path: Path) -> list[torch.Tensor]:
    chunks = [torch.randn(1, 5, 42), torch.randn(1, 5, 42)]
    with MotionTrajectorySession.capture(path, {"avatar_id": "test", "seed": 123}):
        for chunk in chunks:
            assert _intercept(chunk) is chunk
    return chunks


def test_capture_and_replay_use_identical_raw_chunks(tmp_path: Path) -> None:
    path = tmp_path / "motion.safetensors"
    captured = _capture(path)
    generated = torch.zeros_like(captured[0])
    with MotionTrajectorySession.replay(
        path,
        {"avatar_id": "test", "seed": 123},
    ) as replay:
        first = _intercept(generated)
        second = _intercept(generated)
        assert replay.fingerprint is not None
    torch.testing.assert_close(first, captured[0])
    torch.testing.assert_close(second, captured[1])


def test_replay_rejects_metadata_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "motion.safetensors"
    _capture(path)
    with pytest.raises(ValueError, match="metadata mismatch"):
        MotionTrajectorySession.replay(path, {"avatar_id": "different", "seed": 123})


def test_replay_requires_every_chunk_to_be_consumed(tmp_path: Path) -> None:
    path = tmp_path / "motion.safetensors"
    _capture(path)
    with pytest.raises(ValueError, match="before all chunks"):
        with MotionTrajectorySession.replay(path, {"avatar_id": "test", "seed": 123}):
            _intercept(torch.zeros(1, 5, 42))
