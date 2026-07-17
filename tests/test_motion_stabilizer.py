import math

import pytest
import torch

from avtr1_renderer.motion_stabilizer import stabilize_normalized_motion
from avtr1_renderer.types import MotionStabilizationOptions

EXPRESSION_COORDINATES = 39


def _motion(
    rotation_degrees: list[float], expression_coordinate: int | None = None
) -> torch.Tensor:
    motion = torch.zeros((1, len(rotation_degrees), 3 + EXPRESSION_COORDINATES))
    motion[0, :, 0] = torch.tensor([math.radians(value) for value in rotation_degrees])
    if expression_coordinate is not None:
        motion[0, :, 3 + expression_coordinate] = torch.tensor(
            [0.0, 0.1, 4.0, 0.2, 0.3][: len(rotation_degrees)]
        )
    return motion


def _apply(
    motion: torch.Tensor,
    options: MotionStabilizationOptions,
    state=None,
):
    return stabilize_normalized_motion(
        motion,
        so3_offset=torch.zeros(3),
        so3_scale=torch.ones(3),
        options=options,
        state=state,
    )


def test_disabled_mode_is_identity_and_drops_stale_carry() -> None:
    motion = _motion([0.0, 0.1, 3.0])
    enabled = _apply(
        motion,
        MotionStabilizationOptions(mode="rotation"),
    )

    disabled = _apply(
        motion,
        MotionStabilizationOptions(),
        enabled.state,
    )

    assert disabled.normalized is motion
    assert disabled.state is None
    assert disabled.state_reset


def test_rotation_guard_suppresses_isolated_spike() -> None:
    motion = _motion([0.0, 0.1, 3.0, 0.2, 0.3])
    result = _apply(
        motion,
        MotionStabilizationOptions(
            mode="rotation",
            rotation_acceleration_threshold_deg=0.5,
            rotation_max_correction_deg=2.0,
        ),
    )

    raw = torch.rad2deg(result.raw_rotation[:, 0])
    corrected = torch.rad2deg(result.corrected_rotation[:, 0])
    assert corrected[2] < raw[2]
    assert torch.count_nonzero(result.rotation_correction).item() > 0
    assert result.raw_rotation_acceleration.shape == result.corrected_rotation_acceleration.shape
    assert (
        result.corrected_rotation_acceleration[2].norm()
        < result.raw_rotation_acceleration[2].norm()
    )
    assert torch.equal(result.normalized[0, :, 3:], motion[0, :, 3:])


def test_rotation_guard_preserves_constant_velocity() -> None:
    motion = _motion([0.0, 0.2, 0.4, 0.6, 0.8])
    result = _apply(
        motion,
        MotionStabilizationOptions(
            mode="rotation",
            rotation_acceleration_threshold_deg=0.5,
        ),
    )

    torch.testing.assert_close(result.corrected_rotation, result.raw_rotation)
    assert torch.count_nonzero(result.rotation_correction).item() == 0


def test_expression_guard_only_changes_weighted_coordinate() -> None:
    coordinate = 7
    weights = [0.0] * EXPRESSION_COORDINATES
    weights[coordinate] = 1.0
    motion = _motion([0.0] * 5, expression_coordinate=coordinate)
    motion[0, :, 3 + 8] = motion[0, :, 3 + coordinate]

    result = _apply(
        motion,
        MotionStabilizationOptions(
            mode="expression",
            expression_acceleration_threshold_z=0.5,
            expression_max_correction_z=2.0,
            expression_coordinate_weights=tuple(weights),
        ),
    )

    assert result.corrected_expression[2, coordinate] < result.raw_expression[2, coordinate]
    assert (
        result.corrected_expression_acceleration[2, coordinate].abs()
        < result.raw_expression_acceleration[2, coordinate].abs()
    )
    torch.testing.assert_close(
        result.corrected_expression[:, 8],
        result.raw_expression[:, 8],
    )


def test_state_carries_across_chunks_and_resets_on_mode_change() -> None:
    first = _apply(
        _motion([0.0, 0.2]),
        MotionStabilizationOptions(mode="rotation"),
    )
    second = _apply(
        _motion([3.0, 0.4]),
        MotionStabilizationOptions(
            mode="rotation",
            rotation_acceleration_threshold_deg=0.5,
        ),
        first.state,
    )

    assert not second.state_reset
    assert torch.count_nonzero(second.rotation_correction[0]).item() > 0

    weights = tuple(0.0 for _ in range(EXPRESSION_COORDINATES))
    changed_mode = _apply(
        _motion([3.0, 0.4]),
        MotionStabilizationOptions(
            mode="expression",
            expression_coordinate_weights=weights,
        ),
        second.state,
    )
    assert changed_mode.state_reset


def test_expression_mode_requires_explicit_profile() -> None:
    with pytest.raises(ValueError, match="explicit coordinate profile"):
        _apply(
            _motion([0.0, 0.1]),
            MotionStabilizationOptions(mode="expression"),
        )
