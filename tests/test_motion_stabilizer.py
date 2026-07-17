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


def test_enabling_continuous_filter_resets_spike_only_carry() -> None:
    spike_only = _apply(
        _motion([0.0, 0.2]),
        MotionStabilizationOptions(mode="rotation"),
    )
    filtered = _apply(
        _motion([0.3, 0.4]),
        MotionStabilizationOptions(
            mode="rotation",
            rotation_temporal_filter="one_euro",
        ),
        spike_only.state,
    )

    assert filtered.state_reset


def test_one_euro_rotation_filter_is_bounded_and_reduces_oscillation() -> None:
    motion = _motion([0.0, 0.5, -0.5, 0.5, -0.5])
    result = _apply(
        motion,
        MotionStabilizationOptions(
            mode="rotation",
            rotation_spike_guard=False,
            rotation_temporal_filter="one_euro",
            rotation_one_euro_min_cutoff_hz=2.0,
            rotation_one_euro_beta=0.1,
            rotation_temporal_max_correction_deg=0.5,
        ),
    )

    raw_step = torch.diff(result.raw_rotation[:, 0]).abs().mean()
    corrected_step = torch.diff(result.corrected_rotation[:, 0]).abs().mean()
    correction_deg = torch.rad2deg(
        torch.linalg.vector_norm(result.rotation_temporal_correction, dim=-1)
    )
    assert corrected_step < raw_step
    assert correction_deg.max() <= 0.5 + 1e-5
    assert torch.count_nonzero(result.rotation_spike_correction).item() == 0


def test_rotation_acceleration_limiter_bounds_output_acceleration() -> None:
    result = _apply(
        _motion([0.0, 0.0, 1.0, 1.0, 1.0]),
        MotionStabilizationOptions(
            mode="rotation",
            rotation_spike_guard=False,
            rotation_temporal_max_correction_deg=2.0,
            rotation_max_acceleration_deg=0.1,
        ),
    )

    acceleration_deg = torch.rad2deg(
        torch.linalg.vector_norm(result.corrected_rotation_acceleration, dim=-1)
    )
    assert acceleration_deg.max() <= 0.1 + 1e-5


def test_rotation_jerk_limiter_bounds_output_jerk() -> None:
    result = _apply(
        _motion([0.0, 0.0, 0.2, 1.0, -0.5, 0.5]),
        MotionStabilizationOptions(
            mode="rotation",
            rotation_spike_guard=False,
            rotation_temporal_max_correction_deg=3.0,
            rotation_max_jerk_deg=0.1,
        ),
    )

    jerk_deg = torch.rad2deg(
        torch.linalg.vector_norm(torch.diff(result.corrected_rotation_acceleration, dim=0), dim=-1)
    )
    assert jerk_deg.max() <= 0.1 + 1e-5


def test_one_euro_filter_is_continuous_across_chunks() -> None:
    options = MotionStabilizationOptions(
        mode="rotation",
        rotation_spike_guard=False,
        rotation_temporal_filter="one_euro",
        rotation_temporal_max_correction_deg=2.0,
    )
    whole = _apply(_motion([0.0, 0.3, -0.2, 0.6, 0.1]), options)
    first = _apply(_motion([0.0, 0.3]), options)
    second = _apply(_motion([-0.2, 0.6, 0.1]), options, first.state)

    torch.testing.assert_close(
        torch.cat([first.corrected_rotation, second.corrected_rotation]),
        whole.corrected_rotation,
    )


def test_continuous_expression_filter_only_changes_weighted_coordinates() -> None:
    weights = [0.0] * EXPRESSION_COORDINATES
    weights[7] = 0.5
    motion = _motion([0.0] * 5)
    values = torch.tensor([0.0, 1.0, -1.0, 1.0, -1.0])
    motion[0, :, 3 + 7] = values
    motion[0, :, 3 + 8] = values

    result = _apply(
        motion,
        MotionStabilizationOptions(
            mode="expression",
            expression_spike_guard=False,
            expression_temporal_filter="one_euro",
            expression_coordinate_weights=tuple(weights),
            expression_temporal_max_correction_z=0.25,
        ),
    )

    assert torch.count_nonzero(result.expression_temporal_correction[:, 7]).item() > 0
    assert result.expression_temporal_correction[:, 7].abs().max() <= 0.25
    torch.testing.assert_close(
        result.corrected_expression[:, 8],
        result.raw_expression[:, 8],
    )


def test_expression_mode_requires_explicit_profile() -> None:
    with pytest.raises(ValueError, match="explicit coordinate profile"):
        _apply(
            _motion([0.0, 0.1]),
            MotionStabilizationOptions(mode="expression"),
        )


def test_temporal_filter_options_validate_bounds() -> None:
    with pytest.raises(ValueError, match="rotation_one_euro_beta"):
        _apply(
            _motion([0.0, 0.1]),
            MotionStabilizationOptions(
                mode="rotation",
                rotation_temporal_filter="one_euro",
                rotation_one_euro_beta=-0.1,
            ),
        )

    with pytest.raises(ValueError, match="rotation_max_acceleration_deg"):
        _apply(
            _motion([0.0, 0.1]),
            MotionStabilizationOptions(
                mode="rotation",
                rotation_max_acceleration_deg=-0.1,
            ),
        )
