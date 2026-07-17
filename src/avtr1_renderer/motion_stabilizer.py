# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Stateful, opt-in temporal filtering for AVTR-1 motion predictions.

The model emits normalized ``[so3 | expression]`` vectors. This module keeps
the autoregressive model state untouched and corrects only the copy sent to
the renderer. Each channel can independently run an isolated spike guard, an
adaptive One Euro low-pass filter, and bounded acceleration/jerk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from avtr1_renderer.types import MotionStabilizationOptions

_MODE_CODE = {"none": 0, "rotation": 1, "expression": 2, "both": 3}
_MOTION_FPS = 25.0


@dataclass(slots=True, frozen=True)
class MotionStabilizerState:
    mode_code: int
    rotation_position: torch.Tensor | None = None
    rotation_velocity: torch.Tensor | None = None
    rotation_acceleration: torch.Tensor | None = None
    rotation_raw_position: torch.Tensor | None = None
    rotation_raw_velocity: torch.Tensor | None = None
    rotation_filter_input: torch.Tensor | None = None
    rotation_filter_derivative: torch.Tensor | None = None
    expression_position: torch.Tensor | None = None
    expression_velocity: torch.Tensor | None = None
    expression_acceleration: torch.Tensor | None = None
    expression_raw_position: torch.Tensor | None = None
    expression_raw_velocity: torch.Tensor | None = None
    expression_filter_input: torch.Tensor | None = None
    expression_filter_derivative: torch.Tensor | None = None


@dataclass(slots=True, frozen=True)
class MotionStabilizationResult:
    normalized: torch.Tensor
    state: MotionStabilizerState | None
    state_reset: bool
    raw_rotation: torch.Tensor
    spike_corrected_rotation: torch.Tensor
    corrected_rotation: torch.Tensor
    raw_rotation_velocity: torch.Tensor
    corrected_rotation_velocity: torch.Tensor
    raw_rotation_acceleration: torch.Tensor
    corrected_rotation_acceleration: torch.Tensor
    rotation_spike_correction: torch.Tensor
    rotation_lowpass_requested_correction: torch.Tensor
    rotation_kinematic_requested_correction: torch.Tensor
    rotation_temporal_correction: torch.Tensor
    rotation_correction: torch.Tensor
    raw_expression: torch.Tensor
    spike_corrected_expression: torch.Tensor
    corrected_expression: torch.Tensor
    raw_expression_velocity: torch.Tensor
    corrected_expression_velocity: torch.Tensor
    raw_expression_acceleration: torch.Tensor
    corrected_expression_acceleration: torch.Tensor
    expression_spike_correction: torch.Tensor
    expression_lowpass_requested_correction: torch.Tensor
    expression_kinematic_requested_correction: torch.Tensor
    expression_temporal_correction: torch.Tensor
    expression_correction: torch.Tensor


@dataclass(slots=True, frozen=True)
class _TemporalResult:
    values: torch.Tensor
    velocities: torch.Tensor
    accelerations: torch.Tensor
    lowpass_requested_correction: torch.Tensor
    kinematic_requested_correction: torch.Tensor
    temporal_correction: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    filter_input: torch.Tensor
    filter_derivative: torch.Tensor


def _empty_rows(reference: torch.Tensor, columns: int) -> torch.Tensor:
    return reference.new_empty((0, columns))


def _configuration_code(options: MotionStabilizationOptions) -> int:
    """Encode state-shape semantics while preserving legacy spike-only codes."""
    code = _MODE_CODE[options.mode]
    if options.rotation_enabled and not options.rotation_spike_guard:
        code |= 1 << 6
    if options.expression_enabled and not options.expression_spike_guard:
        code |= 1 << 7
    if options.rotation_enabled and options.rotation_temporal_filter != "none":
        code |= 1 << 2
    if options.expression_enabled and options.expression_temporal_filter != "none":
        code |= 1 << 3
    if options.rotation_enabled and options.rotation_kinematic_limiter_enabled:
        code |= 1 << 4
    if options.expression_enabled and options.expression_kinematic_limiter_enabled:
        code |= 1 << 5
    return code


def _raw_kinematics(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    position = previous_position
    velocity = previous_velocity
    for value in values:
        current_velocity = value - position if position is not None else torch.zeros_like(value)
        current_acceleration = (
            current_velocity - velocity if velocity is not None else torch.zeros_like(value)
        )
        velocities.append(current_velocity)
        accelerations.append(current_acceleration)
        position = value
        velocity = current_velocity
    return torch.stack(velocities), torch.stack(accelerations)


def _guard_vector_spikes(
    values: torch.Tensor,
    *,
    previous_raw_position: torch.Tensor | None,
    acceleration_threshold: float,
    max_correction: float,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrected: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    previous_raw = previous_raw_position
    for index, raw in enumerate(values):
        correction = torch.zeros_like(raw)
        if previous_raw is not None and index + 1 < len(values):
            following = values[index + 1]
            delta_before = raw - previous_raw
            delta_after = following - raw
            before_norm = torch.linalg.vector_norm(delta_before)
            after_norm = torch.linalg.vector_norm(delta_after)
            reverses = torch.dot(delta_before, delta_after) < 0
            is_isolated = reverses & (
                torch.minimum(before_norm, after_norm) > acceleration_threshold
            )
            residual = raw - (previous_raw + following) * 0.5
            residual_norm = torch.linalg.vector_norm(residual).clamp_min(1e-8)
            excess = (residual_norm - acceleration_threshold).clamp_min(0.0) * is_isolated
            correction_magnitude = torch.minimum(
                excess * strength,
                residual_norm.new_tensor(max_correction),
            )
            correction = residual * (correction_magnitude / residual_norm)
        corrected.append(raw - correction)
        corrections.append(correction)
        previous_raw = raw
    return torch.stack(corrected), torch.stack(corrections)


def _guard_coordinate_spikes(
    values: torch.Tensor,
    *,
    previous_raw_position: torch.Tensor | None,
    acceleration_threshold: float,
    max_correction: float,
    strength: float,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    corrected: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    previous_raw = previous_raw_position
    for index, raw in enumerate(values):
        correction = torch.zeros_like(raw)
        if previous_raw is not None and index + 1 < len(values):
            following = values[index + 1]
            delta_before = raw - previous_raw
            delta_after = following - raw
            is_isolated = (
                (delta_before * delta_after < 0)
                & (delta_before.abs() > acceleration_threshold)
                & (delta_after.abs() > acceleration_threshold)
            )
            residual = raw - (previous_raw + following) * 0.5
            excess = (residual.abs() - acceleration_threshold).clamp_min(0.0)
            magnitude = (excess * strength).clamp_max(max_correction) * weights
            correction = residual.sign() * magnitude * is_isolated
        corrected.append(raw - correction)
        corrections.append(correction)
        previous_raw = raw
    return torch.stack(corrected), torch.stack(corrections)


def _lowpass_alpha(cutoff_hz: torch.Tensor, *, fps: float) -> torch.Tensor:
    time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
    sample_period = 1.0 / fps
    return 1.0 / (1.0 + time_constant / sample_period)


def _clamp_vector_norm(value: torch.Tensor, maximum: float) -> torch.Tensor:
    if maximum <= 0:
        return value
    norm = torch.linalg.vector_norm(value).clamp_min(1e-8)
    scale = torch.minimum(norm.new_tensor(1.0), norm.new_tensor(maximum) / norm)
    return value * scale


def _clamp_coordinates(value: torch.Tensor, maximum: float) -> torch.Tensor:
    return value.clamp(-maximum, maximum) if maximum > 0 else value


def _filter_vector_sequence(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor | None,
    previous_acceleration: torch.Tensor | None,
    previous_filter_input: torch.Tensor | None,
    previous_filter_derivative: torch.Tensor | None,
    temporal_filter: str,
    min_cutoff_hz: float,
    beta: float,
    derivative_cutoff_hz: float,
    max_temporal_correction: float,
    max_acceleration: float,
    max_jerk: float,
) -> _TemporalResult:
    outputs: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    lowpass_corrections: list[torch.Tensor] = []
    kinematic_corrections: list[torch.Tensor] = []
    temporal_corrections: list[torch.Tensor] = []
    position = previous_position
    velocity = previous_velocity
    acceleration = previous_acceleration
    filter_input = previous_filter_input
    filter_derivative = previous_filter_derivative
    derivative_alpha = _lowpass_alpha(values.new_tensor(derivative_cutoff_hz), fps=_MOTION_FPS)

    for value in values:
        raw_derivative = (
            (value - filter_input) * _MOTION_FPS
            if filter_input is not None
            else torch.zeros_like(value)
        )
        filtered_derivative = (
            derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * filter_derivative
            if filter_derivative is not None
            else raw_derivative
        )
        lowpass_target = value
        if temporal_filter == "one_euro" and position is not None:
            speed = torch.linalg.vector_norm(filtered_derivative)
            cutoff = value.new_tensor(min_cutoff_hz) + beta * speed
            alpha = _lowpass_alpha(cutoff, fps=_MOTION_FPS)
            lowpass_target = alpha * value + (1.0 - alpha) * position

        limited_target = lowpass_target
        if position is not None and velocity is not None:
            desired_velocity = lowpass_target - position
            desired_acceleration = desired_velocity - velocity
            if max_jerk > 0 and acceleration is not None:
                desired_jerk = desired_acceleration - acceleration
                desired_acceleration = acceleration + _clamp_vector_norm(desired_jerk, max_jerk)
            desired_acceleration = _clamp_vector_norm(desired_acceleration, max_acceleration)
            limited_target = position + velocity + desired_acceleration

        temporal_correction = _clamp_vector_norm(
            value - limited_target,
            max_temporal_correction,
        )
        output = value - temporal_correction
        output_velocity = output - position if position is not None else torch.zeros_like(value)
        output_acceleration = (
            output_velocity - velocity if velocity is not None else torch.zeros_like(value)
        )
        outputs.append(output)
        velocities.append(output_velocity)
        accelerations.append(output_acceleration)
        lowpass_corrections.append(value - lowpass_target)
        kinematic_corrections.append(lowpass_target - limited_target)
        temporal_corrections.append(temporal_correction)
        position = output
        velocity = output_velocity
        acceleration = output_acceleration
        filter_input = value
        filter_derivative = filtered_derivative

    assert position is not None
    assert velocity is not None
    assert acceleration is not None
    assert filter_input is not None
    assert filter_derivative is not None
    return _TemporalResult(
        values=torch.stack(outputs),
        velocities=torch.stack(velocities),
        accelerations=torch.stack(accelerations),
        lowpass_requested_correction=torch.stack(lowpass_corrections),
        kinematic_requested_correction=torch.stack(kinematic_corrections),
        temporal_correction=torch.stack(temporal_corrections),
        position=position.clone(),
        velocity=velocity.clone(),
        acceleration=acceleration.clone(),
        filter_input=filter_input.clone(),
        filter_derivative=filter_derivative.clone(),
    )


def _filter_coordinate_sequence(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor | None,
    previous_acceleration: torch.Tensor | None,
    previous_filter_input: torch.Tensor | None,
    previous_filter_derivative: torch.Tensor | None,
    temporal_filter: str,
    min_cutoff_hz: float,
    beta: float,
    derivative_cutoff_hz: float,
    max_temporal_correction: float,
    max_acceleration: float,
    max_jerk: float,
    weights: torch.Tensor,
) -> _TemporalResult:
    outputs: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    accelerations: list[torch.Tensor] = []
    lowpass_corrections: list[torch.Tensor] = []
    kinematic_corrections: list[torch.Tensor] = []
    temporal_corrections: list[torch.Tensor] = []
    position = previous_position
    velocity = previous_velocity
    acceleration = previous_acceleration
    filter_input = previous_filter_input
    filter_derivative = previous_filter_derivative
    derivative_alpha = _lowpass_alpha(values.new_tensor(derivative_cutoff_hz), fps=_MOTION_FPS)

    for value in values:
        raw_derivative = (
            (value - filter_input) * _MOTION_FPS
            if filter_input is not None
            else torch.zeros_like(value)
        )
        filtered_derivative = (
            derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * filter_derivative
            if filter_derivative is not None
            else raw_derivative
        )
        unweighted_lowpass_target = value
        if temporal_filter == "one_euro" and position is not None:
            cutoff = value.new_tensor(min_cutoff_hz) + beta * filtered_derivative.abs()
            alpha = _lowpass_alpha(cutoff, fps=_MOTION_FPS)
            unweighted_lowpass_target = alpha * value + (1.0 - alpha) * position
        lowpass_target = value + weights * (unweighted_lowpass_target - value)

        unweighted_limited_target = lowpass_target
        if position is not None and velocity is not None:
            desired_velocity = lowpass_target - position
            desired_acceleration = desired_velocity - velocity
            if max_jerk > 0 and acceleration is not None:
                desired_jerk = desired_acceleration - acceleration
                desired_acceleration = acceleration + _clamp_coordinates(desired_jerk, max_jerk)
            desired_acceleration = _clamp_coordinates(desired_acceleration, max_acceleration)
            unweighted_limited_target = position + velocity + desired_acceleration
        limited_target = lowpass_target + weights * (unweighted_limited_target - lowpass_target)

        temporal_correction = _clamp_coordinates(
            value - limited_target,
            max_temporal_correction,
        )
        output = value - temporal_correction
        output_velocity = output - position if position is not None else torch.zeros_like(value)
        output_acceleration = (
            output_velocity - velocity if velocity is not None else torch.zeros_like(value)
        )
        outputs.append(output)
        velocities.append(output_velocity)
        accelerations.append(output_acceleration)
        lowpass_corrections.append(value - lowpass_target)
        kinematic_corrections.append(lowpass_target - limited_target)
        temporal_corrections.append(temporal_correction)
        position = output
        velocity = output_velocity
        acceleration = output_acceleration
        filter_input = value
        filter_derivative = filtered_derivative

    assert position is not None
    assert velocity is not None
    assert acceleration is not None
    assert filter_input is not None
    assert filter_derivative is not None
    return _TemporalResult(
        values=torch.stack(outputs),
        velocities=torch.stack(velocities),
        accelerations=torch.stack(accelerations),
        lowpass_requested_correction=torch.stack(lowpass_corrections),
        kinematic_requested_correction=torch.stack(kinematic_corrections),
        temporal_correction=torch.stack(temporal_corrections),
        position=position.clone(),
        velocity=velocity.clone(),
        acceleration=acceleration.clone(),
        filter_input=filter_input.clone(),
        filter_derivative=filter_derivative.clone(),
    )


@torch.no_grad()
def stabilize_normalized_motion(
    normalized: torch.Tensor,
    *,
    so3_offset: torch.Tensor,
    so3_scale: torch.Tensor,
    options: MotionStabilizationOptions,
    state: MotionStabilizerState | None,
) -> MotionStabilizationResult:
    """Return a render-only corrected copy of ``normalized`` and next carry."""
    if (
        normalized.ndim != 3
        or normalized.shape[0] != 1
        or normalized.shape[1] < 1
        or normalized.shape[-1] < 4
    ):
        raise ValueError(
            f"normalized motion must have shape (1, T, 3 + E), got {tuple(normalized.shape)}"
        )
    expression_coordinates = normalized.shape[-1] - 3
    options.validate(expression_coordinates=expression_coordinates)

    raw_rotation = normalized[0, :, :3] * so3_scale + so3_offset
    raw_expression = normalized[0, :, 3:]
    empty_rotation = _empty_rows(normalized, 3)
    empty_expression = _empty_rows(normalized, expression_coordinates)
    zero_rotation = torch.zeros_like(raw_rotation)
    zero_expression = torch.zeros_like(raw_expression)

    if options.mode == "none":
        return MotionStabilizationResult(
            normalized=normalized,
            state=None,
            state_reset=state is not None,
            raw_rotation=raw_rotation,
            spike_corrected_rotation=raw_rotation,
            corrected_rotation=raw_rotation,
            raw_rotation_velocity=empty_rotation,
            corrected_rotation_velocity=empty_rotation,
            raw_rotation_acceleration=empty_rotation,
            corrected_rotation_acceleration=empty_rotation,
            rotation_spike_correction=zero_rotation,
            rotation_lowpass_requested_correction=zero_rotation,
            rotation_kinematic_requested_correction=zero_rotation,
            rotation_temporal_correction=zero_rotation,
            rotation_correction=zero_rotation,
            raw_expression=raw_expression,
            spike_corrected_expression=raw_expression,
            corrected_expression=raw_expression,
            raw_expression_velocity=empty_expression,
            corrected_expression_velocity=empty_expression,
            raw_expression_acceleration=empty_expression,
            corrected_expression_acceleration=empty_expression,
            expression_spike_correction=zero_expression,
            expression_lowpass_requested_correction=zero_expression,
            expression_kinematic_requested_correction=zero_expression,
            expression_temporal_correction=zero_expression,
            expression_correction=zero_expression,
        )

    mode_code = _configuration_code(options)
    state_reset = state is None or state.mode_code != mode_code
    if state_reset:
        state = MotionStabilizerState(mode_code=mode_code)
    assert state is not None
    corrected = normalized.clone()

    if options.rotation_enabled:
        raw_rotation_velocity, raw_rotation_acceleration = _raw_kinematics(
            raw_rotation,
            previous_position=state.rotation_raw_position,
            previous_velocity=state.rotation_raw_velocity,
        )
        if options.rotation_spike_guard:
            spike_rotation, rotation_spike_correction = _guard_vector_spikes(
                raw_rotation,
                previous_raw_position=state.rotation_raw_position,
                acceleration_threshold=math.radians(options.rotation_acceleration_threshold_deg),
                max_correction=math.radians(options.rotation_max_correction_deg),
                strength=options.rotation_strength,
            )
        else:
            spike_rotation = raw_rotation
            rotation_spike_correction = zero_rotation
        rotation_temporal = _filter_vector_sequence(
            spike_rotation,
            previous_position=state.rotation_position,
            previous_velocity=state.rotation_velocity,
            previous_acceleration=state.rotation_acceleration,
            previous_filter_input=state.rotation_filter_input,
            previous_filter_derivative=state.rotation_filter_derivative,
            temporal_filter=options.rotation_temporal_filter,
            min_cutoff_hz=options.rotation_one_euro_min_cutoff_hz,
            beta=options.rotation_one_euro_beta,
            derivative_cutoff_hz=options.rotation_one_euro_derivative_cutoff_hz,
            max_temporal_correction=math.radians(options.rotation_temporal_max_correction_deg),
            max_acceleration=math.radians(options.rotation_max_acceleration_deg),
            max_jerk=math.radians(options.rotation_max_jerk_deg),
        )
        corrected_rotation = rotation_temporal.values
        corrected[0, :, :3] = (corrected_rotation - so3_offset) / so3_scale
    else:
        raw_rotation_velocity = empty_rotation
        raw_rotation_acceleration = empty_rotation
        spike_rotation = raw_rotation
        rotation_spike_correction = zero_rotation
        rotation_temporal = None
        corrected_rotation = raw_rotation

    weights = (
        normalized.new_tensor(options.expression_coordinate_weights)
        if options.expression_coordinate_weights is not None
        else normalized.new_zeros(expression_coordinates)
    )
    if options.expression_enabled:
        raw_expression_velocity, raw_expression_acceleration = _raw_kinematics(
            raw_expression,
            previous_position=state.expression_raw_position,
            previous_velocity=state.expression_raw_velocity,
        )
        if options.expression_spike_guard:
            spike_expression, expression_spike_correction = _guard_coordinate_spikes(
                raw_expression,
                previous_raw_position=state.expression_raw_position,
                acceleration_threshold=options.expression_acceleration_threshold_z,
                max_correction=options.expression_max_correction_z,
                strength=options.expression_strength,
                weights=weights,
            )
        else:
            spike_expression = raw_expression
            expression_spike_correction = zero_expression
        expression_temporal = _filter_coordinate_sequence(
            spike_expression,
            previous_position=state.expression_position,
            previous_velocity=state.expression_velocity,
            previous_acceleration=state.expression_acceleration,
            previous_filter_input=state.expression_filter_input,
            previous_filter_derivative=state.expression_filter_derivative,
            temporal_filter=options.expression_temporal_filter,
            min_cutoff_hz=options.expression_one_euro_min_cutoff_hz,
            beta=options.expression_one_euro_beta,
            derivative_cutoff_hz=options.expression_one_euro_derivative_cutoff_hz,
            max_temporal_correction=options.expression_temporal_max_correction_z,
            max_acceleration=options.expression_max_acceleration_z,
            max_jerk=options.expression_max_jerk_z,
            weights=weights,
        )
        corrected_expression = expression_temporal.values
        corrected[0, :, 3:] = corrected_expression
    else:
        raw_expression_velocity = empty_expression
        raw_expression_acceleration = empty_expression
        spike_expression = raw_expression
        expression_spike_correction = zero_expression
        expression_temporal = None
        corrected_expression = raw_expression

    rotation_temporal_correction = (
        rotation_temporal.temporal_correction if rotation_temporal is not None else zero_rotation
    )
    expression_temporal_correction = (
        expression_temporal.temporal_correction
        if expression_temporal is not None
        else zero_expression
    )
    next_state = MotionStabilizerState(
        mode_code=mode_code,
        rotation_position=(rotation_temporal.position if rotation_temporal else None),
        rotation_velocity=(rotation_temporal.velocity if rotation_temporal else None),
        rotation_acceleration=(rotation_temporal.acceleration if rotation_temporal else None),
        rotation_raw_position=(raw_rotation[-1].clone() if options.rotation_enabled else None),
        rotation_raw_velocity=(
            raw_rotation_velocity[-1].clone() if options.rotation_enabled else None
        ),
        rotation_filter_input=(rotation_temporal.filter_input if rotation_temporal else None),
        rotation_filter_derivative=(
            rotation_temporal.filter_derivative if rotation_temporal else None
        ),
        expression_position=(expression_temporal.position if expression_temporal else None),
        expression_velocity=(expression_temporal.velocity if expression_temporal else None),
        expression_acceleration=(expression_temporal.acceleration if expression_temporal else None),
        expression_raw_position=(
            raw_expression[-1].clone() if options.expression_enabled else None
        ),
        expression_raw_velocity=(
            raw_expression_velocity[-1].clone() if options.expression_enabled else None
        ),
        expression_filter_input=(expression_temporal.filter_input if expression_temporal else None),
        expression_filter_derivative=(
            expression_temporal.filter_derivative if expression_temporal else None
        ),
    )
    return MotionStabilizationResult(
        normalized=corrected,
        state=next_state,
        state_reset=state_reset,
        raw_rotation=raw_rotation,
        spike_corrected_rotation=spike_rotation,
        corrected_rotation=corrected_rotation,
        raw_rotation_velocity=raw_rotation_velocity,
        corrected_rotation_velocity=(
            rotation_temporal.velocities if rotation_temporal else empty_rotation
        ),
        raw_rotation_acceleration=raw_rotation_acceleration,
        corrected_rotation_acceleration=(
            rotation_temporal.accelerations if rotation_temporal else empty_rotation
        ),
        rotation_spike_correction=rotation_spike_correction,
        rotation_lowpass_requested_correction=(
            rotation_temporal.lowpass_requested_correction if rotation_temporal else zero_rotation
        ),
        rotation_kinematic_requested_correction=(
            rotation_temporal.kinematic_requested_correction if rotation_temporal else zero_rotation
        ),
        rotation_temporal_correction=rotation_temporal_correction,
        rotation_correction=raw_rotation - corrected_rotation,
        raw_expression=raw_expression,
        spike_corrected_expression=spike_expression,
        corrected_expression=corrected_expression,
        raw_expression_velocity=raw_expression_velocity,
        corrected_expression_velocity=(
            expression_temporal.velocities if expression_temporal else empty_expression
        ),
        raw_expression_acceleration=raw_expression_acceleration,
        corrected_expression_acceleration=(
            expression_temporal.accelerations if expression_temporal else empty_expression
        ),
        expression_spike_correction=expression_spike_correction,
        expression_lowpass_requested_correction=(
            expression_temporal.lowpass_requested_correction
            if expression_temporal
            else zero_expression
        ),
        expression_kinematic_requested_correction=(
            expression_temporal.kinematic_requested_correction
            if expression_temporal
            else zero_expression
        ),
        expression_temporal_correction=expression_temporal_correction,
        expression_correction=raw_expression - corrected_expression,
    )


__all__ = [
    "MotionStabilizationResult",
    "MotionStabilizerState",
    "stabilize_normalized_motion",
]
