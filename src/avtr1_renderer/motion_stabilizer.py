# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Stateful, opt-in spike guards for AVTR-1 motion predictions.

The model emits normalized ``[so3 | expression]`` vectors. This module keeps
the autoregressive model state untouched and corrects only the copy sent to
the renderer. Both guards detect one-frame reversals against adjacent raw
frames and remove only the excess above a configured threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from avtr1_renderer.types import MotionStabilizationOptions

_MODE_CODE = {"none": 0, "rotation": 1, "expression": 2, "both": 3}


@dataclass(slots=True, frozen=True)
class MotionStabilizerState:
    mode_code: int
    rotation_position: torch.Tensor | None = None
    rotation_velocity: torch.Tensor | None = None
    rotation_raw_position: torch.Tensor | None = None
    rotation_raw_velocity: torch.Tensor | None = None
    expression_position: torch.Tensor | None = None
    expression_velocity: torch.Tensor | None = None
    expression_raw_position: torch.Tensor | None = None
    expression_raw_velocity: torch.Tensor | None = None


@dataclass(slots=True, frozen=True)
class MotionStabilizationResult:
    normalized: torch.Tensor
    state: MotionStabilizerState | None
    state_reset: bool
    raw_rotation: torch.Tensor
    corrected_rotation: torch.Tensor
    raw_rotation_velocity: torch.Tensor
    corrected_rotation_velocity: torch.Tensor
    raw_rotation_acceleration: torch.Tensor
    corrected_rotation_acceleration: torch.Tensor
    rotation_correction: torch.Tensor
    raw_expression: torch.Tensor
    corrected_expression: torch.Tensor
    raw_expression_velocity: torch.Tensor
    corrected_expression_velocity: torch.Tensor
    raw_expression_acceleration: torch.Tensor
    corrected_expression_acceleration: torch.Tensor
    expression_correction: torch.Tensor


def _empty_rows(reference: torch.Tensor, columns: int) -> torch.Tensor:
    return reference.new_empty((0, columns))


def _guard_vector_norm(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor | None,
    previous_raw_position: torch.Tensor | None,
    previous_raw_velocity: torch.Tensor | None,
    acceleration_threshold: float,
    max_correction: float,
    strength: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Suppress isolated vector reversals while preserving sustained motion."""
    corrected: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    corrected_velocities: list[torch.Tensor] = []
    raw_accelerations: list[torch.Tensor] = []
    corrected_accelerations: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []

    position = previous_position
    velocity = previous_velocity
    previous_raw = previous_raw_position

    for index, raw in enumerate(values):
        raw_velocity = raw - previous_raw if previous_raw is not None else torch.zeros_like(raw)
        raw_acceleration = (
            raw_velocity - previous_raw_velocity
            if previous_raw_velocity is not None
            else torch.zeros_like(raw)
        )
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
            predicted = (previous_raw + following) * 0.5
            residual = raw - predicted
            residual_norm = torch.linalg.vector_norm(residual).clamp_min(1e-8)
            excess = (residual_norm - acceleration_threshold).clamp_min(0.0) * is_isolated
            correction_magnitude = torch.minimum(
                excess * strength,
                residual_norm.new_tensor(max_correction),
            )
            correction = residual * (correction_magnitude / residual_norm)
        filtered = raw - correction
        filtered_velocity = filtered - position if position is not None else torch.zeros_like(raw)
        filtered_acceleration = (
            filtered_velocity - velocity if velocity is not None else torch.zeros_like(raw)
        )

        corrected.append(filtered)
        raw_velocities.append(raw_velocity)
        corrected_velocities.append(filtered_velocity)
        raw_accelerations.append(raw_acceleration)
        corrected_accelerations.append(filtered_acceleration)
        corrections.append(correction)
        previous_raw = raw
        previous_raw_velocity = raw_velocity
        position = filtered
        velocity = filtered_velocity

    return (
        torch.stack(corrected),
        torch.stack(raw_velocities),
        torch.stack(corrected_velocities),
        torch.stack(raw_accelerations),
        torch.stack(corrected_accelerations),
        torch.stack(corrections),
    )


def _guard_coordinates(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_velocity: torch.Tensor | None,
    previous_raw_position: torch.Tensor | None,
    previous_raw_velocity: torch.Tensor | None,
    acceleration_threshold: float,
    max_correction: float,
    strength: float,
    weights: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Suppress independent one-frame normalized-coordinate reversals."""
    corrected: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    corrected_velocities: list[torch.Tensor] = []
    raw_accelerations: list[torch.Tensor] = []
    corrected_accelerations: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []

    position = previous_position
    velocity = previous_velocity
    previous_raw = previous_raw_position

    for index, raw in enumerate(values):
        raw_velocity = raw - previous_raw if previous_raw is not None else torch.zeros_like(raw)
        raw_acceleration = (
            raw_velocity - previous_raw_velocity
            if previous_raw_velocity is not None
            else torch.zeros_like(raw)
        )
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
        filtered = raw - correction
        filtered_velocity = filtered - position if position is not None else torch.zeros_like(raw)
        filtered_acceleration = (
            filtered_velocity - velocity if velocity is not None else torch.zeros_like(raw)
        )

        corrected.append(filtered)
        raw_velocities.append(raw_velocity)
        corrected_velocities.append(filtered_velocity)
        raw_accelerations.append(raw_acceleration)
        corrected_accelerations.append(filtered_acceleration)
        corrections.append(correction)
        previous_raw = raw
        previous_raw_velocity = raw_velocity
        position = filtered
        velocity = filtered_velocity

    return (
        torch.stack(corrected),
        torch.stack(raw_velocities),
        torch.stack(corrected_velocities),
        torch.stack(raw_accelerations),
        torch.stack(corrected_accelerations),
        torch.stack(corrections),
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
    """Return a render-only corrected copy of ``normalized`` and next carry.

    ``normalized`` has shape ``(1, T, 3 + E)``. Rotation filtering happens
    in radians after de-normalization; expression filtering stays in z-score
    space. A mode change resets carry rather than applying stale velocity.
    """
    if normalized.ndim != 3 or normalized.shape[0] != 1 or normalized.shape[-1] < 4:
        raise ValueError(
            f"normalized motion must have shape (1, T, 3 + E), got {tuple(normalized.shape)}"
        )
    expression_coordinates = normalized.shape[-1] - 3
    options.validate(expression_coordinates=expression_coordinates)

    raw_rotation = normalized[0, :, :3] * so3_scale + so3_offset
    raw_expression = normalized[0, :, 3:]
    empty_rotation = _empty_rows(normalized, 3)
    empty_expression = _empty_rows(normalized, expression_coordinates)

    if options.mode == "none":
        return MotionStabilizationResult(
            normalized=normalized,
            state=None,
            state_reset=state is not None,
            raw_rotation=raw_rotation,
            corrected_rotation=raw_rotation,
            raw_rotation_velocity=empty_rotation,
            corrected_rotation_velocity=empty_rotation,
            raw_rotation_acceleration=empty_rotation,
            corrected_rotation_acceleration=empty_rotation,
            rotation_correction=torch.zeros_like(raw_rotation),
            raw_expression=raw_expression,
            corrected_expression=raw_expression,
            raw_expression_velocity=empty_expression,
            corrected_expression_velocity=empty_expression,
            raw_expression_acceleration=empty_expression,
            corrected_expression_acceleration=empty_expression,
            expression_correction=torch.zeros_like(raw_expression),
        )

    mode_code = _MODE_CODE[options.mode]
    state_reset = state is None or state.mode_code != mode_code
    if state_reset:
        state = MotionStabilizerState(mode_code=mode_code)

    assert state is not None
    corrected = normalized.clone()

    if options.rotation_enabled:
        (
            corrected_rotation,
            raw_rotation_velocity,
            corrected_rotation_velocity,
            raw_rotation_acceleration,
            corrected_rotation_acceleration,
            rotation_correction,
        ) = _guard_vector_norm(
            raw_rotation,
            previous_position=state.rotation_position,
            previous_velocity=state.rotation_velocity,
            previous_raw_position=state.rotation_raw_position,
            previous_raw_velocity=state.rotation_raw_velocity,
            acceleration_threshold=math.radians(options.rotation_acceleration_threshold_deg),
            max_correction=math.radians(options.rotation_max_correction_deg),
            strength=options.rotation_strength,
        )
        corrected[0, :, :3] = (corrected_rotation - so3_offset) / so3_scale
        next_rotation_position = corrected_rotation[-1].clone()
        next_rotation_velocity = corrected_rotation_velocity[-1].clone()
        next_rotation_raw_position = raw_rotation[-1].clone()
        next_rotation_raw_velocity = raw_rotation_velocity[-1].clone()
    else:
        corrected_rotation = raw_rotation
        raw_rotation_velocity = empty_rotation
        corrected_rotation_velocity = empty_rotation
        raw_rotation_acceleration = empty_rotation
        corrected_rotation_acceleration = empty_rotation
        rotation_correction = torch.zeros_like(raw_rotation)
        next_rotation_position = None
        next_rotation_velocity = None
        next_rotation_raw_position = None
        next_rotation_raw_velocity = None

    if options.expression_enabled:
        assert options.expression_coordinate_weights is not None
        weights = normalized.new_tensor(options.expression_coordinate_weights)
        (
            corrected_expression,
            raw_expression_velocity,
            corrected_expression_velocity,
            raw_expression_acceleration,
            corrected_expression_acceleration,
            expression_correction,
        ) = _guard_coordinates(
            raw_expression,
            previous_position=state.expression_position,
            previous_velocity=state.expression_velocity,
            previous_raw_position=state.expression_raw_position,
            previous_raw_velocity=state.expression_raw_velocity,
            acceleration_threshold=options.expression_acceleration_threshold_z,
            max_correction=options.expression_max_correction_z,
            strength=options.expression_strength,
            weights=weights,
        )
        corrected[0, :, 3:] = corrected_expression
        next_expression_position = corrected_expression[-1].clone()
        next_expression_velocity = corrected_expression_velocity[-1].clone()
        next_expression_raw_position = raw_expression[-1].clone()
        next_expression_raw_velocity = raw_expression_velocity[-1].clone()
    else:
        corrected_expression = raw_expression
        raw_expression_velocity = empty_expression
        corrected_expression_velocity = empty_expression
        raw_expression_acceleration = empty_expression
        corrected_expression_acceleration = empty_expression
        expression_correction = torch.zeros_like(raw_expression)
        next_expression_position = None
        next_expression_velocity = None
        next_expression_raw_position = None
        next_expression_raw_velocity = None

    next_state = MotionStabilizerState(
        mode_code=mode_code,
        rotation_position=next_rotation_position,
        rotation_velocity=next_rotation_velocity,
        rotation_raw_position=next_rotation_raw_position,
        rotation_raw_velocity=next_rotation_raw_velocity,
        expression_position=next_expression_position,
        expression_velocity=next_expression_velocity,
        expression_raw_position=next_expression_raw_position,
        expression_raw_velocity=next_expression_raw_velocity,
    )
    return MotionStabilizationResult(
        normalized=corrected,
        state=next_state,
        state_reset=state_reset,
        raw_rotation=raw_rotation,
        corrected_rotation=corrected_rotation,
        raw_rotation_velocity=raw_rotation_velocity,
        corrected_rotation_velocity=corrected_rotation_velocity,
        raw_rotation_acceleration=raw_rotation_acceleration,
        corrected_rotation_acceleration=corrected_rotation_acceleration,
        rotation_correction=rotation_correction,
        raw_expression=raw_expression,
        corrected_expression=corrected_expression,
        raw_expression_velocity=raw_expression_velocity,
        corrected_expression_velocity=corrected_expression_velocity,
        raw_expression_acceleration=raw_expression_acceleration,
        corrected_expression_acceleration=corrected_expression_acceleration,
        expression_correction=expression_correction,
    )


__all__ = [
    "MotionStabilizationResult",
    "MotionStabilizerState",
    "stabilize_normalized_motion",
]
