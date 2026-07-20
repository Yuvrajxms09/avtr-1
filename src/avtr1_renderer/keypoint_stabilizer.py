# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Stateful, opt-in stabilization after the LivePortrait stitch network."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import torch

from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.types import GeometryStabilizationOptions

_MOTION_FPS = 25.0


@dataclass(slots=True, frozen=True)
class KeypointStabilizerState:
    configuration_code: int
    stitch_position: torch.Tensor | None = None
    stitch_filter_input: torch.Tensor | None = None
    stitch_filter_derivative: torch.Tensor | None = None
    residual_position: torch.Tensor | None = None
    residual_filter_input: torch.Tensor | None = None
    residual_filter_derivative: torch.Tensor | None = None


@dataclass(slots=True, frozen=True)
class KeypointStabilizationResult:
    final: torch.Tensor
    state: KeypointStabilizerState | None
    state_reset: bool
    network_correction: torch.Tensor
    filtered_stitch_correction: torch.Tensor
    stitch_temporal_correction: torch.Tensor
    blended: torch.Tensor
    aligned_residual_raw: torch.Tensor
    aligned_residual_corrected: torch.Tensor
    post_stitch_correction: torch.Tensor
    similarity_scale: torch.Tensor
    keypoint_indices: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class _FilterResult:
    values: torch.Tensor
    correction: torch.Tensor
    position: torch.Tensor
    filter_input: torch.Tensor
    filter_derivative: torch.Tensor


def source_locked_keypoint_indices(keypoint_count: int) -> tuple[int, ...]:
    lipsync = {coordinate // 3 for coordinate in LIPSYNC_COORDS}
    return tuple(index for index in range(keypoint_count) if index not in lipsync)


def _configuration_code(
    options: GeometryStabilizationOptions,
    indices: tuple[int, ...],
) -> int:
    topology = bytearray(
        (
            int(options.stitch_temporal_filter != "none"),
            int(options.post_stitch_enabled),
            *indices,
        )
    )
    return zlib.crc32(topology)


def _lowpass_alpha(cutoff_hz: torch.Tensor) -> torch.Tensor:
    time_constant = 1.0 / (2.0 * math.pi * cutoff_hz)
    sample_period = 1.0 / _MOTION_FPS
    return 1.0 / (1.0 + time_constant / sample_period)


def _clamp_vector_rows(values: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.minimum(values.new_tensor(1.0), values.new_tensor(maximum) / norms)
    return values * scales


def _filter_vectors(
    values: torch.Tensor,
    *,
    previous_position: torch.Tensor | None,
    previous_filter_input: torch.Tensor | None,
    previous_filter_derivative: torch.Tensor | None,
    min_cutoff_hz: float,
    beta: float,
    derivative_cutoff_hz: float,
    max_correction: float,
) -> _FilterResult:
    outputs: list[torch.Tensor] = []
    corrections: list[torch.Tensor] = []
    position = previous_position
    filter_input = previous_filter_input
    filter_derivative = previous_filter_derivative
    derivative_alpha = _lowpass_alpha(values.new_tensor(derivative_cutoff_hz))

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
        target = value
        if position is not None:
            speed = torch.linalg.vector_norm(filtered_derivative, dim=-1, keepdim=True)
            cutoff = value.new_tensor(min_cutoff_hz) + beta * speed
            alpha = _lowpass_alpha(cutoff)
            target = alpha * value + (1.0 - alpha) * position
        correction = _clamp_vector_rows(value - target, max_correction)
        output = value - correction
        outputs.append(output)
        corrections.append(correction)
        position = output
        filter_input = value
        filter_derivative = filtered_derivative

    assert position is not None
    assert filter_input is not None
    assert filter_derivative is not None
    return _FilterResult(
        values=torch.stack(outputs),
        correction=torch.stack(corrections),
        position=position.clone(),
        filter_input=filter_input.clone(),
        filter_derivative=filter_derivative.clone(),
    )


def _similarity_to_reference(
    reference: torch.Tensor,
    current: torch.Tensor,
    indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current aligned to reference plus the forward transform."""
    ref = reference[indices]
    cur = current[indices]
    ref_center = ref.mean(dim=0)
    cur_center = cur.mean(dim=0)
    ref_zero = ref - ref_center
    cur_zero = cur - cur_center
    covariance = ref_zero.transpose(0, 1) @ cur_zero
    u, singular, vh = torch.linalg.svd(covariance)
    sign = torch.where(
        torch.det(u @ vh) < 0,
        singular.new_tensor(-1.0),
        singular.new_tensor(1.0),
    )
    diagonal = torch.ones(3, device=current.device, dtype=current.dtype)
    diagonal[-1] = sign
    rotation = u @ torch.diag(diagonal) @ vh
    variance = ref_zero.square().sum().clamp_min(1e-8)
    scale = (singular * diagonal).sum() / variance
    scale = scale.clamp_min(1e-6)
    translation = cur_center - scale * (ref_center @ rotation)
    aligned = ((current - translation) @ rotation.transpose(0, 1)) / scale
    return aligned, scale, rotation, translation


@torch.no_grad()
def stabilize_keypoints(
    source: torch.Tensor,
    driving_raw: torch.Tensor,
    driving_network: torch.Tensor,
    *,
    options: GeometryStabilizationOptions,
    state: KeypointStabilizerState | None,
) -> KeypointStabilizationResult:
    """Blend/filter stitch output while preserving default renderer behavior."""
    if source.shape != (1, driving_raw.shape[1], 3):
        raise ValueError(
            "source must have shape (1, K, 3) matching driving tensors, got "
            f"{tuple(source.shape)}"
        )
    if driving_raw.shape != driving_network.shape or driving_raw.ndim != 3:
        raise ValueError("driving_raw and driving_network must have matching (T, K, 3) shapes")
    keypoint_count = driving_raw.shape[1]
    options.validate(keypoint_count=keypoint_count)
    indices = options.post_stitch_keypoint_indices or source_locked_keypoint_indices(
        keypoint_count
    )
    protected = set(range(keypoint_count)) - set(source_locked_keypoint_indices(keypoint_count))
    unsafe = sorted(set(indices) & protected)
    if options.post_stitch_enabled and unsafe:
        raise ValueError(
            "post-stitch stabilization cannot target lipsync keypoints: "
            f"{unsafe}"
        )
    network_correction = driving_network - driving_raw
    zeros = torch.zeros_like(driving_raw)

    if not options.enabled:
        return KeypointStabilizationResult(
            final=driving_network,
            state=None,
            state_reset=state is not None,
            network_correction=network_correction,
            filtered_stitch_correction=network_correction,
            stitch_temporal_correction=zeros,
            blended=driving_network,
            aligned_residual_raw=zeros,
            aligned_residual_corrected=zeros,
            post_stitch_correction=zeros,
            similarity_scale=driving_raw.new_ones(driving_raw.shape[0]),
            keypoint_indices=indices,
        )

    configuration_code = _configuration_code(options, indices)
    state_reset = state is None or state.configuration_code != configuration_code
    if state_reset:
        state = KeypointStabilizerState(configuration_code=configuration_code)
    assert state is not None

    stitch_filter = None
    filtered_correction = network_correction
    stitch_temporal_correction = zeros
    if options.stitch_temporal_filter == "one_euro":
        stitch_filter = _filter_vectors(
            network_correction,
            previous_position=state.stitch_position,
            previous_filter_input=state.stitch_filter_input,
            previous_filter_derivative=state.stitch_filter_derivative,
            min_cutoff_hz=options.stitch_one_euro_min_cutoff_hz,
            beta=options.stitch_one_euro_beta,
            derivative_cutoff_hz=options.stitch_one_euro_derivative_cutoff_hz,
            max_correction=options.stitch_temporal_max_correction,
        )
        filtered_correction = stitch_filter.values
        stitch_temporal_correction = stitch_filter.correction
    blended = driving_raw + options.stitch_strength * filtered_correction

    residual_filter = None
    final = blended
    residual_raw = zeros
    residual_corrected = zeros
    post_correction = zeros
    scales: list[torch.Tensor] = []
    if options.post_stitch_enabled:
        aligned_frames: list[torch.Tensor] = []
        rotations: list[torch.Tensor] = []
        translations: list[torch.Tensor] = []
        reference = source[0]
        for frame in blended:
            aligned, scale, rotation, translation = _similarity_to_reference(
                reference,
                frame,
                indices,
            )
            aligned_frames.append(aligned)
            scales.append(scale)
            rotations.append(rotation)
            translations.append(translation)
        aligned_all = torch.stack(aligned_frames)
        residual_raw = aligned_all - reference
        selected = residual_raw[:, indices]
        residual_filter = _filter_vectors(
            selected,
            previous_position=state.residual_position,
            previous_filter_input=state.residual_filter_input,
            previous_filter_derivative=state.residual_filter_derivative,
            min_cutoff_hz=options.post_stitch_one_euro_min_cutoff_hz,
            beta=options.post_stitch_one_euro_beta,
            derivative_cutoff_hz=options.post_stitch_one_euro_derivative_cutoff_hz,
            max_correction=options.post_stitch_max_correction,
        )
        residual_corrected = residual_raw.clone()
        residual_corrected[:, indices] = residual_filter.values
        restored: list[torch.Tensor] = []
        for frame_index, residual in enumerate(residual_corrected):
            restored.append(
                scales[frame_index]
                * ((reference + residual) @ rotations[frame_index])
                + translations[frame_index]
            )
        proposed = torch.stack(restored)
        post_correction = _clamp_vector_rows(
            options.post_stitch_strength * (blended - proposed),
            options.post_stitch_max_correction,
        )
        # The reviewed mask is a hard safety boundary.  Lipsync and all
        # unselected keypoints remain byte-for-byte equal to the stitch blend.
        masked_correction = torch.zeros_like(post_correction)
        masked_correction[:, indices] = post_correction[:, indices]
        post_correction = masked_correction
        final = blended - post_correction

    next_state = KeypointStabilizerState(
        configuration_code=configuration_code,
        stitch_position=(stitch_filter.position if stitch_filter else None),
        stitch_filter_input=(stitch_filter.filter_input if stitch_filter else None),
        stitch_filter_derivative=(
            stitch_filter.filter_derivative if stitch_filter else None
        ),
        residual_position=(residual_filter.position if residual_filter else None),
        residual_filter_input=(residual_filter.filter_input if residual_filter else None),
        residual_filter_derivative=(
            residual_filter.filter_derivative if residual_filter else None
        ),
    )
    return KeypointStabilizationResult(
        final=final,
        state=next_state,
        state_reset=state_reset,
        network_correction=network_correction,
        filtered_stitch_correction=filtered_correction,
        stitch_temporal_correction=stitch_temporal_correction,
        blended=blended,
        aligned_residual_raw=residual_raw,
        aligned_residual_corrected=residual_corrected,
        post_stitch_correction=post_correction,
        similarity_scale=(
            torch.stack(scales) if scales else driving_raw.new_ones(driving_raw.shape[0])
        ),
        keypoint_indices=indices,
    )


__all__ = [
    "KeypointStabilizationResult",
    "KeypointStabilizerState",
    "source_locked_keypoint_indices",
    "stabilize_keypoints",
]
