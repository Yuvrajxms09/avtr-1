# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Opt-in motion diagnostics for tracing temporal and facial geometry changes.

The helpers in this module are intentionally dormant unless the
``avtr1_renderer.motion_debug`` logger is enabled at ``DEBUG``. Metrics call
``Tensor.item()`` and therefore synchronize CUDA; they are for diagnosis, not
production throughput measurements.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import roma
import torch

from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.types import (
    GeometryStabilizationOptions,
    KPInfo,
    MotionStabilizationOptions,
    RenderOptions,
    TurnAwareGuidanceOptions,
)

if TYPE_CHECKING:
    from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
    from avtr1_renderer.keypoint_stabilizer import KeypointStabilizationResult
    from avtr1_renderer.motion_stabilizer import MotionStabilizationResult
    from avtr1_renderer.turn_guidance import TurnGuidanceResult

LOGGER_NAME = "avtr1_renderer.motion_debug"

_LOG = logging.getLogger(LOGGER_NAME)
_TRACE_ID: ContextVar[str | None] = ContextVar("avtr1_motion_trace_id", default=None)
_LIPSYNC_KEYPOINT_INDICES = sorted({coordinate // 3 for coordinate in LIPSYNC_COORDS})


def enabled() -> bool:
    return _LOG.isEnabledFor(logging.DEBUG)


def new_trace_id() -> str | None:
    return uuid.uuid4().hex[:12] if enabled() else None


@contextmanager
def trace_scope(trace_id: str | None) -> Iterator[None]:
    if trace_id is None:
        yield
        return
    token = _TRACE_ID.set(trace_id)
    try:
        yield
    finally:
        _TRACE_ID.reset(token)


def _emit(stage: str, metrics: dict[str, Any]) -> None:
    if not enabled():
        return
    payload = {"stage": stage, "trace_id": _TRACE_ID.get(), **metrics}
    _LOG.debug(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": values.numel(),
        "min": values.min().item(),
        "max": values.max().item(),
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
    }


def _series(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().reshape(-1)
    return {**_summary(values), "values": values.tolist()}


def record_options(options: RenderOptions) -> None:
    _emit(
        "options",
        {
            "cfg_self_audio": options.cfg_self_audio,
            "cfg_other_audio": options.cfg_other_audio,
            "cfg_kp": options.cfg_kp,
            "noise_alpha": options.noise_alpha,
            "noise_trunc_z": options.noise_trunc_z,
            "stream_frames": options.stream_frames,
            "motion_stabilization": {
                "mode": options.stabilization.mode,
                "rotation_spike_guard": options.stabilization.rotation_spike_guard,
                "rotation_acceleration_threshold_deg": (
                    options.stabilization.rotation_acceleration_threshold_deg
                ),
                "rotation_max_correction_deg": (options.stabilization.rotation_max_correction_deg),
                "rotation_strength": options.stabilization.rotation_strength,
                "rotation_temporal_filter": options.stabilization.rotation_temporal_filter,
                "rotation_one_euro_min_cutoff_hz": (
                    options.stabilization.rotation_one_euro_min_cutoff_hz
                ),
                "rotation_one_euro_beta": options.stabilization.rotation_one_euro_beta,
                "rotation_one_euro_derivative_cutoff_hz": (
                    options.stabilization.rotation_one_euro_derivative_cutoff_hz
                ),
                "rotation_temporal_max_correction_deg": (
                    options.stabilization.rotation_temporal_max_correction_deg
                ),
                "rotation_max_acceleration_deg": (
                    options.stabilization.rotation_max_acceleration_deg
                ),
                "rotation_max_jerk_deg": options.stabilization.rotation_max_jerk_deg,
                "expression_spike_guard": options.stabilization.expression_spike_guard,
                "expression_acceleration_threshold_z": (
                    options.stabilization.expression_acceleration_threshold_z
                ),
                "expression_max_correction_z": (options.stabilization.expression_max_correction_z),
                "expression_strength": options.stabilization.expression_strength,
                "expression_temporal_filter": options.stabilization.expression_temporal_filter,
                "expression_one_euro_min_cutoff_hz": (
                    options.stabilization.expression_one_euro_min_cutoff_hz
                ),
                "expression_one_euro_beta": options.stabilization.expression_one_euro_beta,
                "expression_one_euro_derivative_cutoff_hz": (
                    options.stabilization.expression_one_euro_derivative_cutoff_hz
                ),
                "expression_temporal_max_correction_z": (
                    options.stabilization.expression_temporal_max_correction_z
                ),
                "expression_max_acceleration_z": (
                    options.stabilization.expression_max_acceleration_z
                ),
                "expression_max_jerk_z": options.stabilization.expression_max_jerk_z,
                "expression_active_coordinates": (
                    None
                    if options.stabilization.expression_coordinate_weights is None
                    else [
                        index
                        for index, weight in enumerate(
                            options.stabilization.expression_coordinate_weights
                        )
                        if weight > 0
                    ]
                ),
            },
            "geometry_stabilization": {
                "enabled": options.geometry.enabled,
                "stitch_strength": options.geometry.stitch_strength,
                "stitch_temporal_filter": options.geometry.stitch_temporal_filter,
                "stitch_one_euro_min_cutoff_hz": (
                    options.geometry.stitch_one_euro_min_cutoff_hz
                ),
                "stitch_one_euro_beta": options.geometry.stitch_one_euro_beta,
                "stitch_one_euro_derivative_cutoff_hz": (
                    options.geometry.stitch_one_euro_derivative_cutoff_hz
                ),
                "stitch_temporal_max_correction": (
                    options.geometry.stitch_temporal_max_correction
                ),
                "post_stitch_enabled": options.geometry.post_stitch_enabled,
                "post_stitch_one_euro_min_cutoff_hz": (
                    options.geometry.post_stitch_one_euro_min_cutoff_hz
                ),
                "post_stitch_one_euro_beta": options.geometry.post_stitch_one_euro_beta,
                "post_stitch_one_euro_derivative_cutoff_hz": (
                    options.geometry.post_stitch_one_euro_derivative_cutoff_hz
                ),
                "post_stitch_strength": options.geometry.post_stitch_strength,
                "post_stitch_max_correction": options.geometry.post_stitch_max_correction,
                "post_stitch_keypoint_indices": (
                    None
                    if options.geometry.post_stitch_keypoint_indices is None
                    else list(options.geometry.post_stitch_keypoint_indices)
                ),
            },
            "turn_guidance": {
                "mode": options.turn_guidance.mode,
                "speaking_cfg_other_audio": (
                    options.turn_guidance.speaking_cfg_other_audio
                ),
                "listening_cfg_other_audio": (
                    options.turn_guidance.listening_cfg_other_audio
                ),
                "speech_rms_on": options.turn_guidance.speech_rms_on,
                "speech_rms_off": options.turn_guidance.speech_rms_off,
                "listen_rms_on": options.turn_guidance.listen_rms_on,
                "listen_rms_off": options.turn_guidance.listen_rms_off,
                "hysteresis_chunks": options.turn_guidance.hysteresis_chunks,
                "minimum_state_chunks": options.turn_guidance.minimum_state_chunks,
                "transition_chunks": options.turn_guidance.transition_chunks,
            },
        },
    )


def record_session(**metadata: Any) -> None:
    _emit("session", metadata)


def record_motion_trajectory(**metadata: Any) -> None:
    _emit("motion_trajectory", metadata)


def record_stage_artifact(**metadata: Any) -> None:
    _emit("stage_artifact", metadata)


def record_turn_guidance(
    options: TurnAwareGuidanceOptions,
    result: TurnGuidanceResult,
) -> None:
    _emit(
        "turn_guidance",
        {
            "mode": options.mode,
            "observed_turn": result.observed_turn,
            "stable_turn": result.stable_turn,
            "state_changed": result.state_changed,
            "speech_rms": result.speech_rms,
            "listen_rms": result.listen_rms,
            "effective_cfg_other_audio": result.effective_cfg_other_audio,
            "pending_state_code": (
                None if result.state is None else result.state.pending_state_code
            ),
            "pending_chunks": (
                None if result.state is None else result.state.pending_chunks
            ),
            "state_age_chunks": (
                None if result.state is None else result.state.state_age_chunks
            ),
        },
    )


def record_avatar_registration(
    *,
    avatar_id: str,
    kp_info: KPInfo,
    crop_scale: float,
    crop_vx_ratio: float,
    crop_vy_ratio: float,
    crop_rotation: bool,
    source_height: int,
    source_width: int,
    no_matting: bool,
) -> None:
    if not enabled():
        return
    _emit(
        "avatar_registration",
        {
            "avatar_id": avatar_id,
            "source_height": source_height,
            "source_width": source_width,
            "no_matting": no_matting,
            "crop_scale": crop_scale,
            "crop_vx_ratio": crop_vx_ratio,
            "crop_vy_ratio": crop_vy_ratio,
            "crop_rotation": crop_rotation,
            "source_motion_scale": kp_info.scale.detach().float().reshape(-1).tolist(),
            "source_translation": kp_info.t.detach().float().reshape(-1).tolist(),
            "source_pitch_deg": kp_info.pitch.detach().float().reshape(-1).tolist(),
            "source_yaw_deg": kp_info.yaw.detach().float().reshape(-1).tolist(),
            "source_roll_deg": kp_info.roll.detach().float().reshape(-1).tolist(),
        },
    )


def record_audio_chunk(speech: torch.Tensor, listen: torch.Tensor) -> None:
    if not enabled():
        return

    def audio_metrics(audio: torch.Tensor) -> dict[str, float]:
        audio = audio.detach().float()
        return {
            "rms": torch.sqrt(torch.mean(audio.square())).item(),
            "abs_peak": audio.abs().max().item(),
        }

    _emit(
        "audio_chunk",
        {
            "speech": audio_metrics(speech),
            "listen": audio_metrics(listen),
        },
    )


def _rotation_angle_degrees(rotations: torch.Tensor) -> torch.Tensor:
    traces = rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((traces - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / math.pi)


def _rotation_step_degrees(rotations: torch.Tensor) -> torch.Tensor:
    if rotations.shape[0] < 2:
        return rotations.new_empty(0)
    relative = rotations[:-1].transpose(-1, -2) @ rotations[1:]
    return _rotation_angle_degrees(relative)


def record_motion_prediction(
    normalized: torch.Tensor,
    previous_normalized: torch.Tensor,
    motions: MotionFrame,
) -> None:
    if not enabled():
        return

    current = normalized[0]
    previous = previous_normalized.reshape(-1)
    head = current[:, :3]
    expression = current[:, 3:]
    expression_magnitude = torch.linalg.vector_norm(expression, dim=-1)
    expression_steps = torch.linalg.vector_norm(expression[1:] - expression[:-1], dim=-1)

    _emit(
        "motion_prediction",
        {
            "chunk_boundary_head_normalized_l2": torch.linalg.vector_norm(
                head[0] - previous[:3]
            ).item(),
            "chunk_boundary_expression_normalized_l2": torch.linalg.vector_norm(
                expression[0] - previous[3:]
            ).item(),
            "head_rotation_angle_deg": _series(_rotation_angle_degrees(motions.R)),
            "head_rotation_step_deg": _series(_rotation_step_degrees(motions.R)),
            "head_rotvec_rad": roma.rotmat_to_rotvec(motions.R).detach().float().tolist(),
            "expression_normalized": expression.detach().float().tolist(),
            "expression_abs": _summary(motions.exp.abs()),
            "expression_frame_normalized_l2": _series(expression_magnitude),
            "expression_step_normalized_l2": _series(expression_steps),
        },
    )


def _row_norms(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_empty(0)
    return torch.linalg.vector_norm(values, dim=-1)


def record_motion_stabilization(
    options: MotionStabilizationOptions,
    result: MotionStabilizationResult,
    *,
    elapsed_ms: float | None = None,
) -> None:
    if not enabled():
        return

    radians_to_degrees = 180.0 / math.pi
    rotation_correction_deg = _row_norms(result.rotation_correction) * radians_to_degrees
    rotation_spike_correction_deg = (
        _row_norms(result.rotation_spike_correction) * radians_to_degrees
    )
    rotation_temporal_correction_deg = (
        _row_norms(result.rotation_temporal_correction) * radians_to_degrees
    )
    expression_abs_correction = result.expression_correction.abs()
    expression_coordinate_max = (
        expression_abs_correction.amax(dim=0)
        if expression_abs_correction.numel()
        else expression_abs_correction.new_empty(0)
    )
    expression_interventions = expression_coordinate_max > 1e-6
    rotation_intervention_mask = rotation_correction_deg > 1e-6
    rotation_spike_intervention_mask = rotation_spike_correction_deg > 1e-6
    rotation_temporal_intervention_mask = rotation_temporal_correction_deg > 1e-6
    expression_intervention_mask = expression_abs_correction > 1e-6
    expression_frame_interventions = expression_intervention_mask.any(dim=-1)
    expression_correction_ratio = (
        _row_norms(result.expression_correction)
        / _row_norms(result.raw_expression_velocity).clamp_min(1e-8)
        if result.raw_expression_velocity.numel()
        else result.raw_expression_velocity.new_empty(0)
    )

    _emit(
        "motion_stabilization",
        {
            "mode": options.mode,
            "state_reset": result.state_reset,
            "elapsed_ms": elapsed_ms,
            "rotation": {
                "raw_rotvec_rad": result.raw_rotation.detach().float().tolist(),
                "spike_corrected_rotvec_rad": (
                    result.spike_corrected_rotation.detach().float().tolist()
                ),
                "corrected_rotvec_rad": result.corrected_rotation.detach().float().tolist(),
                "raw_velocity_deg": _series(
                    _row_norms(result.raw_rotation_velocity) * radians_to_degrees
                ),
                "corrected_velocity_deg": _series(
                    _row_norms(result.corrected_rotation_velocity) * radians_to_degrees
                ),
                "raw_acceleration_deg": _series(
                    _row_norms(result.raw_rotation_acceleration) * radians_to_degrees
                ),
                "corrected_acceleration_deg": _series(
                    _row_norms(result.corrected_rotation_acceleration) * radians_to_degrees
                ),
                "correction_deg": _series(rotation_correction_deg),
                "spike_correction_deg": _series(rotation_spike_correction_deg),
                "lowpass_requested_correction_deg": _series(
                    _row_norms(result.rotation_lowpass_requested_correction) * radians_to_degrees
                ),
                "kinematic_requested_correction_deg": _series(
                    _row_norms(result.rotation_kinematic_requested_correction) * radians_to_degrees
                ),
                "temporal_correction_deg": _series(rotation_temporal_correction_deg),
                "intervention_frames": int(rotation_intervention_mask.sum().item()),
                "spike_intervention_frames": int(rotation_spike_intervention_mask.sum().item()),
                "temporal_intervention_frames": int(
                    rotation_temporal_intervention_mask.sum().item()
                ),
                "intervention_frame_indices": rotation_intervention_mask.nonzero(as_tuple=False)
                .flatten()
                .tolist(),
            },
            "expression": {
                "raw_normalized": result.raw_expression.detach().float().tolist(),
                "spike_corrected_normalized": (
                    result.spike_corrected_expression.detach().float().tolist()
                ),
                "corrected_normalized": (result.corrected_expression.detach().float().tolist()),
                "raw_velocity_l2": _series(_row_norms(result.raw_expression_velocity)),
                "corrected_velocity_l2": _series(_row_norms(result.corrected_expression_velocity)),
                "raw_acceleration_l2": _series(_row_norms(result.raw_expression_acceleration)),
                "corrected_acceleration_l2": _series(
                    _row_norms(result.corrected_expression_acceleration)
                ),
                "correction_l2": _series(_row_norms(result.expression_correction)),
                "spike_correction_l2": _series(_row_norms(result.expression_spike_correction)),
                "lowpass_requested_correction_l2": _series(
                    _row_norms(result.expression_lowpass_requested_correction)
                ),
                "kinematic_requested_correction_l2": _series(
                    _row_norms(result.expression_kinematic_requested_correction)
                ),
                "temporal_correction_l2": _series(
                    _row_norms(result.expression_temporal_correction)
                ),
                "coordinate_max_abs_correction_z": _series(expression_coordinate_max),
                "correction_to_raw_velocity_ratio": _series(expression_correction_ratio),
                "intervention_frames": int(expression_frame_interventions.sum().item()),
                "intervention_frame_indices": expression_frame_interventions.nonzero(as_tuple=False)
                .flatten()
                .tolist(),
                "intervention_coordinates": expression_interventions.nonzero(as_tuple=False)
                .flatten()
                .tolist(),
                "intervention_coordinates_by_frame": [
                    coordinates.nonzero(as_tuple=False).flatten().tolist()
                    for coordinates in expression_intervention_mask
                ],
            },
        },
    )


def _keypoint_geometry(keypoints: torch.Tensor) -> dict[str, Any]:
    xy = keypoints[..., :2]
    centers = xy.mean(dim=1)
    extents = xy.amax(dim=1) - xy.amin(dim=1)
    radius = torch.linalg.vector_norm(xy - centers[:, None, :], dim=-1).mean(dim=1)
    center_steps = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)
    keypoint_steps = torch.linalg.vector_norm(keypoints[1:] - keypoints[:-1], dim=-1)

    mean_radius = radius.mean().clamp_min(1e-8)
    radius_range_pct = ((radius.max() - radius.min()) / mean_radius * 100.0).item()
    return {
        "width": _series(extents[:, 0]),
        "height": _series(extents[:, 1]),
        "radius": _series(radius),
        "radius_range_pct": radius_range_pct,
        "center_step_l2": _series(center_steps),
        "coordinates_xyz": keypoints.detach().float().tolist(),
        "per_keypoint_step_l2": {
            "values": keypoint_steps.detach().float().transpose(0, 1).tolist(),
            "p95": (
                torch.quantile(keypoint_steps.float(), 0.95, dim=0).tolist()
                if keypoint_steps.numel()
                else []
            ),
            "max": (keypoint_steps.float().amax(dim=0).tolist() if keypoint_steps.numel() else []),
        },
    }


def _keypoint_correction_kinematics(values: torch.Tensor) -> dict[str, Any]:
    velocity = values[1:] - values[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]

    def frame_mean_l2(rows: torch.Tensor) -> torch.Tensor:
        if rows.numel() == 0:
            return rows.new_empty(0)
        return torch.linalg.vector_norm(rows, dim=-1).mean(dim=-1)

    return {
        "magnitude_mean_l2": _series(frame_mean_l2(values)),
        "velocity_mean_l2": _series(frame_mean_l2(velocity)),
        "acceleration_mean_l2": _series(frame_mean_l2(acceleration)),
        "jerk_mean_l2": _series(frame_mean_l2(jerk)),
    }


def record_keypoint_geometry(
    source: torch.Tensor,
    driving_raw: torch.Tensor,
    driving_network: torch.Tensor,
    driving_final: torch.Tensor,
    *,
    stabilization: KeypointStabilizationResult,
    options: GeometryStabilizationOptions,
) -> None:
    if not enabled():
        return

    correction = torch.linalg.vector_norm(driving_network - driving_raw, dim=-1).mean(dim=1)
    per_keypoint_correction = torch.linalg.vector_norm(
        driving_network - driving_raw,
        dim=-1,
    )
    final_correction = torch.linalg.vector_norm(driving_final - driving_raw, dim=-1).mean(dim=1)
    stitch_temporal = torch.linalg.vector_norm(
        stabilization.stitch_temporal_correction,
        dim=-1,
    )
    post_stitch = torch.linalg.vector_norm(stabilization.post_stitch_correction, dim=-1)
    stitch_cap_hits = stitch_temporal >= options.stitch_temporal_max_correction - 1e-8
    post_cap_hits = post_stitch >= options.post_stitch_max_correction - 1e-8
    affected = _LIPSYNC_KEYPOINT_INDICES
    source_locked = [index for index in range(source.shape[1]) if index not in affected]
    _emit(
        "keypoint_geometry",
        {
            "source": _keypoint_geometry(source),
            "driving_raw": _keypoint_geometry(driving_raw),
            "driving_network": _keypoint_geometry(driving_network),
            # Keep the established field as the geometry actually sent to warp.
            "driving_stitched": _keypoint_geometry(driving_final),
            "driving_final": _keypoint_geometry(driving_final),
            "lipsync_keypoint_indices": affected,
            "source_locked_keypoint_indices": source_locked,
            "driving_raw_lipsync_subset": _keypoint_geometry(driving_raw[:, affected]),
            "driving_network_lipsync_subset": _keypoint_geometry(
                driving_network[:, affected]
            ),
            "driving_stitched_lipsync_subset": _keypoint_geometry(driving_final[:, affected]),
            "driving_raw_source_locked_subset": _keypoint_geometry(driving_raw[:, source_locked]),
            "driving_stitched_source_locked_subset": _keypoint_geometry(
                driving_final[:, source_locked]
            ),
            "stitch_correction_mean_l2": _series(correction),
            "final_correction_mean_l2": _series(final_correction),
            "stitch_correction_per_keypoint_l2": {
                "values": per_keypoint_correction.detach().float().transpose(0, 1).tolist(),
                "p95": torch.quantile(
                    per_keypoint_correction.float(),
                    0.95,
                    dim=0,
                ).tolist(),
                "max": per_keypoint_correction.float().amax(dim=0).tolist(),
            },
            "geometry_stabilization": {
                "enabled": options.enabled,
                "state_reset": stabilization.state_reset,
                "keypoint_indices": list(stabilization.keypoint_indices),
                "similarity_scale": _series(stabilization.similarity_scale),
                "stitch_temporal_correction_per_keypoint_l2": {
                    "values": stitch_temporal.detach().float().transpose(0, 1).tolist(),
                    "p95": torch.quantile(stitch_temporal.float(), 0.95, dim=0).tolist(),
                    "max": stitch_temporal.float().amax(dim=0).tolist(),
                },
                "stitch_network_correction_kinematics": (
                    _keypoint_correction_kinematics(stabilization.network_correction)
                ),
                "filtered_stitch_correction_kinematics": (
                    _keypoint_correction_kinematics(
                        stabilization.filtered_stitch_correction
                    )
                ),
                "post_stitch_correction_kinematics": (
                    _keypoint_correction_kinematics(stabilization.post_stitch_correction)
                ),
                "post_stitch_correction_per_keypoint_l2": {
                    "values": post_stitch.detach().float().transpose(0, 1).tolist(),
                    "p95": torch.quantile(post_stitch.float(), 0.95, dim=0).tolist(),
                    "max": post_stitch.float().amax(dim=0).tolist(),
                },
                "aligned_residual_raw_l2": _series(
                    _row_norms(stabilization.aligned_residual_raw.reshape(-1, 3))
                ),
                "aligned_residual_corrected_l2": _series(
                    _row_norms(stabilization.aligned_residual_corrected.reshape(-1, 3))
                ),
                "stitch_temporal_intervention_frames": int(
                    (stitch_temporal > 1e-8).any(dim=1).sum().item()
                ),
                "post_stitch_intervention_frames": int(
                    (post_stitch > 1e-8).any(dim=1).sum().item()
                ),
                "stitch_temporal_cap_hits": int(stitch_cap_hits.sum().item()),
                "post_stitch_cap_hits": int(post_cap_hits.sum().item()),
            },
        },
    )


def alpha_frame_metrics(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame mean alpha and fraction above the 0.5 threshold."""
    dims = tuple(range(1, alpha.ndim))
    return alpha.mean(dim=dims), (alpha > 0.5).float().mean(dim=dims)


def record_render_alpha(
    alpha_mean: torch.Tensor,
    alpha_coverage: torch.Tensor,
    *,
    no_matting: bool,
) -> None:
    if not enabled():
        return
    _emit(
        "render_alpha",
        {
            "mode": "opaque_source" if no_matting else "modnet",
            "alpha_mean": _series(alpha_mean),
            "alpha_coverage_over_0_5": _series(alpha_coverage),
        },
    )


__all__ = [
    "LOGGER_NAME",
    "alpha_frame_metrics",
    "enabled",
    "new_trace_id",
    "record_audio_chunk",
    "record_avatar_registration",
    "record_keypoint_geometry",
    "record_motion_prediction",
    "record_motion_stabilization",
    "record_motion_trajectory",
    "record_options",
    "record_render_alpha",
    "record_session",
    "record_stage_artifact",
    "record_turn_guidance",
    "trace_scope",
]
