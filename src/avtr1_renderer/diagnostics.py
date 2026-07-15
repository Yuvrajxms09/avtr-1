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

import torch

from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.types import KPInfo, RenderOptions

if TYPE_CHECKING:
    from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame

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
        },
    )


def record_session(**metadata: Any) -> None:
    _emit("session", metadata)


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
            "source_pitch_rad": kp_info.pitch.detach().float().reshape(-1).tolist(),
            "source_yaw_rad": kp_info.yaw.detach().float().reshape(-1).tolist(),
            "source_roll_rad": kp_info.roll.detach().float().reshape(-1).tolist(),
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
            "expression_abs": _summary(motions.exp.abs()),
            "expression_frame_normalized_l2": _series(expression_magnitude),
            "expression_step_normalized_l2": _series(expression_steps),
        },
    )


def _keypoint_geometry(keypoints: torch.Tensor) -> dict[str, Any]:
    xy = keypoints[..., :2]
    centers = xy.mean(dim=1)
    extents = xy.amax(dim=1) - xy.amin(dim=1)
    radius = torch.linalg.vector_norm(xy - centers[:, None, :], dim=-1).mean(dim=1)
    center_steps = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)

    mean_radius = radius.mean().clamp_min(1e-8)
    radius_range_pct = ((radius.max() - radius.min()) / mean_radius * 100.0).item()
    return {
        "width": _series(extents[:, 0]),
        "height": _series(extents[:, 1]),
        "radius": _series(radius),
        "radius_range_pct": radius_range_pct,
        "center_step_l2": _series(center_steps),
    }


def record_keypoint_geometry(
    source: torch.Tensor,
    driving_raw: torch.Tensor,
    driving_stitched: torch.Tensor,
) -> None:
    if not enabled():
        return

    correction = torch.linalg.vector_norm(driving_stitched - driving_raw, dim=-1).mean(dim=1)
    affected = _LIPSYNC_KEYPOINT_INDICES
    _emit(
        "keypoint_geometry",
        {
            "source": _keypoint_geometry(source),
            "driving_raw": _keypoint_geometry(driving_raw),
            "driving_stitched": _keypoint_geometry(driving_stitched),
            "lipsync_keypoint_indices": affected,
            "driving_raw_lipsync_subset": _keypoint_geometry(driving_raw[:, affected]),
            "driving_stitched_lipsync_subset": _keypoint_geometry(driving_stitched[:, affected]),
            "stitch_correction_mean_l2": _series(correction),
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
    "record_options",
    "record_render_alpha",
    "record_session",
    "trace_scope",
]
