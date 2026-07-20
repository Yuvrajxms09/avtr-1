# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Public dataclasses for the streaming face animation pipeline.

These types form the contract between the orchestrator (`Model.process_chunk`)
and its caller.

Shape philosophy:
- `Avatar` (defined alongside the loader in ``avatar_loader``) is
  **immutable** and built once from a portrait. Carries every
  per-portrait tensor the inner loop needs.
- A "registry" is just a plain ``dict[str, Avatar]`` held wherever it makes
  sense (the orchestrator, the API layer, a process-wide singleton). The
  loader is the producer; the registry is just data.
- `Chunk` is per-call audio (speech + listen tracks).
- The per-session state blob is opaque -- its shape is the motion
  generator's choice (see ``AVTR1State`` in
  ``avtr1_motion_generator``).
- `Frame` is one rendered RGB image, yielded one at a time so the API can
  stream without buffering.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from avtr1_renderer.components.pixel_format import PixelFormat


@dataclass(slots=True, frozen=True)
class KPInfo:
    """Source-side keypoint / pose / expression bundle.

    All tensors are torch CUDA float32. Shapes match the LivePortrait
    convention used by the reference's ``MotionExtractor`` output.
    """

    kp: torch.Tensor  # (1, 21, 3)  canonical 3D keypoints
    exp: torch.Tensor  # (1, 21, 3)  expression deltas
    scale: torch.Tensor  # (1, 1)      per-portrait scale
    t: torch.Tensor  # (1, 3)      translation
    pitch: torch.Tensor  # (1, 1)      pitch in degrees
    yaw: torch.Tensor  # (1, 1)      yaw in degrees
    roll: torch.Tensor  # (1, 1)      roll in degrees
    R: torch.Tensor  # (1, 3, 3)   rotation matrix derived from p/y/r


@dataclass(slots=True, frozen=True)
class Chunk:
    """One request's worth of audio.

    Both tracks are mandatory:
    - ``audio_speech`` drives lip-sync / talking motion.
    - ``audio_listen`` drives idle / listening motion (parallel pass).

    Both are float32 in [-1, 1] at 16 kHz mono. Length must be
    ``sum(chunksize[1:]) * frame_len + shift = (5+5)*640 + 80 = 6480``
    samples with the default config; the model raises on mismatch.
    """

    audio_speech: np.ndarray  # float32, shape (N,)
    audio_listen: np.ndarray  # float32, shape (N,)

    def __post_init__(self) -> None:
        for name, arr in (("audio_speech", self.audio_speech), ("audio_listen", self.audio_listen)):
            if arr.ndim != 1:
                raise ValueError(f"Chunk.{name} must be 1-D, got shape {arr.shape}")
            if arr.dtype != np.float32:
                raise ValueError(f"Chunk.{name} must be float32, got {arr.dtype}")
        if len(self.audio_speech) != len(self.audio_listen):
            raise ValueError(
                f"Chunk audio tracks must have equal length "
                f"(audio_speech={len(self.audio_speech)}, audio_listen={len(self.audio_listen)})"
            )


@dataclass(slots=True, frozen=True)
class Frame:
    """One rendered frame, materialised on host in the configured pixel format.

    Layout depends on ``format``:

    - ``"yuv_i420"`` (default): ``data`` is ``(3 H // 2, W)`` uint8, planar
      I420. The renderer composites the head over the avatar's bg using the
      predicted alpha matte before packing, so no separate alpha plane is
      shipped. 1.5 bpp.
    - ``"yuv_i420_stacked_alpha"``: ``(3 H, W)`` uint8 packed
      ``(Y, AY, U, AU, V, AV)``. The buffer is also a valid I420 frame at
      ``(1.5 H, W)`` -- useful for shipping alpha through a stock H.264 /
      VP9 / AV1 pipeline so downstream callers can recomposite over a
      different bg. 3 bpp.

    ``height`` / ``width`` are the *original* color-frame dimensions, not
    the packed buffer's; consumers use them to unpack the planes.

    The renderer pipeline materialises each frame on CPU before yielding,
    so callers can stream without holding references to GPU buffers.
    """

    data: np.ndarray
    format: Literal["yuv_i420", "yuv_i420_stacked_alpha"] = "yuv_i420"
    height: int = 0
    width: int = 0


FrameIterator = Iterator[Frame]

MotionStabilizationMode = Literal["none", "rotation", "expression", "both"]
TemporalFilterMode = Literal["none", "one_euro"]
TurnGuidanceMode = Literal[
    "disabled",
    "auto",
    "speaking",
    "listening",
    "overlap",
    "silence",
]


@dataclass(slots=True, frozen=True)
class MotionStabilizationOptions:
    """Experimental temporal filtering applied to normalized motion output.

    Each enabled channel can run three independent render-only layers:
    isolated spike suppression, an adaptive One Euro low-pass filter, and
    bounded acceleration/jerk. Disabled defaults are deliberately
    output-preserving. Expression filtering operates in the model's
    z-score-normalized space and is blended by an explicit 39-coordinate
    profile so unreviewed mouth motion is never changed implicitly.
    """

    mode: MotionStabilizationMode = "none"

    rotation_spike_guard: bool = True
    rotation_acceleration_threshold_deg: float = 0.75
    rotation_max_correction_deg: float = 1.5
    rotation_strength: float = 1.0
    rotation_temporal_filter: TemporalFilterMode = "none"
    rotation_one_euro_min_cutoff_hz: float = 2.0
    rotation_one_euro_beta: float = 0.1
    rotation_one_euro_derivative_cutoff_hz: float = 1.0
    rotation_temporal_max_correction_deg: float = 0.5
    rotation_max_acceleration_deg: float = 0.0
    rotation_max_jerk_deg: float = 0.0

    expression_spike_guard: bool = True
    expression_acceleration_threshold_z: float = 1.5
    expression_max_correction_z: float = 2.0
    expression_strength: float = 1.0
    expression_temporal_filter: TemporalFilterMode = "none"
    expression_one_euro_min_cutoff_hz: float = 3.0
    expression_one_euro_beta: float = 0.5
    expression_one_euro_derivative_cutoff_hz: float = 1.0
    expression_temporal_max_correction_z: float = 0.25
    expression_max_acceleration_z: float = 0.0
    expression_max_jerk_z: float = 0.0
    expression_coordinate_weights: tuple[float, ...] | None = None

    @property
    def rotation_enabled(self) -> bool:
        return self.mode in {"rotation", "both"}

    @property
    def expression_enabled(self) -> bool:
        return self.mode in {"expression", "both"}

    @property
    def rotation_kinematic_limiter_enabled(self) -> bool:
        return self.rotation_max_acceleration_deg > 0 or self.rotation_max_jerk_deg > 0

    @property
    def expression_kinematic_limiter_enabled(self) -> bool:
        return self.expression_max_acceleration_z > 0 or self.expression_max_jerk_z > 0

    def validate(self, *, expression_coordinates: int) -> None:
        if self.mode not in {"none", "rotation", "expression", "both"}:
            raise ValueError(f"Unknown motion stabilization mode: {self.mode!r}")
        for name, value in (
            ("rotation_temporal_filter", self.rotation_temporal_filter),
            ("expression_temporal_filter", self.expression_temporal_filter),
        ):
            if value not in {"none", "one_euro"}:
                raise ValueError(f"Unknown {name}: {value!r}")
        for name, value in (
            ("rotation_acceleration_threshold_deg", self.rotation_acceleration_threshold_deg),
            ("rotation_max_correction_deg", self.rotation_max_correction_deg),
            ("expression_acceleration_threshold_z", self.expression_acceleration_threshold_z),
            ("expression_max_correction_z", self.expression_max_correction_z),
            ("rotation_one_euro_min_cutoff_hz", self.rotation_one_euro_min_cutoff_hz),
            (
                "rotation_one_euro_derivative_cutoff_hz",
                self.rotation_one_euro_derivative_cutoff_hz,
            ),
            ("rotation_temporal_max_correction_deg", self.rotation_temporal_max_correction_deg),
            ("expression_one_euro_min_cutoff_hz", self.expression_one_euro_min_cutoff_hz),
            (
                "expression_one_euro_derivative_cutoff_hz",
                self.expression_one_euro_derivative_cutoff_hz,
            ),
            ("expression_temporal_max_correction_z", self.expression_temporal_max_correction_z),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero, got {value}")
        for name, value in (
            ("rotation_strength", self.rotation_strength),
            ("expression_strength", self.expression_strength),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name, value in (
            ("rotation_one_euro_beta", self.rotation_one_euro_beta),
            ("expression_one_euro_beta", self.expression_one_euro_beta),
            ("rotation_max_acceleration_deg", self.rotation_max_acceleration_deg),
            ("rotation_max_jerk_deg", self.rotation_max_jerk_deg),
            ("expression_max_acceleration_z", self.expression_max_acceleration_z),
            ("expression_max_jerk_z", self.expression_max_jerk_z),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        weights = self.expression_coordinate_weights
        if self.expression_enabled and weights is None:
            raise ValueError(
                "Expression stabilization requires an explicit coordinate profile; "
                "run scripts/analyze_expression_sensitivity.py first"
            )
        if weights is not None:
            if len(weights) != expression_coordinates:
                raise ValueError(
                    "expression_coordinate_weights must contain exactly "
                    f"{expression_coordinates} values, got {len(weights)}"
                )
            if any(not math.isfinite(weight) or not 0.0 <= weight <= 1.0 for weight in weights):
                raise ValueError("expression coordinate weights must all be in [0, 1]")


@dataclass(slots=True, frozen=True)
class GeometryStabilizationOptions:
    """Experimental post-motion geometry controls.

    Defaults exactly preserve the stitch network output.  Stitch filtering
    operates on the network correction before blending.  Post-stitch
    filtering removes a per-frame 3D similarity transform, filters only the
    reviewed source-locked residuals, and restores the original transform so
    it cannot smooth global pose, translation, or scale.
    """

    stitch_strength: float = 1.0
    stitch_temporal_filter: TemporalFilterMode = "none"
    stitch_one_euro_min_cutoff_hz: float = 3.0
    stitch_one_euro_beta: float = 0.2
    stitch_one_euro_derivative_cutoff_hz: float = 1.0
    stitch_temporal_max_correction: float = 0.01

    post_stitch_enabled: bool = False
    post_stitch_one_euro_min_cutoff_hz: float = 3.0
    post_stitch_one_euro_beta: float = 0.2
    post_stitch_one_euro_derivative_cutoff_hz: float = 1.0
    post_stitch_strength: float = 1.0
    post_stitch_max_correction: float = 0.01
    post_stitch_keypoint_indices: tuple[int, ...] | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.stitch_strength != 1.0
            or self.stitch_temporal_filter != "none"
            or self.post_stitch_enabled
        )

    def validate(self, *, keypoint_count: int) -> None:
        if not math.isfinite(self.stitch_strength) or not 0.0 <= self.stitch_strength <= 1.0:
            raise ValueError(
                f"stitch_strength must be in [0, 1], got {self.stitch_strength}"
            )
        if (
            not math.isfinite(self.post_stitch_strength)
            or not 0.0 <= self.post_stitch_strength <= 1.0
        ):
            raise ValueError(
                "post_stitch_strength must be in [0, 1], got "
                f"{self.post_stitch_strength}"
            )
        if self.stitch_temporal_filter not in {"none", "one_euro"}:
            raise ValueError(
                f"Unknown stitch_temporal_filter: {self.stitch_temporal_filter!r}"
            )
        for name, value in (
            ("stitch_one_euro_min_cutoff_hz", self.stitch_one_euro_min_cutoff_hz),
            (
                "stitch_one_euro_derivative_cutoff_hz",
                self.stitch_one_euro_derivative_cutoff_hz,
            ),
            ("stitch_temporal_max_correction", self.stitch_temporal_max_correction),
            (
                "post_stitch_one_euro_min_cutoff_hz",
                self.post_stitch_one_euro_min_cutoff_hz,
            ),
            (
                "post_stitch_one_euro_derivative_cutoff_hz",
                self.post_stitch_one_euro_derivative_cutoff_hz,
            ),
            ("post_stitch_max_correction", self.post_stitch_max_correction),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero, got {value}")
        for name, value in (
            ("stitch_one_euro_beta", self.stitch_one_euro_beta),
            ("post_stitch_one_euro_beta", self.post_stitch_one_euro_beta),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        indices = self.post_stitch_keypoint_indices
        if indices is not None:
            if self.post_stitch_enabled and not indices:
                raise ValueError("post_stitch_keypoint_indices must not be empty when enabled")
            if len(indices) != len(set(indices)):
                raise ValueError("post_stitch_keypoint_indices must not contain duplicates")
            if any(index < 0 or index >= keypoint_count for index in indices):
                raise ValueError(
                    "post_stitch_keypoint_indices must be valid keypoint indices in "
                    f"[0, {keypoint_count - 1}]"
                )
            if self.post_stitch_enabled and len(indices) < 3:
                raise ValueError("post-stitch stabilization requires at least three keypoints")


@dataclass(slots=True, frozen=True)
class TurnAwareGuidanceOptions:
    """Chunk-level speaking/listening guidance controller.

    ``auto`` uses Schmitt-trigger RMS thresholds. Explicit turn modes allow a
    trusted upstream VAD/conversation controller to provide the observed state.
    Overlap and silence hold the previous stable speaking/listening target.
    """

    mode: TurnGuidanceMode = "disabled"
    speaking_cfg_other_audio: float = 1.0
    listening_cfg_other_audio: float = 2.0
    speech_rms_on: float = 0.01
    speech_rms_off: float = 0.005
    listen_rms_on: float = 0.01
    listen_rms_off: float = 0.005
    hysteresis_chunks: int = 2
    minimum_state_chunks: int = 2
    transition_chunks: int = 2

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def validate(self) -> None:
        if self.mode not in {
            "disabled",
            "auto",
            "speaking",
            "listening",
            "overlap",
            "silence",
        }:
            raise ValueError(f"Unknown turn guidance mode: {self.mode!r}")
        for name, value in (
            ("speaking_cfg_other_audio", self.speaking_cfg_other_audio),
            ("listening_cfg_other_audio", self.listening_cfg_other_audio),
            ("speech_rms_on", self.speech_rms_on),
            ("speech_rms_off", self.speech_rms_off),
            ("listen_rms_on", self.listen_rms_on),
            ("listen_rms_off", self.listen_rms_off),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.speech_rms_off > self.speech_rms_on:
            raise ValueError("speech_rms_off must not exceed speech_rms_on")
        if self.listen_rms_off > self.listen_rms_on:
            raise ValueError("listen_rms_off must not exceed listen_rms_on")
        for name, value in (
            ("hysteresis_chunks", self.hysteresis_chunks),
            ("minimum_state_chunks", self.minimum_state_chunks),
            ("transition_chunks", self.transition_chunks),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")


@dataclass(slots=True, frozen=True)
class RenderOptions:
    """Per-request knobs for ``Pipeline.process_chunk``.

    All defaults preserve the pipeline's previous behaviour, so existing
    call sites can pass ``RenderOptions()`` (or nothing) and observe no
    change. New surface is grouped here so the orchestrator signature
    stays stable as more knobs come online.

    Field groups:

    - **Output layout.** ``pixel_format`` picks the output frame layout;
      only ``yuv_i420_stacked_alpha`` ships the MODNet matte. ``bg_id``
      is required and must name an entry in the pipeline's background
      registry. The reserved value ``"transparent"`` skips bg compositing —
      pair with ``yuv_i420_stacked_alpha`` for clean alpha.
    - **Per-condition CFG weights.** The ``avtr1_decode`` engine fans out
      four condition passes (``past`` / ``self_audio`` / ``other_audio``
      / ``kp``); these three floats are how strongly each conditional
      pulls away from the unconditional ``past`` pass. They're broadcast
      across all ``latent_dim`` coords and passed as engine inputs each
      chunk, so retuning guidance no longer requires a rebuild.
    - **Progressive AR(1) noise.** ``noise_alpha`` and ``noise_trunc_z``
      were ``AVTR1MotionGenerator.__init__`` args; they live here
      now so a caller can vary them per chunk (e.g. ``noise_alpha=0``
      for independent frames).
    - **Streaming.** When ``stream_frames=True`` (default) the pipeline
      yields each frame as soon as decoder + putback + matting + pack +
      H2D finish for it (warp still runs once on the full chunk). When
      ``False`` the whole chunk is batched and all frames are packed in
      one H2D copy — lower latency to *last* frame, higher to *first*.
    """

    # --- Output layout (read by Pipeline / frame_sink) ---
    pixel_format: PixelFormat = "yuv_i420"
    bg_id: str | None = None

    # --- CFG weights (read by AVTR1MotionGenerator) ---
    cfg_self_audio: float = 2.0
    cfg_other_audio: float = 2.0
    cfg_kp: float = 3.0

    # --- AR(1) progressive noise (read by AVTR1MotionGenerator) ---
    noise_alpha: float = 2.0
    noise_trunc_z: float = 1.2

    # --- Optional post-prediction temporal guards ---
    stabilization: MotionStabilizationOptions = field(default_factory=MotionStabilizationOptions)

    # --- Optional stitch/post-stitch experiments ---
    geometry: GeometryStabilizationOptions = field(default_factory=GeometryStabilizationOptions)

    # --- Optional chunk-level speaking/listening CFG controller ---
    turn_guidance: TurnAwareGuidanceOptions = field(default_factory=TurnAwareGuidanceOptions)

    stream_frames: bool = True


__all__ = [
    "Chunk",
    "Frame",
    "FrameIterator",
    "GeometryStabilizationOptions",
    "KPInfo",
    "MotionStabilizationMode",
    "MotionStabilizationOptions",
    "RenderOptions",
    "TemporalFilterMode",
    "TurnAwareGuidanceOptions",
    "TurnGuidanceMode",
]
