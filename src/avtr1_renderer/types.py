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


@dataclass(slots=True, frozen=True)
class MotionStabilizationOptions:
    """Experimental temporal guards applied to normalized motion output.

    The guards suppress isolated acceleration spikes; they are not general
    low-pass filters. Disabled defaults are deliberately output-preserving.
    Expression filtering operates in the model's z-score-normalized space so
    one threshold has comparable meaning across all 39 coordinates.
    """

    mode: MotionStabilizationMode = "none"

    rotation_acceleration_threshold_deg: float = 0.75
    rotation_max_correction_deg: float = 1.5
    rotation_strength: float = 1.0

    expression_acceleration_threshold_z: float = 1.5
    expression_max_correction_z: float = 2.0
    expression_strength: float = 1.0
    expression_coordinate_weights: tuple[float, ...] | None = None

    @property
    def rotation_enabled(self) -> bool:
        return self.mode in {"rotation", "both"}

    @property
    def expression_enabled(self) -> bool:
        return self.mode in {"expression", "both"}

    def validate(self, *, expression_coordinates: int) -> None:
        if self.mode not in {"none", "rotation", "expression", "both"}:
            raise ValueError(f"Unknown motion stabilization mode: {self.mode!r}")
        for name, value in (
            ("rotation_acceleration_threshold_deg", self.rotation_acceleration_threshold_deg),
            ("rotation_max_correction_deg", self.rotation_max_correction_deg),
            ("expression_acceleration_threshold_z", self.expression_acceleration_threshold_z),
            ("expression_max_correction_z", self.expression_max_correction_z),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be greater than zero, got {value}")
        for name, value in (
            ("rotation_strength", self.rotation_strength),
            ("expression_strength", self.expression_strength),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

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

    stream_frames: bool = True


__all__ = [
    "Chunk",
    "Frame",
    "FrameIterator",
    "KPInfo",
    "MotionStabilizationMode",
    "MotionStabilizationOptions",
    "RenderOptions",
]
