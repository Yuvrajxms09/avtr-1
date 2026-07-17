# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Render a talking-head video from audio.

Downloads any missing models from HuggingFace on first run.

Usage:
    pixi run generate_offline --speech path/to/speech.mp3 --bg plain_white
    pixi run generate_offline --speech s.wav --listen l.wav --bg plain_white
    pixi run generate_offline --duration 10 --bg plain_white                          # 10 s silence
    pixi run generate_offline --speech s.wav --bg plain_white --no-mux                # video only
    pixi run generate_offline --speech s.wav --bg plain_white --stream-frames         # per-frame (default)
    pixi run generate_offline --speech s.wav --bg plain_white --no-stream-frames      # batched mode
    pixi run generate_offline --speech s.wav --avatar custom --bg plain_white --native-size

Requirements:
    AVTR1 TRT engines must be built first:
        pixi run build-avtr1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import soxr
import torch

from avtr1_renderer.avatar_loader import CropConfig
from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.diagnostics import LOGGER_NAME, record_session
from avtr1_renderer.frame_size import read_native_output_size
from avtr1_renderer.pipeline import Pipeline
from avtr1_renderer.types import Chunk, MotionStabilizationOptions, RenderOptions

SAMPLE_RATE = 16_000
FPS = 25


def _load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ")
    return audio.astype(np.float32)


def _fit_duration(audio: np.ndarray, target_samples: int) -> np.ndarray:
    n = audio.shape[0]
    if n == target_samples:
        return audio
    if n > target_samples:
        return audio[:target_samples]
    return np.concatenate([audio, np.zeros(target_samples - n, dtype=audio.dtype)])


def _align_tracks(
    speech: np.ndarray | None,
    listen: np.ndarray | None,
    duration: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speech_f32, listen_f32) of equal length.

    If ``duration`` is set, both tracks are trimmed/padded to it.
    Otherwise length = the longer of the two; missing track defaults to silence.
    """
    if speech is None and listen is None and duration is None:
        raise ValueError("Provide at least one of --speech, --listen, --duration.")
    if duration is not None:
        n = round(duration * SAMPLE_RATE)
    else:
        n = max(
            speech.shape[0] if speech is not None else 0,
            listen.shape[0] if listen is not None else 0,
        )
    s = speech if speech is not None else np.zeros(n, dtype=np.float32)
    listening = listen if listen is not None else np.zeros(n, dtype=np.float32)
    return _fit_duration(s, n), _fit_duration(listening, n)


def _chunk_window(pipeline: Pipeline) -> int:
    mg = pipeline._motion_generator
    return (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift


def _chunk_step(pipeline: Pipeline) -> int:
    return pipeline._motion_generator.chunk_size * pipeline._motion_generator.frame_len


def _slice_chunks(audio: np.ndarray, window: int, step: int) -> list[np.ndarray]:
    n_steps = max(1, (len(audio) + step - 1) // step)
    chunks = []
    for i in range(n_steps):
        start = i * step
        piece = audio[start : start + window]
        if len(piece) < window:
            piece = np.pad(piece, (0, window - len(piece)))
        chunks.append(piece)
    return chunks


def _configure_motion_diagnostics(*, console: bool, jsonl_path: Path | None) -> None:
    if not console and jsonl_path is None:
        return

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(jsonl_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _load_expression_profile(path: Path) -> tuple[float, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load expression profile {path}: {exc}") from exc
    if payload.get("version") != 1:
        raise ValueError(f"Expression profile {path} must have version=1")
    weights = payload.get("coordinate_weights")
    if not isinstance(weights, list):
        raise ValueError(f"Expression profile {path} must contain coordinate_weights")
    expected = len(LIPSYNC_COORDS)
    if len(weights) != expected:
        raise ValueError(
            f"Expression profile {path} must contain {expected} weights, got {len(weights)}"
        )
    try:
        parsed = tuple(float(weight) for weight in weights)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expression profile {path} contains non-numeric weights") from exc
    if any(not 0.0 <= weight <= 1.0 for weight in parsed):
        raise ValueError(f"Expression profile {path} weights must all be in [0, 1]")
    return parsed


def main() -> None:
    render_defaults = RenderOptions()
    crop_defaults = CropConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speech", type=Path, default=None, help="Speech / self audio (any format)"
    )
    parser.add_argument(
        "--listen", type=Path, default=None, help="Listen / other audio (any format)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Force render duration in seconds; audio is trimmed/padded to fit",
    )
    parser.add_argument("--avatar", default="maria", help="Avatar ID to render")
    parser.add_argument("--out", type=Path, default=Path("demo_output.mp4"))
    parser.add_argument(
        "--bg",
        required=True,
        help="Background ID (must match a file in the backgrounds artifact, e.g. 'plain_white')",
    )
    parser.add_argument(
        "--no-mux",
        dest="mux",
        action="store_false",
        help="Don't mux the speech audio into the output video.",
    )
    parser.add_argument(
        "--stream-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Yield each frame as soon as it's ready (default: on). "
        "Use --no-stream-frames for batched mode.",
    )
    parser.add_argument(
        "--native-size",
        action="store_true",
        help="Render at the source portrait's native size instead of 1280x720. "
        "Odd dimensions are reduced by one pixel for YUV420 encoding.",
    )
    parser.add_argument(
        "--cfg-self-audio",
        type=float,
        default=render_defaults.cfg_self_audio,
        help="Self/speech-audio guidance strength.",
    )
    parser.add_argument(
        "--cfg-other-audio",
        type=float,
        default=render_defaults.cfg_other_audio,
        help="Other/listening-audio guidance strength.",
    )
    parser.add_argument(
        "--cfg-kp",
        type=float,
        default=render_defaults.cfg_kp,
        help="Source keypoint/identity guidance strength.",
    )
    parser.add_argument(
        "--noise-alpha",
        type=float,
        default=render_defaults.noise_alpha,
        help="AR(1) noise correlation. Higher values correlate adjacent frames more strongly.",
    )
    parser.add_argument(
        "--noise-trunc-z",
        type=float,
        default=render_defaults.noise_trunc_z,
        help="Truncated-normal noise limit. Lower values reduce extreme samples.",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=5,
        help="Flow integration time points; must be at least 2 (default: 5).",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=crop_defaults.scale,
        help="Source face crop scale; affects static framing, not temporal smoothing.",
    )
    parser.add_argument(
        "--crop-vx-ratio",
        type=float,
        default=crop_defaults.vx_ratio,
        help="Horizontal source crop offset ratio.",
    )
    parser.add_argument(
        "--crop-vy-ratio",
        type=float,
        default=crop_defaults.vy_ratio,
        help="Vertical source crop offset ratio.",
    )
    parser.add_argument(
        "--crop-rotation",
        action=argparse.BooleanOptionalAction,
        default=crop_defaults.flag_do_rot,
        help="Align the source crop to detected eye/lip rotation (default: on).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch random seed for reproducible motion sampling.",
    )
    parser.add_argument(
        "--debug-motion",
        action="store_true",
        help="Print JSON motion/keypoint/render diagnostics; reduces throughput.",
    )
    parser.add_argument(
        "--motion-debug-jsonl",
        type=Path,
        default=None,
        help="Write JSONL diagnostics to this path; reduces throughput.",
    )
    parser.add_argument(
        "--motion-stabilization",
        choices=("none", "rotation", "expression", "both"),
        default="none",
        help="Experimental render-only temporal stabilization (default: none).",
    )
    parser.add_argument(
        "--rotation-spike-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable isolated rotation-reversal suppression (default: on).",
    )
    parser.add_argument(
        "--rotation-acceleration-threshold-deg",
        type=float,
        default=0.75,
        help="Rotation residual allowed per frame before correction, in degrees.",
    )
    parser.add_argument(
        "--rotation-max-correction-deg",
        type=float,
        default=1.5,
        help="Maximum rotation correction applied to one frame, in degrees.",
    )
    parser.add_argument(
        "--rotation-stabilization-strength",
        type=float,
        default=1.0,
        help="Rotation correction blend in [0, 1].",
    )
    parser.add_argument(
        "--rotation-temporal-filter",
        choices=("none", "one_euro"),
        default="none",
        help="Continuous adaptive rotation filter (default: none).",
    )
    parser.add_argument("--rotation-one-euro-min-cutoff-hz", type=float, default=2.0)
    parser.add_argument("--rotation-one-euro-beta", type=float, default=0.1)
    parser.add_argument("--rotation-one-euro-derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--rotation-temporal-max-correction-deg", type=float, default=0.5)
    parser.add_argument(
        "--rotation-max-acceleration-deg",
        type=float,
        default=0.0,
        help="Per-frame angular acceleration limit; zero disables it.",
    )
    parser.add_argument(
        "--rotation-max-jerk-deg",
        type=float,
        default=0.0,
        help="Per-frame angular jerk limit; zero disables it.",
    )
    parser.add_argument(
        "--expression-profile",
        type=Path,
        default=None,
        help="Versioned JSON profile with 39 expression coordinate weights.",
    )
    parser.add_argument(
        "--expression-spike-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable isolated expression-reversal suppression (default: on).",
    )
    parser.add_argument(
        "--expression-acceleration-threshold-z",
        type=float,
        default=1.5,
        help="Normalized expression residual allowed before correction.",
    )
    parser.add_argument(
        "--expression-max-correction-z",
        type=float,
        default=2.0,
        help="Maximum normalized expression correction per coordinate/frame.",
    )
    parser.add_argument(
        "--expression-stabilization-strength",
        type=float,
        default=1.0,
        help="Expression correction blend in [0, 1].",
    )
    parser.add_argument(
        "--expression-temporal-filter",
        choices=("none", "one_euro"),
        default="none",
        help="Continuous coordinate-weighted expression filter (default: none).",
    )
    parser.add_argument("--expression-one-euro-min-cutoff-hz", type=float, default=3.0)
    parser.add_argument("--expression-one-euro-beta", type=float, default=0.5)
    parser.add_argument("--expression-one-euro-derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--expression-temporal-max-correction-z", type=float, default=0.25)
    parser.add_argument(
        "--expression-max-acceleration-z",
        type=float,
        default=0.0,
        help="Per-frame normalized acceleration limit; zero disables it.",
    )
    parser.add_argument(
        "--expression-max-jerk-z",
        type=float,
        default=0.0,
        help="Per-frame normalized jerk limit; zero disables it.",
    )
    args = parser.parse_args()

    if args.speech is None and args.listen is None and args.duration is None:
        parser.error("Provide at least one of --speech, --listen, --duration.")
    if args.noise_alpha < 0:
        parser.error("--noise-alpha must be non-negative.")
    if args.noise_trunc_z <= 0:
        parser.error("--noise-trunc-z must be greater than zero.")
    if args.ode_steps < 2:
        parser.error("--ode-steps must be at least 2.")
    if args.crop_scale <= 0:
        parser.error("--crop-scale must be greater than zero.")
    for flag, value in (
        ("--rotation-acceleration-threshold-deg", args.rotation_acceleration_threshold_deg),
        ("--rotation-max-correction-deg", args.rotation_max_correction_deg),
        ("--expression-acceleration-threshold-z", args.expression_acceleration_threshold_z),
        ("--expression-max-correction-z", args.expression_max_correction_z),
    ):
        if value <= 0:
            parser.error(f"{flag} must be greater than zero.")
    for flag, value in (
        ("--rotation-stabilization-strength", args.rotation_stabilization_strength),
        ("--expression-stabilization-strength", args.expression_stabilization_strength),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{flag} must be in [0, 1].")
    expression_weights = None
    if args.expression_profile is not None:
        try:
            expression_weights = _load_expression_profile(args.expression_profile)
        except ValueError as exc:
            parser.error(str(exc))
    if args.motion_stabilization in {"expression", "both"} and expression_weights is None:
        parser.error("--expression-profile is required when expression stabilization is enabled")
    stabilization = MotionStabilizationOptions(
        mode=args.motion_stabilization,
        rotation_spike_guard=args.rotation_spike_guard,
        rotation_acceleration_threshold_deg=args.rotation_acceleration_threshold_deg,
        rotation_max_correction_deg=args.rotation_max_correction_deg,
        rotation_strength=args.rotation_stabilization_strength,
        rotation_temporal_filter=args.rotation_temporal_filter,
        rotation_one_euro_min_cutoff_hz=args.rotation_one_euro_min_cutoff_hz,
        rotation_one_euro_beta=args.rotation_one_euro_beta,
        rotation_one_euro_derivative_cutoff_hz=(args.rotation_one_euro_derivative_cutoff_hz),
        rotation_temporal_max_correction_deg=args.rotation_temporal_max_correction_deg,
        rotation_max_acceleration_deg=args.rotation_max_acceleration_deg,
        rotation_max_jerk_deg=args.rotation_max_jerk_deg,
        expression_spike_guard=args.expression_spike_guard,
        expression_acceleration_threshold_z=args.expression_acceleration_threshold_z,
        expression_max_correction_z=args.expression_max_correction_z,
        expression_strength=args.expression_stabilization_strength,
        expression_temporal_filter=args.expression_temporal_filter,
        expression_one_euro_min_cutoff_hz=args.expression_one_euro_min_cutoff_hz,
        expression_one_euro_beta=args.expression_one_euro_beta,
        expression_one_euro_derivative_cutoff_hz=(args.expression_one_euro_derivative_cutoff_hz),
        expression_temporal_max_correction_z=args.expression_temporal_max_correction_z,
        expression_max_acceleration_z=args.expression_max_acceleration_z,
        expression_max_jerk_z=args.expression_max_jerk_z,
        expression_coordinate_weights=expression_weights,
    )
    try:
        stabilization.validate(expression_coordinates=len(LIPSYNC_COORDS))
    except ValueError as exc:
        parser.error(str(exc))

    out_size = (720, 1280)
    if args.native_size:
        portraits_dir = get_artifact_manager().get_artifact_path("reference_frames")
        portrait_path = portraits_dir / f"{args.avatar}.png"
        if not portrait_path.is_file():
            parser.error(f"No portrait at {portrait_path}")
        out_size = read_native_output_size(portrait_path)
        print(f"Native output size: {out_size[1]}x{out_size[0]} (width x height, YUV420-aligned)")

    _configure_motion_diagnostics(
        console=args.debug_motion,
        jsonl_path=args.motion_debug_jsonl,
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    record_session(
        speech=str(args.speech) if args.speech else None,
        listen=str(args.listen) if args.listen else None,
        duration=args.duration,
        avatar=args.avatar,
        background=args.bg,
        output=str(args.out),
        stream_frames=args.stream_frames,
        seed=args.seed,
        ode_steps=args.ode_steps,
        crop_scale=args.crop_scale,
        crop_vx_ratio=args.crop_vx_ratio,
        crop_vy_ratio=args.crop_vy_ratio,
        crop_rotation=args.crop_rotation,
        native_size=args.native_size,
        output_height=out_size[0],
        output_width=out_size[1],
        motion_stabilization=args.motion_stabilization,
        expression_profile=(
            str(args.expression_profile) if args.expression_profile is not None else None
        ),
    )

    speech_raw = _load_mono_16k(args.speech) if args.speech else None
    listen_raw = _load_mono_16k(args.listen) if args.listen else None
    speech, listen = _align_tracks(speech_raw, listen_raw, args.duration)

    print(
        f"Audio: {speech.shape[0] / SAMPLE_RATE:.2f}s "
        f"(speech={'set' if args.speech else 'silence'}, "
        f"listen={'set' if args.listen else 'silence'})"
    )

    print(f"Loading pipeline for avatar '{args.avatar}'...")
    print("  (models are downloaded from HuggingFace on first run — this may take a few minutes)")
    crop_config = CropConfig(
        scale=args.crop_scale,
        vx_ratio=args.crop_vx_ratio,
        vy_ratio=args.crop_vy_ratio,
        flag_do_rot=args.crop_rotation,
    )
    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=[args.avatar],
        crop_config=crop_config,
        n_ode_steps=args.ode_steps,
        out_size=out_size,
    )
    avatar = registry[args.avatar]

    window = _chunk_window(pipeline)
    step = _chunk_step(pipeline)
    frames_per_chunk = pipeline._motion_generator.chunk_size

    speech_chunks = _slice_chunks(speech, window, step)
    listen_chunks = _slice_chunks(listen, window, step)
    n_chunks = len(speech_chunks)
    print(
        f"Chunks: {n_chunks}  frames: {n_chunks * frames_per_chunk}  "
        f"({n_chunks * frames_per_chunk / FPS:.1f}s at {FPS} fps)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_h, out_w = avatar.source.shape[-2:]

    audio_path = args.speech if (args.mux and args.speech is not None) else None
    writer = imageio_ffmpeg.write_frames(
        str(args.out),
        size=(out_w, out_h),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="yuv420p",
        pix_fmt_out="yuv420p",
        quality=8,
        macro_block_size=1,
        audio_path=str(audio_path) if audio_path is not None else None,
        audio_codec="aac" if audio_path is not None else None,
    )
    writer.send(None)

    mode = "streaming" if args.stream_frames else "batched"
    print(f"Render mode: {mode}")
    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id=args.bg,
        cfg_self_audio=args.cfg_self_audio,
        cfg_other_audio=args.cfg_other_audio,
        cfg_kp=args.cfg_kp,
        noise_alpha=args.noise_alpha,
        noise_trunc_z=args.noise_trunc_z,
        stabilization=stabilization,
        stream_frames=args.stream_frames,
    )
    print(
        "Motion options: "
        f"cfg_self={args.cfg_self_audio:g} "
        f"cfg_other={args.cfg_other_audio:g} "
        f"cfg_kp={args.cfg_kp:g} "
        f"noise_alpha={args.noise_alpha:g} "
        f"noise_trunc_z={args.noise_trunc_z:g} "
        f"ode_steps={args.ode_steps} "
        f"seed={args.seed}"
    )
    print(
        "Stabilization: "
        f"mode={args.motion_stabilization} "
        f"rotation_spike={args.rotation_spike_guard} "
        f"rotation_threshold={args.rotation_acceleration_threshold_deg:g}deg "
        f"rotation_max={args.rotation_max_correction_deg:g}deg "
        f"rotation_filter={args.rotation_temporal_filter} "
        f"rotation_cutoff={args.rotation_one_euro_min_cutoff_hz:g}Hz "
        f"rotation_beta={args.rotation_one_euro_beta:g} "
        f"rotation_accel_limit={args.rotation_max_acceleration_deg:g}deg "
        f"rotation_jerk_limit={args.rotation_max_jerk_deg:g}deg "
        f"expression_filter={args.expression_temporal_filter} "
        f"expression_profile={args.expression_profile or 'none'}"
    )
    state = None
    produced = 0
    chunk_times: list[float] = []

    try:
        for i, (sp, ls) in enumerate(zip(speech_chunks, listen_chunks, strict=True)):
            t0 = time.perf_counter()
            chunk = Chunk(audio_speech=sp, audio_listen=ls)
            state, frames_iter = pipeline.process_chunk(avatar, chunk, state, options)
            for frame in frames_iter:
                writer.send(frame.data.tobytes())
                produced += 1
            chunk_times.append((time.perf_counter() - t0) * 1000)
            if (i + 1) % 25 == 0 or i == n_chunks - 1:
                avg_ms = sum(chunk_times) / len(chunk_times)
                chunk_times.clear()
                print(
                    f"  chunk {i + 1}/{n_chunks} "
                    f"({produced} frames, {produced / FPS:.1f}s, avg {avg_ms:.0f} ms/chunk)"
                )
    finally:
        writer.close()

    print(f"\nDone. {produced} frames ({produced / FPS:.1f}s) → {args.out}")


if __name__ == "__main__":
    main()
