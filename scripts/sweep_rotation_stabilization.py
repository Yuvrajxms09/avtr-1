# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Run a reproducible AVTR-1 temporal-filter sweep and summarize diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class SweepConfig:
    run_id: str
    label: str
    arguments: tuple[str, ...] = field(default_factory=tuple)


ROTATION_CONFIGS = (
    SweepConfig("00_baseline", "BASELINE"),
    SweepConfig(
        "01_spike_t010_s050",
        "SPIKE threshold=0.10 strength=0.50",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "0.50",
        ),
    ),
    SweepConfig(
        "02_spike_t010_s075",
        "SPIKE threshold=0.10 strength=0.75",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "0.75",
        ),
    ),
    SweepConfig(
        "03_spike_t010_s100",
        "SPIKE threshold=0.10 strength=1.00",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "1.00",
        ),
    ),
    SweepConfig(
        "04_spike_t015_s100",
        "SPIKE threshold=0.15 strength=1.00",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.15",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "1.00",
        ),
    ),
    SweepConfig(
        "05_spike_t020_s100",
        "SPIKE threshold=0.20 strength=1.00",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.20",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "1.00",
        ),
    ),
    SweepConfig(
        "06_one_euro_strong",
        "ONE EURO cutoff=1.5 beta=0.05",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "1.5",
            "--rotation-one-euro-beta",
            "0.05",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
    SweepConfig(
        "07_one_euro_moderate",
        "ONE EURO cutoff=2.0 beta=0.10",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "2.0",
            "--rotation-one-euro-beta",
            "0.10",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
    SweepConfig(
        "08_one_euro_light",
        "ONE EURO cutoff=3.0 beta=0.20",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "3.0",
            "--rotation-one-euro-beta",
            "0.20",
            "--rotation-temporal-max-correction-deg",
            "0.40",
        ),
    ),
    SweepConfig(
        "09_limiter_light",
        "LIMITER acceleration=0.35 jerk=0.50",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-max-acceleration-deg",
            "0.35",
            "--rotation-max-jerk-deg",
            "0.50",
            "--rotation-temporal-max-correction-deg",
            "0.40",
        ),
    ),
    SweepConfig(
        "10_limiter_moderate",
        "LIMITER acceleration=0.25 jerk=0.35",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-max-acceleration-deg",
            "0.25",
            "--rotation-max-jerk-deg",
            "0.35",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
    SweepConfig(
        "11_spike_one_euro_light",
        "SPIKE + ONE EURO light",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "0.75",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "3.0",
            "--rotation-one-euro-beta",
            "0.20",
            "--rotation-temporal-max-correction-deg",
            "0.40",
        ),
    ),
    SweepConfig(
        "12_spike_one_euro_moderate",
        "SPIKE + ONE EURO moderate",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "0.75",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "2.0",
            "--rotation-one-euro-beta",
            "0.10",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
    SweepConfig(
        "13_one_euro_limiter",
        "ONE EURO moderate + LIMITER light",
        (
            "--motion-stabilization",
            "rotation",
            "--no-rotation-spike-guard",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "2.0",
            "--rotation-one-euro-beta",
            "0.10",
            "--rotation-max-acceleration-deg",
            "0.35",
            "--rotation-max-jerk-deg",
            "0.50",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
    SweepConfig(
        "14_all_rotation_layers",
        "SPIKE + ONE EURO + LIMITER",
        (
            "--motion-stabilization",
            "rotation",
            "--rotation-acceleration-threshold-deg",
            "0.10",
            "--rotation-max-correction-deg",
            "0.25",
            "--rotation-stabilization-strength",
            "0.75",
            "--rotation-temporal-filter",
            "one_euro",
            "--rotation-one-euro-min-cutoff-hz",
            "2.0",
            "--rotation-one-euro-beta",
            "0.10",
            "--rotation-max-acceleration-deg",
            "0.35",
            "--rotation-max-jerk-deg",
            "0.50",
            "--rotation-temporal-max-correction-deg",
            "0.50",
        ),
    ),
)


def _with_motion_mode(arguments: tuple[str, ...], mode: str) -> tuple[str, ...]:
    updated = list(arguments)
    try:
        mode_index = updated.index("--motion-stabilization") + 1
    except ValueError as error:
        raise ValueError("Sweep arguments do not define --motion-stabilization") from error
    updated[mode_index] = mode
    return tuple(updated)


def _expression_configs(profile: Path) -> tuple[SweepConfig, ...]:
    common = ("--expression-profile", str(profile))
    return (
        SweepConfig(
            "15_expression_light",
            "EXPRESSION ONE EURO light",
            (
                "--motion-stabilization",
                "expression",
                "--no-expression-spike-guard",
                "--expression-temporal-filter",
                "one_euro",
                "--expression-one-euro-min-cutoff-hz",
                "6.0",
                "--expression-one-euro-beta",
                "1.0",
                "--expression-temporal-max-correction-z",
                "0.10",
                *common,
            ),
        ),
        SweepConfig(
            "16_expression_moderate",
            "EXPRESSION ONE EURO moderate",
            (
                "--motion-stabilization",
                "expression",
                "--no-expression-spike-guard",
                "--expression-temporal-filter",
                "one_euro",
                "--expression-one-euro-min-cutoff-hz",
                "4.0",
                "--expression-one-euro-beta",
                "0.75",
                "--expression-temporal-max-correction-z",
                "0.15",
                *common,
            ),
        ),
        SweepConfig(
            "17_combined_light",
            "ROTATION + EXPRESSION light",
            (
                *_with_motion_mode(ROTATION_CONFIGS[11].arguments, "both"),
                "--no-expression-spike-guard",
                "--expression-temporal-filter",
                "one_euro",
                "--expression-one-euro-min-cutoff-hz",
                "6.0",
                "--expression-one-euro-beta",
                "1.0",
                "--expression-temporal-max-correction-z",
                "0.10",
                *common,
            ),
        ),
        SweepConfig(
            "18_combined_moderate",
            "ROTATION + EXPRESSION moderate",
            (
                *_with_motion_mode(ROTATION_CONFIGS[12].arguments, "both"),
                "--no-expression-spike-guard",
                "--expression-temporal-filter",
                "one_euro",
                "--expression-one-euro-min-cutoff-hz",
                "4.0",
                "--expression-one-euro-beta",
                "0.75",
                "--expression-temporal-max-correction-z",
                "0.15",
                *common,
            ),
        ),
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error
    return events


def _stage(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("stage") == name]


def _flatten(events: list[dict[str, Any]], *keys: str) -> np.ndarray:
    values: list[float] = []
    for event in events:
        current: Any = event
        for key in keys:
            current = current[key]
        values.extend(current)
    return np.asarray(values, dtype=np.float64)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _rotvec_kinematics_degrees(
    rotvecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linalg.norm(rotvecs, axis=1)
    quaternions = np.zeros((len(rotvecs), 4), dtype=np.float64)
    quaternions[:, 0] = np.cos(angles * 0.5)
    nonzero = angles > 1e-12
    quaternions[nonzero, 1:] = (
        rotvecs[nonzero] / angles[nonzero, None] * np.sin(angles[nonzero, None] * 0.5)
    )
    if len(quaternions) < 2:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    previous = quaternions[:-1]
    current = quaternions[1:]
    relative_w = previous[:, 0] * current[:, 0] + np.sum(
        previous[:, 1:] * current[:, 1:],
        axis=1,
    )
    relative_xyz = (
        previous[:, :1] * current[:, 1:]
        - current[:, :1] * previous[:, 1:]
        - np.cross(previous[:, 1:], current[:, 1:])
    )
    flip = relative_w < 0
    relative_w[flip] *= -1
    relative_xyz[flip] *= -1
    xyz_norm = np.linalg.norm(relative_xyz, axis=1)
    relative_angle = 2.0 * np.arctan2(xyz_norm, np.clip(relative_w, 0.0, 1.0))
    angular_velocity = np.zeros_like(relative_xyz)
    nonzero_relative = xyz_norm > 1e-12
    angular_velocity[nonzero_relative] = (
        relative_xyz[nonzero_relative]
        / xyz_norm[nonzero_relative, None]
        * relative_angle[nonzero_relative, None]
    )
    angular_acceleration = np.diff(angular_velocity, axis=0)
    angular_jerk = np.diff(angular_acceleration, axis=0)
    return (
        np.degrees(np.linalg.norm(angular_velocity, axis=1)),
        np.degrees(np.linalg.norm(angular_acceleration, axis=1)),
        np.degrees(np.linalg.norm(angular_jerk, axis=1)),
    )


def _chunk_boundary_steps_degrees(chunks: list[np.ndarray]) -> np.ndarray:
    values: list[float] = []
    for previous, current in pairwise(chunks):
        if previous.size == 0 or current.size == 0:
            continue
        steps, _, _ = _rotvec_kinematics_degrees(
            np.stack([previous[-1], current[0]]),
        )
        values.append(float(steps[0]))
    return np.asarray(values, dtype=np.float64)


def _chunk_boundary_l2(chunks: list[np.ndarray]) -> np.ndarray:
    values = [
        np.linalg.norm(current[0] - previous[-1])
        for previous, current in pairwise(chunks)
        if previous.size and current.size
    ]
    return np.asarray(values, dtype=np.float64)


def _summarize(log_path: Path) -> dict[str, float | int]:
    events = _read_events(log_path)
    predictions = _stage(events, "motion_prediction")
    stabilization = _stage(events, "motion_stabilization")
    geometry = _stage(events, "keypoint_geometry")
    raw_rotvec_chunks = [
        np.asarray(event["head_rotvec_rad"], dtype=np.float64)
        for event in predictions
        if "head_rotvec_rad" in event
    ]
    if raw_rotvec_chunks:
        raw_head_steps, raw_head_acceleration, raw_head_jerk = _rotvec_kinematics_degrees(
            np.concatenate(raw_rotvec_chunks)
        )
        raw_head_boundaries = _chunk_boundary_steps_degrees(raw_rotvec_chunks)
    else:
        raw_head_steps = _flatten(predictions, "head_rotation_step_deg", "values")
        raw_head_acceleration = np.abs(np.diff(raw_head_steps))
        raw_head_jerk = np.abs(np.diff(raw_head_acceleration))
        raw_head_boundaries = np.empty(0)
    raw_expression_chunks = [
        np.asarray(event["expression_normalized"], dtype=np.float64)
        for event in predictions
        if "expression_normalized" in event
    ]
    if raw_expression_chunks:
        raw_expression_steps = np.linalg.norm(
            np.diff(np.concatenate(raw_expression_chunks), axis=0),
            axis=1,
        )
        raw_expression_boundaries = _chunk_boundary_l2(raw_expression_chunks)
    else:
        raw_expression_steps = _flatten(
            predictions,
            "expression_step_normalized_l2",
            "values",
        )
        raw_expression_boundaries = np.empty(0)
    if stabilization:
        filter_elapsed_ms = np.asarray(
            [
                float(event["elapsed_ms"])
                for event in stabilization
                if event.get("elapsed_ms") is not None
            ],
            dtype=np.float64,
        )
        corrected_rotvec_chunks = [
            np.asarray(event["rotation"]["corrected_rotvec_rad"], dtype=np.float64)
            for event in stabilization
        ]
        corrected_steps, corrected_acceleration, corrected_jerk = _rotvec_kinematics_degrees(
            np.concatenate(corrected_rotvec_chunks)
        )
        corrected_boundaries = _chunk_boundary_steps_degrees(corrected_rotvec_chunks)
        rotation_correction = _flatten(
            stabilization,
            "rotation",
            "correction_deg",
            "values",
        )
        temporal_correction = np.asarray(
            [
                value
                for event in stabilization
                for value in event["rotation"].get(
                    "temporal_correction_deg",
                    event["rotation"]["correction_deg"],
                )["values"]
            ],
            dtype=np.float64,
        )
        rotation_interventions = sum(
            int(event["rotation"]["intervention_frames"]) for event in stabilization
        )
        corrected_expression_chunks = [
            np.asarray(
                event["expression"]["corrected_normalized"],
                dtype=np.float64,
            )
            for event in stabilization
        ]
        corrected_expression_steps = np.linalg.norm(
            np.diff(np.concatenate(corrected_expression_chunks), axis=0),
            axis=1,
        )
        corrected_expression_boundaries = _chunk_boundary_l2(corrected_expression_chunks)
    else:
        filter_elapsed_ms = np.empty(0)
        corrected_steps = raw_head_steps
        corrected_acceleration = raw_head_acceleration
        corrected_jerk = raw_head_jerk
        corrected_boundaries = raw_head_boundaries
        rotation_correction = np.empty(0)
        temporal_correction = np.empty(0)
        rotation_interventions = 0
        corrected_expression_steps = raw_expression_steps
        corrected_expression_boundaries = raw_expression_boundaries
    face_variation = np.asarray(
        [event["driving_stitched"]["radius_range_pct"] for event in geometry],
        dtype=np.float64,
    )
    lipsync_variation = np.asarray(
        [event["driving_stitched_lipsync_subset"]["radius_range_pct"] for event in geometry],
        dtype=np.float64,
    )
    expression_correction = (
        _flatten(stabilization, "expression", "correction_l2", "values")
        if stabilization
        else np.empty(0)
    )
    raw_expression_p95 = _percentile(raw_expression_steps, 95)
    corrected_expression_p95 = _percentile(corrected_expression_steps, 95)
    return {
        "head_step_p95_deg": _percentile(corrected_steps, 95),
        "head_step_p99_deg": _percentile(corrected_steps, 99),
        "head_step_max_deg": float(corrected_steps.max()) if corrected_steps.size else math.nan,
        "head_acceleration_p95_deg": _percentile(corrected_acceleration, 95),
        "head_jerk_p95_deg": _percentile(corrected_jerk, 95),
        "head_boundary_step_p95_deg": _percentile(corrected_boundaries, 95),
        "face_radius_p95_pct": _percentile(face_variation, 95),
        "lipsync_radius_p95_pct": _percentile(lipsync_variation, 95),
        "rotation_intervention_frames": rotation_interventions,
        "rotation_correction_p95_deg": _percentile(rotation_correction, 95),
        "rotation_correction_max_deg": (
            float(rotation_correction.max()) if rotation_correction.size else 0.0
        ),
        "rotation_temporal_correction_p95_deg": _percentile(temporal_correction, 95),
        "expression_correction_p95_l2": _percentile(expression_correction, 95),
        "expression_step_raw_p95_l2": raw_expression_p95,
        "expression_step_corrected_p95_l2": corrected_expression_p95,
        "expression_boundary_step_p95_l2": _percentile(
            corrected_expression_boundaries,
            95,
        ),
        "expression_activity_retention": (
            corrected_expression_p95 / raw_expression_p95 if raw_expression_p95 > 0 else math.nan
        ),
        "filter_elapsed_p50_ms": _percentile(filter_elapsed_ms, 50),
        "filter_elapsed_p95_ms": _percentile(filter_elapsed_ms, 95),
    }


def _tee_run(command: list[str], *, cwd: Path, log_path: Path) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Inference failed with exit code {return_code}; see {log_path}")
    return time.perf_counter() - started


def _profile_has_active_weights(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("coordinate_weights")
    if not isinstance(weights, list) or len(weights) != 39:
        raise ValueError(f"{path} must contain exactly 39 coordinate_weights")
    return any(float(weight) > 0 for weight in weights)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats before writing strict JSON artifacts."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--listen", type=Path)
    parser.add_argument("--avatar", default="maria")
    parser.add_argument("--bg", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expression-profile", type=Path)
    parser.add_argument("--native-size", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--cfg-self-audio", type=float, default=2.0)
    parser.add_argument("--cfg-other-audio", type=float, default=1.0)
    parser.add_argument("--cfg-kp", type=float, default=3.0)
    parser.add_argument("--noise-alpha", type=float, default=2.0)
    parser.add_argument("--noise-trunc-z", type=float, default=1.2)
    parser.add_argument("--ode-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not args.speech.is_file():
        parser.error(f"Speech audio not found: {args.speech}")
    if args.listen is not None and not args.listen.is_file():
        parser.error(f"Listen audio not found: {args.listen}")
    configs = list(ROTATION_CONFIGS)
    if args.expression_profile is not None:
        if not args.expression_profile.is_file():
            parser.error(f"Expression profile not found: {args.expression_profile}")
        try:
            active = _profile_has_active_weights(args.expression_profile)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if not active:
            parser.error(
                "Expression profile has no active weights; review sensitivity artifacts first"
            )
        configs.extend(_expression_configs(args.expression_profile))

    output = args.out.resolve()
    videos = output / "videos"
    diagnostics = output / "diagnostics"
    logs = output / "logs"
    for directory in (videos, diagnostics, logs):
        directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    generator = repo / "scripts" / "generate_offline.py"
    common = [
        sys.executable,
        str(generator),
        "--speech",
        str(args.speech.resolve()),
        "--avatar",
        args.avatar,
        "--bg",
        args.bg,
        "--cfg-self-audio",
        str(args.cfg_self_audio),
        "--cfg-other-audio",
        str(args.cfg_other_audio),
        "--cfg-kp",
        str(args.cfg_kp),
        "--noise-alpha",
        str(args.noise_alpha),
        "--noise-trunc-z",
        str(args.noise_trunc_z),
        "--ode-steps",
        str(args.ode_steps),
        "--seed",
        str(args.seed),
    ]
    if args.native_size:
        common.append("--native-size")
    if args.listen is not None:
        common.extend(("--listen", str(args.listen.resolve())))

    rows: list[dict[str, Any]] = []
    print(f"Running {len(configs)} configurations into {output}")
    for index, config in enumerate(configs, start=1):
        video_path = videos / f"{config.run_id}.mp4"
        diagnostic_path = diagnostics / f"{config.run_id}.jsonl"
        run_log_path = logs / f"{config.run_id}.log"
        print(f"\n[{index}/{len(configs)}] {config.label}")
        if args.reuse_existing and video_path.is_file() and diagnostic_path.is_file():
            print("  Reusing existing video and diagnostics")
            runtime_seconds = math.nan
        else:
            command = [
                *common,
                *config.arguments,
                "--motion-debug-jsonl",
                str(diagnostic_path),
                "--out",
                str(video_path),
            ]
            runtime_seconds = _tee_run(command, cwd=repo, log_path=run_log_path)
        metrics = _summarize(diagnostic_path)
        rows.append(
            {
                "run_id": config.run_id,
                "label": config.label,
                "runtime_seconds": runtime_seconds,
                **metrics,
                "video": str(video_path),
                "diagnostics": str(diagnostic_path),
            }
        )

    baseline = rows[0]
    denominators = {
        "head_step_p95_deg": float(baseline["head_step_p95_deg"]),
        "head_acceleration_p95_deg": float(baseline["head_acceleration_p95_deg"]),
        "head_jerk_p95_deg": float(baseline["head_jerk_p95_deg"]),
        "head_boundary_step_p95_deg": float(baseline["head_boundary_step_p95_deg"]),
        "face_radius_p95_pct": float(baseline["face_radius_p95_pct"]),
        "lipsync_radius_p95_pct": float(baseline["lipsync_radius_p95_pct"]),
    }
    for row in rows:
        ratios = [
            float(row[name]) / denominator
            for name, denominator in denominators.items()
            if denominator > 0 and math.isfinite(float(row[name]))
        ]
        row["stability_score"] = sum(ratios) / len(ratios) if ratios else math.nan
        retention = float(row["expression_activity_retention"])
        mouth_penalty = max(0.0, 0.9 - retention) * 2.0 if math.isfinite(retention) else 0.0
        correction_p95 = float(row["rotation_correction_p95_deg"])
        correction_penalty = (
            max(0.0, correction_p95 - 0.5) if math.isfinite(correction_p95) else 0.0
        )
        row["selection_score"] = float(row["stability_score"]) + mouth_penalty + correction_penalty
    rows.sort(key=lambda row: float(row["selection_score"]))
    summary_csv = output / "rotation_sweep_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "rotation_sweep_summary.json").write_text(
        json.dumps(_json_safe(rows), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / "sweep_configurations.json").write_text(
        json.dumps([asdict(config) for config in configs], indent=2),
        encoding="utf-8",
    )

    print("\nTop configurations — lower selection score is better")
    print(
        "rank  score    head_p95  jerk_p95  boundary_p95  face_p95  "
        "lipsync_p95  mouth_keep  correction_p95  configuration"
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4}  {float(row['selection_score']):>7.4f}  "
            f"{float(row['head_step_p95_deg']):>8.4f}  "
            f"{float(row['head_jerk_p95_deg']):>8.4f}  "
            f"{float(row['head_boundary_step_p95_deg']):>12.4f}  "
            f"{float(row['face_radius_p95_pct']):>8.4f}  "
            f"{float(row['lipsync_radius_p95_pct']):>11.4f}  "
            f"{float(row['expression_activity_retention']):>10.3f}  "
            f"{float(row['rotation_correction_p95_deg']):>14.4f}  "
            f"{row['run_id']}"
        )
    print(f"\nSummary: {summary_csv}")
    print(f"Videos: {videos}")
    print(f"Diagnostics: {diagnostics}")


if __name__ == "__main__":
    main()
