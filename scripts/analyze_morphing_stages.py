# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Analyze deterministic AVTR-1 stage captures for non-rigid morphing."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from safetensors.torch import load_file

from avtr1_renderer.constants import LIPSYNC_COORDS

FPS = 25.0
GEOMETRY_STAGES = ("driving_raw", "driving_network", "driving_final")


def _distribution(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(flat.size),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def _top_events(values: np.ndarray, count: int = 10) -> list[dict[str, float | int]]:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return []
    indices = np.argsort(flat)[-count:][::-1]
    return [
        {"frame": int(index), "seconds": float(index / FPS), "value": float(flat[index])}
        for index in indices
    ]


def _top_events_masked(
    values: np.ndarray,
    mask: np.ndarray,
    count: int = 10,
) -> list[dict[str, float | int]]:
    flat = np.asarray(values).reshape(-1)
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        return []
    ranked = valid_indices[np.argsort(flat[valid_indices])[-count:][::-1]]
    return [
        {"frame": int(index), "seconds": float(index / FPS), "value": float(flat[index])}
        for index in ranked
    ]


def _top_events_indexed(
    values: np.ndarray,
    frame_indices: np.ndarray,
    count: int = 10,
) -> list[dict[str, float | int]]:
    flat = np.asarray(values).reshape(-1)
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if flat.size != indices.size:
        raise ValueError("values and frame_indices must have matching lengths")
    ranked = np.argsort(flat)[-count:][::-1]
    return [
        {
            "frame": int(indices[index]),
            "seconds": float(indices[index] / FPS),
            "value": float(flat[index]),
        }
        for index in ranked
    ]


def _align_similarity(
    reference: np.ndarray,
    frames: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = reference[fit_indices]
    ref_center = ref.mean(axis=0)
    ref_zero = ref - ref_center
    variance = max(float(np.square(ref_zero).sum()), 1e-8)
    aligned = np.empty_like(frames, dtype=np.float64)
    scales = np.empty(frames.shape[0], dtype=np.float64)
    for index, frame in enumerate(frames):
        cur = frame[fit_indices]
        cur_center = cur.mean(axis=0)
        covariance = ref_zero.T @ (cur - cur_center)
        u, singular, vh = np.linalg.svd(covariance)
        diagonal = np.ones(3)
        if np.linalg.det(u @ vh) < 0:
            diagonal[-1] = -1.0
        rotation = u @ np.diag(diagonal) @ vh
        scale = max(float((singular * diagonal).sum() / variance), 1e-6)
        translation = cur_center - scale * (ref_center @ rotation)
        aligned[index] = ((frame - translation) @ rotation.T) / scale
        scales[index] = scale
    return aligned, scales


def _pairwise_distances(frames: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pairs = np.asarray(list(combinations(indices.tolist(), 2)), dtype=np.int64)
    return np.linalg.norm(frames[:, pairs[:, 0]] - frames[:, pairs[:, 1]], axis=-1)


def _best_temporal_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    indices: np.ndarray,
    fit_indices: np.ndarray,
    max_lag: int = 2,
) -> dict[str, float | int | None]:
    aligned_reference, _ = _align_similarity(reference[0], reference, fit_indices)
    aligned_candidate, _ = _align_similarity(reference[0], candidate, fit_indices)
    reference_step = np.linalg.norm(
        np.diff(aligned_reference[:, indices], axis=0),
        axis=-1,
    ).mean(axis=1)
    candidate_step = np.linalg.norm(
        np.diff(aligned_candidate[:, indices], axis=0),
        axis=-1,
    ).mean(axis=1)
    best_lag = None
    best_correlation = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = reference_step[-lag:], candidate_step[:lag]
        elif lag > 0:
            left, right = reference_step[:-lag], candidate_step[lag:]
        else:
            left, right = reference_step, candidate_step
        if left.size < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return {
        "frames": best_lag,
        "milliseconds": None if best_lag is None else best_lag / FPS * 1000.0,
        "correlation": None if best_lag is None else best_correlation,
    }


def _geometry_metrics(
    reference: np.ndarray,
    frames: np.ndarray,
    indices: np.ndarray,
    fit_indices: np.ndarray,
    frame_mask: np.ndarray | None = None,
    boundary_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    aligned, scales = _align_similarity(reference, frames, fit_indices)
    selected = aligned[:, indices]
    ref_selected = reference[indices]
    residual = np.sqrt(np.square(selected - ref_selected).sum(axis=-1).mean(axis=-1))
    pairwise = _pairwise_distances(aligned, indices)
    ref_pairwise = _pairwise_distances(reference[None], indices)[0]
    pairwise_change = np.sqrt(np.square(pairwise - ref_pairwise).mean(axis=-1))
    step = np.sqrt(np.square(np.diff(selected, axis=0)).sum(axis=-1).mean(axis=-1))
    acceleration = np.sqrt(
        np.square(np.diff(selected, n=2, axis=0)).sum(axis=-1).mean(axis=-1)
    )
    jerk = np.sqrt(np.square(np.diff(selected, n=3, axis=0)).sum(axis=-1).mean(axis=-1))
    xy = selected[..., :2]
    extent = xy.max(axis=1) - xy.min(axis=1)
    radius = np.linalg.norm(xy - xy.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
    area = extent[:, 0] * extent[:, 1]
    mask = (
        np.ones(frames.shape[0], dtype=bool)
        if frame_mask is None
        else np.asarray(frame_mask, dtype=bool)
    )
    if mask.shape != (frames.shape[0],):
        raise ValueError("frame_mask must contain one value per frame")
    step_mask = mask[1:] & mask[:-1]
    acceleration_mask = mask[2:] & mask[1:-1] & mask[:-2]
    jerk_mask = mask[3:] & mask[2:-1] & mask[1:-2] & mask[:-3]
    boundaries = (
        np.empty(0, dtype=np.int64)
        if boundary_indices is None
        else np.asarray(boundary_indices, dtype=np.int64)
    )
    boundary_steps = boundaries - 1
    boundary_steps = boundary_steps[
        (boundary_steps >= 0) & (boundary_steps < step.shape[0])
    ]
    boundary_steps = boundary_steps[step_mask[boundary_steps]]

    def percent_step(values: np.ndarray) -> np.ndarray:
        denominator = np.maximum(np.abs(values[:-1]), 1e-8)
        return np.abs(np.diff(values)) / denominator * 100.0

    return {
        "frames": int(mask.sum()),
        "nonrigid_residual_rms": _distribution(residual[mask]),
        "pairwise_distance_change_rms": _distribution(pairwise_change[mask]),
        "pairwise_distance_range": _distribution(
            np.ptp(pairwise[mask], axis=0) if mask.any() else np.empty(0)
        ),
        "temporal_step_rms": _distribution(step[step_mask]),
        "temporal_acceleration_rms": _distribution(acceleration[acceleration_mask]),
        "temporal_jerk_rms": _distribution(jerk[jerk_mask]),
        "chunk_boundary_step_rms": _distribution(step[boundary_steps]),
        "aligned_width_step_pct": _distribution(percent_step(extent[:, 0])[step_mask]),
        "aligned_height_step_pct": _distribution(percent_step(extent[:, 1])[step_mask]),
        "aligned_area_step_pct": _distribution(percent_step(area)[step_mask]),
        "aligned_radius_step_pct": _distribution(percent_step(radius)[step_mask]),
        "aligned_radius_range_pct": (
            float(
                (radius[mask].max() - radius[mask].min())
                / max(float(radius[mask].mean()), 1e-8)
                * 100.0
            )
            if mask.any()
            else None
        ),
        "removed_similarity_scale_step_pct": _distribution(
            percent_step(scales)[step_mask]
        ),
        "worst_nonrigid_events": _top_events_masked(residual, mask),
        "worst_temporal_events": _top_events_masked(
            np.pad(step, (1, 0)),
            np.pad(step_mask, (1, 0)),
        ),
    }


def _pixel_motion_metrics(
    directory: Path,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = sorted(directory.glob("*.png"))
    if len(paths) < 2:
        if expected is not None:
            if len(paths) != expected["frames"]:
                raise ValueError(
                    f"Pixel stage frame count mismatch in {directory}: "
                    f"{len(paths)} != {expected['frames']}"
                )
            digest = hashlib.sha256()
            for path in paths:
                frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise OSError(f"Could not read {path}")
                digest.update(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes())
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Pixel stage fingerprint mismatch in {directory}")
        return {"frames": len(paths), "available": False}
    first_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise OSError(f"Could not read {paths[0]}")
    digest = hashlib.sha256()
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    digest.update(first_rgb.tobytes())
    previous = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    h, w = previous.shape
    mask = np.zeros_like(previous)
    mask[int(0.12 * h) : int(0.88 * h), int(0.12 * w) : int(0.88 * w)] = 255
    translations: list[float] = []
    rotations: list[float] = []
    scales: list[float] = []
    residuals: list[float] = []
    valid_counts: list[int] = []
    transition_frames: list[int] = []
    for frame_index, path in enumerate(paths[1:], start=1):
        current_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if current_bgr is None:
            raise OSError(f"Invalid stage frame {path}")
        current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
        digest.update(current_rgb.tobytes())
        current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        if current.shape != previous.shape:
            raise OSError(f"Stage frame shape changed at {path}")
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=400,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
        )
        if points is None or len(points) < 12:
            previous = current
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if tracked is None or status is None:
            previous = current
            continue
        valid = status.reshape(-1).astype(bool)
        before = points.reshape(-1, 2)[valid]
        after = tracked.reshape(-1, 2)[valid]
        if len(before) < 12:
            previous = current
            continue
        transform, inliers = cv2.estimateAffinePartial2D(
            before,
            after,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if transform is None:
            previous = current
            continue
        predicted = cv2.transform(before[None], transform)[0]
        residual = np.linalg.norm(after - predicted, axis=1)
        if inliers is not None:
            keep = inliers.reshape(-1).astype(bool)
            residual = residual[keep]
        a, b = float(transform[0, 0]), float(transform[1, 0])
        translations.append(float(np.linalg.norm(transform[:, 2])))
        rotations.append(float(np.degrees(np.arctan2(b, a))))
        scales.append(float(np.hypot(a, b)))
        residuals.append(float(np.percentile(residual, 95)) if residual.size else 0.0)
        valid_counts.append(int(residual.size))
        transition_frames.append(frame_index)
        previous = current
    if expected is not None:
        if len(paths) != expected["frames"]:
            raise ValueError(
                f"Pixel stage frame count mismatch in {directory}: "
                f"{len(paths)} != {expected['frames']}"
            )
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Pixel stage fingerprint mismatch in {directory}")
    return {
        "frames": len(paths),
        "available": bool(residuals),
        "global_translation_px": _distribution(np.asarray(translations)),
        "global_rotation_deg": _distribution(np.abs(rotations)),
        "global_scale_step_pct": _distribution(np.abs(np.asarray(scales) - 1.0) * 100.0),
        "rigid_aligned_feature_residual_p95_px": _distribution(np.asarray(residuals)),
        "rigid_aligned_feature_residual_p95_width_pct": _distribution(
            np.asarray(residuals) / w * 100.0
        ),
        "tracked_inlier_count": _distribution(np.asarray(valid_counts)),
        "worst_residual_events": _top_events_indexed(
            np.asarray(residuals),
            np.asarray(transition_frames),
        ),
    }


def _load_capture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported stage manifest schema in {manifest_path}")
    tensors = {
        name: tensor.numpy()
        for name, tensor in load_file(path / manifest["geometry_file"]).items()
    }
    for stage, stage_manifest in manifest["geometry_stages"].items():
        if stage not in tensors:
            raise ValueError(f"Geometry manifest references missing tensor {stage}")
        if list(tensors[stage].shape) != stage_manifest["shape"]:
            raise ValueError(f"Geometry shape manifest mismatch for {stage}")
        lengths_name = f"{stage}__chunk_lengths"
        if lengths_name not in tensors:
            raise ValueError(f"Geometry stage {stage} is missing chunk lengths")
        actual = hashlib.sha256(np.ascontiguousarray(tensors[stage]).tobytes()).hexdigest()
        if actual != stage_manifest["sha256"]:
            raise ValueError(
                f"Geometry stage fingerprint mismatch for {stage}: "
                f"{actual} != {stage_manifest['sha256']}"
            )
        lengths = np.ascontiguousarray(tensors[lengths_name])
        lengths_hash = hashlib.sha256(lengths.tobytes()).hexdigest()
        if lengths_hash != stage_manifest["chunk_lengths_sha256"]:
            raise ValueError(f"Chunk-boundary fingerprint mismatch for {stage}")
        if lengths.tolist() != stage_manifest["chunk_lengths"]:
            raise ValueError(f"Chunk-boundary manifest mismatch for {stage}")
        if len(lengths) != stage_manifest["chunks"] or int(lengths.sum()) != len(
            tensors[stage]
        ):
            raise ValueError(f"Invalid chunk boundaries for geometry stage {stage}")
    return manifest, tensors


def _analyze_capture(path: Path, *, audio_rms_threshold: float) -> dict[str, Any]:
    manifest, tensors = _load_capture(path)
    missing = [stage for stage in ("source_keypoints", *GEOMETRY_STAGES) if stage not in tensors]
    if missing:
        raise ValueError(f"Stage capture {path} is missing {missing}")
    frame_counts = {stage: tensors[stage].shape[0] for stage in GEOMETRY_STAGES}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"Geometry stage frame counts differ in {path}: {frame_counts}")
    chunk_boundaries = {
        stage: tensors[f"{stage}__chunk_lengths"].tolist() for stage in GEOMETRY_STAGES
    }
    if len({tuple(lengths) for lengths in chunk_boundaries.values()}) != 1:
        raise ValueError(f"Geometry stage chunk boundaries differ in {path}: {chunk_boundaries}")
    expected_frames = next(iter(frame_counts.values()))
    first_lengths = next(iter(chunk_boundaries.values()))
    boundary_indices = np.cumsum(np.asarray(first_lengths, dtype=np.int64))[:-1]
    for stage, pixel_manifest in manifest.get("pixel_stages", {}).items():
        if pixel_manifest["frames"] != expected_frames:
            raise ValueError(
                f"Pixel stage {stage} has {pixel_manifest['frames']} frames; "
                f"expected {expected_frames}"
            )
    reference = tensors["source_keypoints"][0].astype(np.float64)
    keypoint_count = reference.shape[0]
    lipsync = np.asarray(sorted({coordinate // 3 for coordinate in LIPSYNC_COORDS}))
    source_locked = np.asarray(
        [index for index in range(keypoint_count) if index not in set(lipsync.tolist())]
    )
    turn_masks: dict[str, np.ndarray] = {}
    if "audio_rms" in tensors:
        audio_rms = tensors["audio_rms"]
        if audio_rms.shape != (frame_counts["driving_raw"], 2):
            raise ValueError(f"audio_rms has invalid shape in {path}: {audio_rms.shape}")
        speaking = audio_rms[:, 0] >= audio_rms_threshold
        listening = audio_rms[:, 1] >= audio_rms_threshold
        turn_masks = {
            "speaking": speaking & ~listening,
            "listening": listening & ~speaking,
            "overlap": speaking & listening,
            "silence": ~speaking & ~listening,
        }
    geometry: dict[str, Any] = {}
    for stage in GEOMETRY_STAGES:
        frames = tensors[stage].astype(np.float64)
        geometry[stage] = {
            "all": _geometry_metrics(
                reference,
                frames,
                np.arange(keypoint_count),
                source_locked,
                boundary_indices=boundary_indices,
            ),
            "lipsync": _geometry_metrics(
                reference,
                frames,
                lipsync,
                source_locked,
                boundary_indices=boundary_indices,
            ),
            "source_locked": _geometry_metrics(
                reference,
                frames,
                source_locked,
                source_locked,
                boundary_indices=boundary_indices,
            ),
        }
        for subset_name, subset_indices in (
            ("all", np.arange(keypoint_count)),
            ("lipsync", lipsync),
            ("source_locked", source_locked),
        ):
            geometry[stage][subset_name]["turn_segments"] = {
                turn: _geometry_metrics(
                    reference,
                    frames,
                    subset_indices,
                    source_locked,
                    frame_mask=mask,
                    boundary_indices=boundary_indices,
                )
                for turn, mask in turn_masks.items()
            }

    raw_frames = tensors["driving_raw"].astype(np.float64)
    final_frames = tensors["driving_final"].astype(np.float64)
    temporal_lag = _best_temporal_lag(
        raw_frames,
        final_frames,
        source_locked,
        source_locked,
    )

    temporal_p95 = {
        stage: geometry[stage]["source_locked"]["temporal_step_rms"]["p95"]
        for stage in GEOMETRY_STAGES
    }
    amplification = {
        "network_vs_raw_pct": _relative_change(
            temporal_p95["driving_raw"], temporal_p95["driving_network"]
        ),
        "final_vs_network_pct": _relative_change(
            temporal_p95["driving_network"], temporal_p95["driving_final"]
        ),
    }
    first_material_amplification = None
    if amplification["network_vs_raw_pct"] is not None and amplification["network_vs_raw_pct"] > 10:
        first_material_amplification = "stitch_network"
    elif (
        amplification["final_vs_network_pct"] is not None
        and amplification["final_vs_network_pct"] > 10
    ):
        first_material_amplification = "post_stitch_controls"

    pixels = {}
    for stage in ("decoded_face", "final_composite"):
        stage_dir = path / stage
        if stage_dir.is_dir():
            pixels[stage] = _pixel_motion_metrics(
                stage_dir,
                manifest.get("pixel_stages", {}).get(stage),
            )
    pixel_amplification = None
    if all(
        pixels.get(stage, {}).get("available")
        for stage in ("decoded_face", "final_composite")
    ):
        decoded = pixels["decoded_face"][
            "rigid_aligned_feature_residual_p95_width_pct"
        ]["p95"]
        composite = pixels["final_composite"][
            "rigid_aligned_feature_residual_p95_width_pct"
        ]["p95"]
        pixel_amplification = _relative_change(decoded, composite)
    return {
        "path": str(path),
        "metadata": manifest["metadata"],
        "turn_segmentation": {
            "audio_rms_threshold": audio_rms_threshold,
            "frame_counts": {turn: int(mask.sum()) for turn, mask in turn_masks.items()},
        },
        "geometry": geometry,
        "stage_amplification": amplification,
        "raw_to_final_source_locked_best_lag": temporal_lag,
        "first_material_geometry_amplification": first_material_amplification,
        "pixel_motion": pixels,
        "final_composite_vs_decoded_residual_change_pct": pixel_amplification,
    }


def _relative_change(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or abs(baseline) < 1e-12:
        return None
    return (candidate - baseline) / baseline * 100.0


def _compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_fingerprint = baseline["metadata"].get("motion_fingerprint_sha256")
    candidate_fingerprint = candidate["metadata"].get("motion_fingerprint_sha256")
    if not base_fingerprint or candidate_fingerprint != base_fingerprint:
        raise ValueError(
            "Cannot compare stage captures with different raw-motion fingerprints: "
            f"{base_fingerprint!r} != {candidate_fingerprint!r}"
        )
    geometry = {}
    for stage in GEOMETRY_STAGES:
        geometry[stage] = {}
        for subset in ("all", "lipsync", "source_locked"):
            geometry[stage][subset] = {}
            compared_metrics = (
                "nonrigid_residual_rms",
                "pairwise_distance_change_rms",
                "pairwise_distance_range",
                "temporal_step_rms",
                "temporal_acceleration_rms",
                "temporal_jerk_rms",
                "chunk_boundary_step_rms",
                "aligned_area_step_pct",
            )
            for metric in compared_metrics:
                base_value = baseline["geometry"][stage][subset][metric]["p95"]
                candidate_value = candidate["geometry"][stage][subset][metric]["p95"]
                geometry[stage][subset][metric] = {
                    "baseline_p95": base_value,
                    "candidate_p95": candidate_value,
                    "change_pct": _relative_change(base_value, candidate_value),
                }
            base_segments = baseline["geometry"][stage][subset].get("turn_segments", {})
            candidate_segments = candidate["geometry"][stage][subset].get(
                "turn_segments", {}
            )
            geometry[stage][subset]["turn_segments"] = {}
            for turn in sorted(set(base_segments) & set(candidate_segments)):
                geometry[stage][subset]["turn_segments"][turn] = {}
                for metric in compared_metrics:
                    base_value = base_segments[turn][metric]["p95"]
                    candidate_value = candidate_segments[turn][metric]["p95"]
                    geometry[stage][subset]["turn_segments"][turn][metric] = {
                        "baseline_p95": base_value,
                        "candidate_p95": candidate_value,
                        "change_pct": _relative_change(base_value, candidate_value),
                    }
    final_source_locked = geometry["driving_final"]["source_locked"]
    final_lipsync = geometry["driving_final"]["lipsync"]
    target_change = final_source_locked["temporal_step_rms"]["change_pct"]
    lipsync_change = final_lipsync["temporal_step_rms"]["change_pct"]
    lipsync_range_change = final_lipsync["pairwise_distance_range"]["change_pct"]
    pixels: dict[str, Any] = {}
    for stage in ("decoded_face", "final_composite"):
        base_stage = baseline["pixel_motion"].get(stage, {})
        candidate_stage = candidate["pixel_motion"].get(stage, {})
        if not base_stage.get("available") or not candidate_stage.get("available"):
            continue
        pixels[stage] = {}
        for metric in (
            "global_translation_px",
            "global_rotation_deg",
            "global_scale_step_pct",
            "rigid_aligned_feature_residual_p95_px",
            "rigid_aligned_feature_residual_p95_width_pct",
        ):
            base_value = base_stage[metric]["p95"]
            candidate_value = candidate_stage[metric]["p95"]
            pixels[stage][metric] = {
                "baseline_p95": base_value,
                "candidate_p95": candidate_value,
                "change_pct": _relative_change(base_value, candidate_value),
            }
    return {
        "motion_fingerprint_sha256": base_fingerprint,
        "geometry": geometry,
        "pixel_motion": pixels,
        "geometry_gate": {
            "target_improves_at_least_10_pct": target_change is not None and target_change <= -10,
            "lipsync_motion_range_change_within_5_pct": (
                lipsync_range_change is not None and abs(lipsync_range_change) <= 5
            ),
            "target_change_pct": target_change,
            "lipsync_change_pct": lipsync_change,
            "lipsync_motion_range_change_pct": lipsync_range_change,
            "note": "External lip-sync confidence, mouth range, and visual gates remain required.",
        },
    }


def _parse_candidate(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH") from exc
    if not label:
        raise argparse.ArgumentTypeError("candidate label must not be empty")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=_parse_candidate,
        action="append",
        default=[],
        help="Repeatable LABEL=PATH deterministic candidate capture.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--audio-rms-threshold",
        type=float,
        default=0.001,
        help="Per-track RMS threshold used for speaking/listening segmentation.",
    )
    args = parser.parse_args()
    if args.audio_rms_threshold < 0:
        parser.error("--audio-rms-threshold must be non-negative")

    baseline = _analyze_capture(
        args.baseline,
        audio_rms_threshold=args.audio_rms_threshold,
    )
    candidates = {
        label: _analyze_capture(path, audio_rms_threshold=args.audio_rms_threshold)
        for label, path in args.candidate
    }
    report = {
        "schema_version": 1,
        "methodology": {
            "geometry_alignment": (
                "Per-frame 3D similarity alignment on source-locked learned keypoints; "
                "translation, rotation, and uniform scale are removed before residual metrics."
            ),
            "pixel_alignment": (
                "Central-region pyramidal optical flow plus robust 2D similarity estimation."
            ),
            "limitations": (
                "Learned keypoint subsets are not anatomical labels. Pixel tracking is secondary; "
                "normal-speed full-resolution review and external lip-sync validation "
                "remain required."
            ),
        },
        "baseline": baseline,
        "candidates": candidates,
        "comparisons": {
            label: _compare(baseline, candidate) for label, candidate in candidates.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
