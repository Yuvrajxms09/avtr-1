# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Map normalized AVTR-1 expression coordinates to visible image changes.

This is an offline CUDA analysis tool. It perturbs each of the 39 learned
expression coordinates around a portrait's neutral motion, renders the exact
production warp/decoder path, and records spatial image-difference metrics.
It also measures post-stitch keypoint displacement and face/lipsync radius
changes, which directly expose coordinates capable of geometric morphing.
The generated profile is deliberately disabled (all weights zero) until a
human reviews the contact sheets and selects coordinates safe to constrain.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.avtr1_motion_generator import AVTR1MotionGenerator
from avtr1_renderer.components.liveportrait.motion_stitch import (
    MotionFrame,
    motion_stitch,
)
from avtr1_renderer.constants import LIPSYNC_COORDS
from avtr1_renderer.models.decoder import DecoderInput, DecoderOutput
from avtr1_renderer.models.warp import WarpInput
from avtr1_renderer.pipeline import Pipeline

REGIONS = (
    "upper_left",
    "upper_center",
    "upper_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "lower_left",
    "lower_center",
    "lower_right",
)
_COORDINATE_COUNTS = {
    coordinate: LIPSYNC_COORDS.count(coordinate) for coordinate in set(LIPSYNC_COORDS)
}
_LIPSYNC_KEYPOINTS = sorted({coordinate // 3 for coordinate in LIPSYNC_COORDS})


@dataclass(slots=True, frozen=True)
class SensitivityRender:
    image: np.ndarray
    keypoints: np.ndarray


def _parse_avatars(value: str) -> list[str]:
    avatars = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not avatars:
        raise argparse.ArgumentTypeError("provide at least one avatar id")
    return avatars


@torch.no_grad()
def _render_face_crop(
    pipeline: Pipeline,
    avatar: Avatar,
    motion: MotionFrame,
) -> SensitivityRender:
    """Render the canonical 512px face before pasteback/background effects."""
    x_s, x_d = motion_stitch(avatar.kp_info, motion, stitch=pipeline._stitch)
    feature = avatar.f_s.expand(1, -1, -1, -1, -1).contiguous()
    warped = pipeline._warp(
        WarpInput(
            feature_3d=feature,
            kp_source=x_s.contiguous(),
            kp_driving=x_d.contiguous(),
        )
    ).out
    face = torch.empty((1, 3, 512, 512), dtype=torch.float32, device="cuda")
    pipeline._decoder(DecoderInput(feature=warped), out=DecoderOutput(output=face))
    image = (
        face.clamp(0.0, 1.0)[0].permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    return SensitivityRender(
        image=image,
        keypoints=x_d[0].detach().float().cpu().numpy(),
    )


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def _label(image: np.ndarray, text: str) -> np.ndarray:
    bar_height = 42
    labelled = np.zeros((image.shape[0] + bar_height, image.shape[1], 3), dtype=np.uint8)
    labelled[bar_height:] = image
    cv2.putText(
        labelled,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labelled


def _difference_metrics(reference: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    difference = np.abs(changed.astype(np.float32) - reference.astype(np.float32)) / 255.0
    scalar = difference.mean(axis=2)
    height, width = scalar.shape
    yy, xx = np.mgrid[:height, :width]
    mass = float(scalar.sum())
    metrics: dict[str, Any] = {
        "mean_abs_rgb": float(difference.mean()),
        "p95_abs_rgb": float(np.percentile(difference, 95)),
        "changed_fraction_2_255": float((scalar > (2.0 / 255.0)).mean()),
        "difference_centroid_x": float((scalar * xx).sum() / mass / width) if mass else 0.5,
        "difference_centroid_y": float((scalar * yy).sum() / mass / height) if mass else 0.5,
    }
    row_edges = np.linspace(0, height, 4, dtype=int)
    column_edges = np.linspace(0, width, 4, dtype=int)
    region_index = 0
    for row in range(3):
        for column in range(3):
            region = scalar[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            metrics[f"region_{REGIONS[region_index]}_mean"] = float(region.mean())
            region_index += 1
    return metrics


def _radius(keypoints: np.ndarray) -> float:
    xy = keypoints[:, :2]
    center = xy.mean(axis=0)
    return float(np.linalg.norm(xy - center, axis=1).mean())


def _keypoint_metrics(reference: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    reference_xy = reference[:, :2]
    changed_xy = changed[:, :2]
    displacement = np.linalg.norm(changed_xy - reference_xy, axis=1)
    reference_extent = np.ptp(reference_xy, axis=0)
    changed_extent = np.ptp(changed_xy, axis=0)
    reference_radius = max(_radius(reference), 1e-8)
    changed_radius = _radius(changed)
    reference_lipsync_radius = max(_radius(reference[_LIPSYNC_KEYPOINTS]), 1e-8)
    changed_lipsync_radius = _radius(changed[_LIPSYNC_KEYPOINTS])
    return {
        "keypoint_displacement_mean_l2": float(displacement.mean()),
        "keypoint_displacement_max_l2": float(displacement.max()),
        "face_radius_change_pct": (changed_radius - reference_radius) / reference_radius * 100.0,
        "lipsync_radius_change_pct": (
            (changed_lipsync_radius - reference_lipsync_radius) / reference_lipsync_radius * 100.0
        ),
        "keypoint_width_change_pct": (
            (float(changed_extent[0]) - float(reference_extent[0]))
            / max(float(reference_extent[0]), 1e-8)
            * 100.0
        ),
        "keypoint_height_change_pct": (
            (float(changed_extent[1]) - float(reference_extent[1]))
            / max(float(reference_extent[1]), 1e-8)
            * 100.0
        ),
    }


def _heatmap(reference: np.ndarray, changed: np.ndarray) -> np.ndarray:
    difference = np.abs(changed.astype(np.float32) - reference.astype(np.float32)).mean(axis=2)
    scale = max(float(np.percentile(difference, 99)), 1.0)
    normalized = np.clip(difference / scale * 255.0, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No sensitivity rows were generated")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["expression_coordinate"])].append(row)

    aggregated: list[dict[str, Any]] = []
    metric_names = (
        "mean_abs_rgb",
        "p95_abs_rgb",
        "changed_fraction_2_255",
        "difference_centroid_x",
        "difference_centroid_y",
        "keypoint_displacement_mean_l2",
        "keypoint_displacement_max_l2",
        "face_radius_change_abs_pct",
        "lipsync_radius_change_abs_pct",
        "keypoint_width_change_abs_pct",
        "keypoint_height_change_abs_pct",
        *(f"region_{region}_mean" for region in REGIONS),
    )
    for coordinate, coordinate_rows in sorted(grouped.items()):
        flattened = LIPSYNC_COORDS[coordinate]
        result: dict[str, Any] = {
            "expression_coordinate": coordinate,
            "flattened_expression_coordinate": flattened,
            "learned_keypoint": flattened // 3,
            "axis": "xyz"[flattened % 3],
            "duplicate_flattened_coordinate": _COORDINATE_COUNTS[flattened] > 1,
        }
        for name in metric_names:
            result[name] = float(np.mean([float(row[name]) for row in coordinate_rows]))
        result["dominant_spatial_region"] = max(
            REGIONS,
            key=lambda region: float(result[f"region_{region}_mean"]),
        )
        aggregated.append(result)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatars", type=_parse_avatars, default=["maria"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--perturbation-z",
        type=float,
        default=0.5,
        help="Positive/negative normalized perturbation applied per coordinate.",
    )
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    if args.perturbation_z <= 0:
        parser.error("--perturbation-z must be greater than zero")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be greater than zero")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading pipeline for: {', '.join(args.avatars)}")
    print(f"Output directory: {args.out}")
    print(f"Perturbation: +/-{args.perturbation_z:g} normalized units")
    try:
        pipeline, registry = Pipeline.from_artifacts(
            avatar_ids=args.avatars,
            out_size=(args.height, args.width),
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to load the production pipeline and requested avatars for "
            "expression sensitivity analysis"
        ) from error
    motion_generator = pipeline._motion_generator
    if not isinstance(motion_generator, AVTR1MotionGenerator):
        raise TypeError("Expression sensitivity requires AVTR1MotionGenerator")

    rows: list[dict[str, Any]] = []
    for avatar_id in args.avatars:
        print(f"\nPreparing reference render for {avatar_id}...")
        try:
            avatar = registry[avatar_id]
            reference_normalized = motion_generator.reference_normalized_motion(avatar)
            reference_motion = motion_generator.motion_from_normalized(reference_normalized)
            reference_render = _render_face_crop(pipeline, avatar, reference_motion)
        except Exception as error:
            raise RuntimeError(
                f"Failed to prepare reference render for avatar={avatar_id!r}"
            ) from error
        avatar_dir = args.out / "contact_sheets" / avatar_id
        _write_rgb(avatar_dir / "reference.png", reference_render.image)

        print(f"Analyzing {avatar_id}: {len(LIPSYNC_COORDS)} coordinates x 2 directions")
        for coordinate, flattened in enumerate(LIPSYNC_COORDS):
            print(
                f"  [{coordinate + 1:02d}/{len(LIPSYNC_COORDS)}] "
                f"coordinate={coordinate:02d} keypoint={flattened // 3:02d} "
                f"axis={'xyz'[flattened % 3]}"
            )
            variants: dict[str, np.ndarray] = {}
            for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
                try:
                    perturbed = reference_normalized.clone()
                    perturbed[0, 0, 3 + coordinate] += sign * args.perturbation_z
                    motion = motion_generator.motion_from_normalized(perturbed)
                    rendered = _render_face_crop(pipeline, avatar, motion)
                except Exception as error:
                    raise RuntimeError(
                        "Expression sensitivity render failed: "
                        f"avatar={avatar_id!r}, coordinate={coordinate}, "
                        f"keypoint={flattened // 3}, axis={'xyz'[flattened % 3]!r}, "
                        f"direction={sign_name!r}, perturbation_z="
                        f"{sign * args.perturbation_z:g}"
                    ) from error
                variants[sign_name] = rendered.image
                geometry = _keypoint_metrics(reference_render.keypoints, rendered.keypoints)
                metrics = _difference_metrics(reference_render.image, rendered.image)
                rows.append(
                    {
                        "avatar": avatar_id,
                        "expression_coordinate": coordinate,
                        "flattened_expression_coordinate": flattened,
                        "learned_keypoint": flattened // 3,
                        "axis": "xyz"[flattened % 3],
                        "duplicate_flattened_coordinate": _COORDINATE_COUNTS[flattened] > 1,
                        "perturbation_z": sign * args.perturbation_z,
                        **geometry,
                        **{
                            f"{name.removesuffix('_pct')}_abs_pct": abs(value)
                            for name, value in geometry.items()
                            if name.endswith("_pct")
                        },
                        **metrics,
                    }
                )

            comparison = np.concatenate(
                [
                    _label(variants["minus"], f"coord {coordinate:02d}  -{args.perturbation_z:g}z"),
                    _label(reference_render.image, "reference"),
                    _label(variants["plus"], f"coord {coordinate:02d}  +{args.perturbation_z:g}z"),
                ],
                axis=1,
            )
            heatmaps = np.concatenate(
                [
                    _label(
                        _heatmap(reference_render.image, variants["minus"]),
                        "minus difference",
                    ),
                    _label(np.zeros_like(reference_render.image), "reference"),
                    _label(
                        _heatmap(reference_render.image, variants["plus"]),
                        "plus difference",
                    ),
                ],
                axis=1,
            )
            _write_rgb(avatar_dir / f"coordinate_{coordinate:02d}.png", comparison)
            _write_rgb(avatar_dir / f"coordinate_{coordinate:02d}_heatmap.png", heatmaps)

    aggregate = _aggregate(rows)
    _write_csv(args.out / "sensitivity_per_render.csv", rows)
    _write_csv(args.out / "sensitivity_by_coordinate.csv", aggregate)
    (args.out / "sensitivity_by_coordinate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    profile = {
        "version": 1,
        "coordinate_weights": [0.0] * len(LIPSYNC_COORDS),
        "review_required": True,
        "source_sensitivity": "sensitivity_by_coordinate.json",
        "note": (
            "All weights intentionally default to zero. Review contact sheets and "
            "lip-sync behavior before enabling individual coordinates."
        ),
    }
    (args.out / "expression_profile.review.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )

    print("\nMost visually sensitive coordinates:")
    for row in sorted(aggregate, key=lambda item: item["mean_abs_rgb"], reverse=True)[:10]:
        print(
            f"  coord={row['expression_coordinate']:02d} "
            f"kp={row['learned_keypoint']:02d}{row['axis']} "
            f"mean_abs={row['mean_abs_rgb']:.6f} "
            f"region={row['dominant_spatial_region']}"
        )
    print("\nLargest face-radius effects:")
    for row in sorted(
        aggregate,
        key=lambda item: item["face_radius_change_abs_pct"],
        reverse=True,
    )[:10]:
        print(
            f"  coord={row['expression_coordinate']:02d} "
            f"kp={row['learned_keypoint']:02d}{row['axis']} "
            f"face_radius={row['face_radius_change_abs_pct']:.4f}% "
            f"lipsync_radius={row['lipsync_radius_change_abs_pct']:.4f}%"
        )
    print(f"\nResults: {args.out}")
    print(f"Review profile: {args.out / 'expression_profile.review.json'}")


if __name__ == "__main__":
    main()
