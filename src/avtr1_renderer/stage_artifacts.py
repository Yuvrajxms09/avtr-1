# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Lossless, opt-in stage artifacts for deterministic morphing localization."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import torch

from avtr1_renderer.diagnostics import record_stage_artifact

_SCHEMA_VERSION = 1
_ACTIVE: ContextVar[StageArtifactSession | None] = ContextVar(
    "avtr1_stage_artifact_session",
    default=None,
)


class StageArtifactSession:
    """Collect compact geometry tensors and optionally lossless pixel stages."""

    def __init__(
        self,
        output_dir: Path,
        *,
        metadata: dict[str, Any],
        capture_pixels: bool,
    ) -> None:
        self.output_dir = output_dir
        self.metadata = {key: str(value) for key, value in metadata.items()}
        self.capture_pixels = capture_pixels
        self._token = None
        self._geometry: dict[str, list[torch.Tensor]] = {}
        self._chunk_lengths: dict[str, list[int]] = {}
        self._pixel_counts: dict[str, int] = {}
        self._digests: dict[str, Any] = {}

    def __enter__(self) -> StageArtifactSession:
        if _ACTIVE.get() is not None:
            raise RuntimeError("A stage artifact session is already active")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Stage artifact directory must be empty or absent: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._token = _ACTIVE.set(self)
        record_stage_artifact(
            event="session_start",
            output_dir=str(self.output_dir),
            capture_pixels=self.capture_pixels,
            metadata=self.metadata,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc_type is None:
                self._save()
        finally:
            if self._token is not None:
                _ACTIVE.reset(self._token)

    def record_geometry(self, stage: str, values: torch.Tensor) -> None:
        captured = values.detach().float().cpu().contiguous()
        if stage == "source_keypoints" and stage in self._geometry:
            if not torch.equal(self._geometry[stage][0], captured):
                raise ValueError("Source keypoints changed within one stage capture")
            return
        self._geometry.setdefault(stage, []).append(captured)
        self._chunk_lengths.setdefault(stage, []).append(captured.shape[0])
        self._digest(stage).update(captured.numpy().tobytes())
        record_stage_artifact(
            event="geometry_chunk",
            stage_name=stage,
            shape=list(captured.shape),
        )

    def record_pixels(self, stage: str, values: torch.Tensor) -> None:
        if not self.capture_pixels:
            return
        frames = (
            values.detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(dtype=torch.uint8)
            .cpu()
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        stage_dir = self.output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        start = self._pixel_counts.get(stage, 0)
        for offset, frame in enumerate(frames):
            index = start + offset
            path = stage_dir / f"{index:06d}.png"
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"Could not write stage frame: {path}")
            self._digest(stage).update(frame.tobytes())
        self._pixel_counts[stage] = start + len(frames)
        record_stage_artifact(
            event="pixel_chunk",
            stage_name=stage,
            first_frame=start,
            frames=len(frames),
        )

    def _digest(self, stage: str) -> Any:
        return self._digests.setdefault(stage, hashlib.sha256())

    def _save(self) -> None:
        from safetensors.torch import save_file

        if not self._geometry:
            raise ValueError("Stage capture completed without geometry")
        tensors: dict[str, torch.Tensor] = {}
        for stage, chunks in self._geometry.items():
            tensors[stage] = chunks[0] if stage == "source_keypoints" else torch.cat(chunks)
            tensors[f"{stage}__chunk_lengths"] = torch.tensor(
                self._chunk_lengths[stage],
                dtype=torch.int64,
            )
        geometry_path = self.output_dir / "geometry.safetensors"
        geometry_temporary = geometry_path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(geometry_temporary))
        geometry_temporary.replace(geometry_path)
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "metadata": self.metadata,
            "geometry_file": geometry_path.name,
            "geometry_stages": {
                stage: {
                    "shape": list(tensors[stage].shape),
                    "chunks": len(self._chunk_lengths[stage]),
                    "chunk_lengths": self._chunk_lengths[stage],
                    "chunk_lengths_sha256": hashlib.sha256(
                        tensors[f"{stage}__chunk_lengths"].numpy().tobytes()
                    ).hexdigest(),
                    "sha256": self._digests[stage].hexdigest(),
                }
                for stage in self._geometry
            },
            "pixel_stages": {
                stage: {
                    "directory": stage,
                    "frames": count,
                    "sha256": self._digests[stage].hexdigest(),
                }
                for stage, count in self._pixel_counts.items()
            },
        }
        manifest_path = self.output_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        record_stage_artifact(
            event="session_complete",
            output_dir=str(self.output_dir),
            geometry_stages=sorted(self._geometry),
            pixel_stages=sorted(self._pixel_counts),
        )


def record_geometry_stage(stage: str, values: torch.Tensor) -> None:
    session = _ACTIVE.get()
    if session is not None:
        session.record_geometry(stage, values)


def record_pixel_stage(stage: str, values: torch.Tensor) -> None:
    session = _ACTIVE.get()
    if session is not None:
        session.record_pixels(stage, values)


__all__ = ["StageArtifactSession", "record_geometry_stage", "record_pixel_stage"]
