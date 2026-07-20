# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Deterministic raw-motion capture and replay for causal renderer A/B runs."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import torch

from avtr1_renderer.diagnostics import record_motion_trajectory

_SCHEMA_VERSION = 1
_ACTIVE: ContextVar[MotionTrajectorySession | None] = ContextVar(
    "avtr1_motion_trajectory_session",
    default=None,
)


def _canonical_metadata(metadata: dict[str, str]) -> bytes:
    filtered = {key: value for key, value in metadata.items() if key != "fingerprint_sha256"}
    return json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(
    normalized_motion: torch.Tensor,
    raw_rotation: torch.Tensor,
    chunk_lengths: torch.Tensor,
    metadata: dict[str, str],
) -> str:
    digest = hashlib.sha256(_canonical_metadata(metadata))
    digest.update(normalized_motion.detach().cpu().contiguous().numpy().tobytes())
    digest.update(raw_rotation.detach().cpu().contiguous().numpy().tobytes())
    digest.update(chunk_lengths.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    result = {key: str(value) for key, value in metadata.items()}
    result["schema_version"] = str(_SCHEMA_VERSION)
    return result


class MotionTrajectorySession:
    """Context-bound capture/replay session used at the raw normalized boundary."""

    def __init__(
        self,
        *,
        mode: Literal["capture", "replay"],
        path: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.mode = mode
        self.path = path
        self.metadata = _stringify_metadata(metadata)
        self._chunks: list[torch.Tensor] = []
        self._rotation_chunks: list[torch.Tensor] = []
        self._chunk_lengths: list[int] = []
        self._cursor = 0
        self._frame_cursor = 0
        self._token = None
        self._replay_motion: torch.Tensor | None = None
        self._replay_lengths: list[int] = []
        self.fingerprint: str | None = None
        if mode == "replay":
            self._load_replay()

    @classmethod
    def capture(cls, path: Path, metadata: dict[str, Any]) -> MotionTrajectorySession:
        return cls(mode="capture", path=path, metadata=metadata)

    @classmethod
    def replay(cls, path: Path, expected_metadata: dict[str, Any]) -> MotionTrajectorySession:
        return cls(mode="replay", path=path, metadata=expected_metadata)

    def _load_replay(self) -> None:
        from safetensors import safe_open

        if not self.path.is_file():
            raise FileNotFoundError(f"Motion trajectory not found: {self.path}")
        with safe_open(self.path, framework="pt", device="cpu") as handle:
            stored_metadata = dict(handle.metadata() or {})
            normalized = handle.get_tensor("normalized_motion").float().contiguous()
            raw_rotation = handle.get_tensor("raw_rotation_rotvec_rad").float().contiguous()
            expression = handle.get_tensor("expression_normalized").float().contiguous()
            lengths_tensor = handle.get_tensor("chunk_lengths").to(dtype=torch.int64)
        if stored_metadata.get("schema_version") != str(_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported trajectory schema {stored_metadata.get('schema_version')!r}"
            )
        expected_fingerprint = stored_metadata.get("fingerprint_sha256")
        actual_fingerprint = _fingerprint(
            normalized,
            raw_rotation,
            lengths_tensor,
            stored_metadata,
        )
        if expected_fingerprint != actual_fingerprint:
            raise ValueError(
                "Motion trajectory fingerprint mismatch: "
                f"stored={expected_fingerprint}, actual={actual_fingerprint}"
            )
        for key, expected in self.metadata.items():
            if key == "schema_version":
                continue
            actual = stored_metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"Motion replay metadata mismatch for {key}: "
                    f"expected {expected!r}, captured {actual!r}"
                )
        lengths = [int(value) for value in lengths_tensor.tolist()]
        if any(length <= 0 for length in lengths) or sum(lengths) != normalized.shape[0]:
            raise ValueError("Motion trajectory has invalid chunk lengths")
        if raw_rotation.shape != (normalized.shape[0], 3):
            raise ValueError("Motion trajectory has invalid raw rotation shape")
        if not torch.equal(expression, normalized[:, 3:]):
            raise ValueError("Motion trajectory expression tensor does not match normalized motion")
        self.metadata = stored_metadata
        self._replay_motion = normalized
        self._replay_lengths = lengths
        self.fingerprint = actual_fingerprint

    def __enter__(self) -> MotionTrajectorySession:
        if _ACTIVE.get() is not None:
            raise RuntimeError("A motion trajectory session is already active")
        if self.mode == "capture":
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            existing = self.path if self.path.exists() else temporary
            if existing.exists():
                raise FileExistsError(
                    f"Motion capture path already exists; refusing to overwrite: {existing}"
                )
        self._token = _ACTIVE.set(self)
        record_motion_trajectory(
            event="session_start",
            mode=self.mode,
            path=str(self.path),
            fingerprint=self.fingerprint,
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
                if self.mode == "capture":
                    self._save_capture()
                elif self._cursor != len(self._replay_lengths):
                    raise ValueError(
                        "Motion replay ended before all chunks were consumed: "
                        f"{self._cursor}/{len(self._replay_lengths)}"
                    )
                record_motion_trajectory(
                    event="session_complete",
                    mode=self.mode,
                    path=str(self.path),
                    fingerprint=self.fingerprint,
                    chunks=self._cursor if self.mode == "replay" else len(self._chunks),
                )
        finally:
            if self._token is not None:
                _ACTIVE.reset(self._token)

    def intercept(
        self,
        generated: torch.Tensor,
        *,
        so3_offset: torch.Tensor,
        so3_scale: torch.Tensor,
    ) -> torch.Tensor:
        if generated.ndim != 3 or generated.shape[0] != 1:
            raise ValueError(
                f"Raw normalized motion must have shape (1, T, F), got {tuple(generated.shape)}"
            )
        if self.mode == "capture":
            captured = generated[0].detach().float().cpu().contiguous()
            self._chunks.append(captured)
            self._rotation_chunks.append(
                (generated[0, :, :3] * so3_scale + so3_offset)
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
            self._chunk_lengths.append(captured.shape[0])
            chunk_index = len(self._chunks) - 1
            record_motion_trajectory(
                event="chunk",
                mode=self.mode,
                chunk_index=chunk_index,
                frames=captured.shape[0],
            )
            return generated

        assert self._replay_motion is not None
        if self._cursor >= len(self._replay_lengths):
            raise ValueError("Motion replay requested more chunks than the capture contains")
        length = self._replay_lengths[self._cursor]
        start = self._frame_cursor
        replayed = self._replay_motion[start : start + length]
        if tuple(replayed.shape) != tuple(generated.shape[1:]):
            raise ValueError(
                "Motion replay chunk shape mismatch: "
                f"captured={tuple(replayed.shape)}, runtime={tuple(generated.shape[1:])}"
            )
        chunk_index = self._cursor
        self._cursor += 1
        self._frame_cursor += length
        record_motion_trajectory(
            event="chunk",
            mode=self.mode,
            chunk_index=chunk_index,
            frames=length,
            fingerprint=self.fingerprint,
        )
        return replayed.to(device=generated.device, dtype=generated.dtype).unsqueeze(0)

    def _save_capture(self) -> None:
        from safetensors.torch import save_file

        if not self._chunks:
            raise ValueError("Cannot save an empty motion trajectory")
        normalized = torch.cat(self._chunks, dim=0).contiguous()
        raw_rotation = torch.cat(self._rotation_chunks, dim=0).contiguous()
        lengths = torch.tensor(self._chunk_lengths, dtype=torch.int64)
        self.metadata.update(
            {
                "frames": str(normalized.shape[0]),
                "features": str(normalized.shape[1]),
                "chunks": str(len(self._chunks)),
            }
        )
        self.fingerprint = _fingerprint(normalized, raw_rotation, lengths, self.metadata)
        self.metadata["fingerprint_sha256"] = self.fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        save_file(
            {
                "normalized_motion": normalized,
                "raw_rotation_rotvec_rad": raw_rotation,
                "expression_normalized": normalized[:, 3:].contiguous(),
                "chunk_lengths": lengths,
            },
            str(temporary),
            metadata=self.metadata,
        )
        temporary.replace(self.path)


def intercept_raw_motion(
    generated: torch.Tensor,
    *,
    so3_offset: torch.Tensor,
    so3_scale: torch.Tensor,
) -> torch.Tensor:
    session = _ACTIVE.get()
    return (
        generated
        if session is None
        else session.intercept(
            generated,
            so3_offset=so3_offset,
            so3_scale=so3_scale,
        )
    )


def active_trajectory_fingerprint() -> str | None:
    session = _ACTIVE.get()
    return None if session is None else session.fingerprint


__all__ = [
    "MotionTrajectorySession",
    "active_trajectory_fingerprint",
    "intercept_raw_motion",
]
