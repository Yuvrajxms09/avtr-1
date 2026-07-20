# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Stateful chunk-level speaking/listening guidance selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from avtr1_renderer.types import TurnAwareGuidanceOptions

ObservedTurn = Literal["speaking", "listening", "overlap", "silence"]

_STATE_CODE = {"speaking": 1, "listening": 2}
_STATE_NAME = {value: key for key, value in _STATE_CODE.items()}


@dataclass(slots=True, frozen=True)
class TurnGuidanceState:
    stable_state_code: int
    pending_state_code: int
    pending_chunks: int
    state_age_chunks: int
    effective_cfg_other_audio: float


@dataclass(slots=True, frozen=True)
class TurnGuidanceResult:
    effective_cfg_other_audio: float
    state: TurnGuidanceState | None
    observed_turn: ObservedTurn
    stable_turn: str | None
    state_changed: bool
    speech_rms: float
    listen_rms: float


def _observed_turn(
    options: TurnAwareGuidanceOptions,
    *,
    state: TurnGuidanceState | None,
    speech_rms: float,
    listen_rms: float,
) -> ObservedTurn:
    if options.mode != "auto":
        assert options.mode in {"speaking", "listening", "overlap", "silence"}
        return options.mode
    stable = _STATE_NAME.get(state.stable_state_code) if state is not None else None
    speech_threshold = options.speech_rms_off if stable == "speaking" else options.speech_rms_on
    listen_threshold = options.listen_rms_off if stable == "listening" else options.listen_rms_on
    speech_active = speech_rms >= speech_threshold
    listen_active = listen_rms >= listen_threshold
    if speech_active and listen_active:
        return "overlap"
    if speech_active:
        return "speaking"
    if listen_active:
        return "listening"
    return "silence"


def _move_toward(value: float, target: float, maximum_step: float) -> float:
    delta = target - value
    if abs(delta) <= maximum_step or maximum_step <= 0:
        return target
    return value + math.copysign(maximum_step, delta)


def update_turn_guidance(
    options: TurnAwareGuidanceOptions,
    *,
    state: TurnGuidanceState | None,
    fallback_cfg_other_audio: float,
    speech_rms: float,
    listen_rms: float,
) -> TurnGuidanceResult:
    options.validate()
    if not options.enabled:
        return TurnGuidanceResult(
            effective_cfg_other_audio=fallback_cfg_other_audio,
            state=None,
            observed_turn="silence",
            stable_turn=None,
            state_changed=state is not None,
            speech_rms=speech_rms,
            listen_rms=listen_rms,
        )

    observed = _observed_turn(
        options,
        state=state,
        speech_rms=speech_rms,
        listen_rms=listen_rms,
    )
    observed_code = _STATE_CODE.get(observed, 0)
    if state is None:
        state = TurnGuidanceState(
            # Keep the existing/default listening guidance until a new state
            # satisfies the same hysteresis contract as every later switch.
            stable_state_code=_STATE_CODE["listening"],
            pending_state_code=0,
            pending_chunks=0,
            state_age_chunks=0,
            effective_cfg_other_audio=options.listening_cfg_other_audio,
        )

    stable_code = state.stable_state_code
    pending_code = state.pending_state_code
    pending_chunks = state.pending_chunks
    state_age = state.state_age_chunks + 1
    state_changed = False

    # Overlap and silence are deliberately not stable CFG targets. They hold
    # the prior speaking/listening state and clear a partially observed switch.
    if observed_code == 0 or observed_code == stable_code:
        pending_code = 0
        pending_chunks = 0
    else:
        if pending_code == observed_code:
            pending_chunks += 1
        else:
            pending_code = observed_code
            pending_chunks = 1
        if (
            pending_chunks >= options.hysteresis_chunks
            and state_age >= options.minimum_state_chunks
        ):
            stable_code = observed_code
            pending_code = 0
            pending_chunks = 0
            state_age = 0
            state_changed = True

    target = (
        options.speaking_cfg_other_audio
        if stable_code == _STATE_CODE["speaking"]
        else options.listening_cfg_other_audio
    )
    full_range = abs(
        options.listening_cfg_other_audio - options.speaking_cfg_other_audio
    )
    maximum_step = full_range / options.transition_chunks
    effective = _move_toward(
        state.effective_cfg_other_audio,
        target,
        maximum_step,
    )
    next_state = TurnGuidanceState(
        stable_state_code=stable_code,
        pending_state_code=pending_code,
        pending_chunks=pending_chunks,
        state_age_chunks=state_age,
        effective_cfg_other_audio=effective,
    )
    return TurnGuidanceResult(
        effective_cfg_other_audio=effective,
        state=next_state,
        observed_turn=observed,
        stable_turn=_STATE_NAME[stable_code],
        state_changed=state_changed,
        speech_rms=speech_rms,
        listen_rms=listen_rms,
    )


__all__ = [
    "ObservedTurn",
    "TurnGuidanceResult",
    "TurnGuidanceState",
    "update_turn_guidance",
]
