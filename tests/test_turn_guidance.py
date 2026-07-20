import pytest

from avtr1_renderer.turn_guidance import update_turn_guidance
from avtr1_renderer.types import TurnAwareGuidanceOptions


def _update(options, state, speech, listen):
    return update_turn_guidance(
        options,
        state=state,
        fallback_cfg_other_audio=2.0,
        speech_rms=speech,
        listen_rms=listen,
    )


def test_disabled_guidance_preserves_static_cfg_and_has_no_state() -> None:
    result = _update(TurnAwareGuidanceOptions(), None, 0.5, 0.0)

    assert result.effective_cfg_other_audio == 2.0
    assert result.state is None


def test_auto_guidance_hysteresis_and_transition_are_chunk_continuous() -> None:
    options = TurnAwareGuidanceOptions(
        mode="auto",
        hysteresis_chunks=2,
        minimum_state_chunks=2,
        transition_chunks=2,
    )
    first = _update(options, None, 0.1, 0.0)
    second = _update(options, first.state, 0.1, 0.0)
    settled = _update(options, second.state, 0.1, 0.0)
    pending = _update(options, settled.state, 0.0, 0.1)
    switched = _update(options, pending.state, 0.0, 0.1)

    assert first.stable_turn == "listening"
    assert first.effective_cfg_other_audio == pytest.approx(2.0)
    assert second.stable_turn == "speaking"
    assert second.effective_cfg_other_audio == pytest.approx(1.5)
    assert settled.effective_cfg_other_audio == pytest.approx(1.0)
    assert pending.stable_turn == "speaking"
    assert not pending.state_changed
    assert switched.stable_turn == "listening"
    assert switched.state_changed
    assert switched.effective_cfg_other_audio == pytest.approx(1.5)


def test_overlap_and_silence_hold_the_previous_stable_target() -> None:
    options = TurnAwareGuidanceOptions(
        mode="auto",
        hysteresis_chunks=1,
        minimum_state_chunks=1,
        transition_chunks=1,
    )
    speaking = _update(options, None, 0.1, 0.0)
    overlap = _update(options, speaking.state, 0.1, 0.1)
    silence = _update(options, overlap.state, 0.0, 0.0)

    assert overlap.observed_turn == "overlap"
    assert silence.observed_turn == "silence"
    assert overlap.stable_turn == silence.stable_turn == "speaking"
    assert overlap.effective_cfg_other_audio == 1.0
    assert silence.effective_cfg_other_audio == 1.0


def test_explicit_turn_mode_bypasses_rms_classification() -> None:
    options = TurnAwareGuidanceOptions(
        mode="listening",
        hysteresis_chunks=1,
        minimum_state_chunks=1,
        transition_chunks=1,
    )
    result = _update(options, None, 1.0, 0.0)

    assert result.observed_turn == "listening"
    assert result.stable_turn == "listening"
    assert result.effective_cfg_other_audio == 2.0
