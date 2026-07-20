from dataclasses import replace

import torch

from avtr1_renderer.avtr1_motion_generator import (
    AVTR1State,
    state_from_safetensors,
    state_to_safetensors,
)
from avtr1_renderer.keypoint_stabilizer import KeypointStabilizerState
from avtr1_renderer.motion_stabilizer import MotionStabilizerState
from avtr1_renderer.turn_guidance import TurnGuidanceState


def _avtr_state(stabilizer: MotionStabilizerState | None) -> AVTR1State:
    return AVTR1State(
        audio_prev_speech=torch.zeros(4),
        audio_prev_listen=torch.zeros(4),
        audio_features=torch.zeros((1, 2, 6)),
        past_cond=torch.zeros((1, 2, 42)),
        noise_shared=torch.zeros((1, 1, 42)),
        stabilizer=stabilizer,
    )


def test_state_codec_preserves_stabilizer_carry() -> None:
    stabilizer = MotionStabilizerState(
        mode_code=3,
        rotation_position=torch.tensor([0.1, 0.2, 0.3]),
        rotation_velocity=torch.tensor([0.01, 0.02, 0.03]),
        rotation_acceleration=torch.tensor([0.001, 0.002, 0.003]),
        rotation_raw_position=torch.tensor([0.2, 0.3, 0.4]),
        rotation_raw_velocity=torch.tensor([0.02, 0.03, 0.04]),
        rotation_filter_input=torch.tensor([0.3, 0.4, 0.5]),
        rotation_filter_derivative=torch.tensor([0.03, 0.04, 0.05]),
        expression_position=torch.arange(39, dtype=torch.float32),
        expression_velocity=torch.ones(39),
        expression_acceleration=torch.full((39,), 0.5),
        expression_raw_position=torch.arange(39, dtype=torch.float32) + 1,
        expression_raw_velocity=torch.full((39,), 2.0),
        expression_filter_input=torch.arange(39, dtype=torch.float32) + 2,
        expression_filter_derivative=torch.full((39,), 3.0),
    )

    restored = state_from_safetensors(
        state_to_safetensors(_avtr_state(stabilizer)),
        device="cpu",
    )

    assert restored.stabilizer is not None
    assert restored.stabilizer.mode_code == 3
    for name in (
        "rotation_position",
        "rotation_velocity",
        "rotation_acceleration",
        "rotation_raw_position",
        "rotation_raw_velocity",
        "rotation_filter_input",
        "rotation_filter_derivative",
        "expression_position",
        "expression_velocity",
        "expression_acceleration",
        "expression_raw_position",
        "expression_raw_velocity",
        "expression_filter_input",
        "expression_filter_derivative",
    ):
        torch.testing.assert_close(
            getattr(restored.stabilizer, name),
            getattr(stabilizer, name),
        )


def test_state_codec_remains_compatible_when_stabilization_is_disabled() -> None:
    restored = state_from_safetensors(
        state_to_safetensors(_avtr_state(None)),
        device="cpu",
    )

    assert restored.stabilizer is None
    assert restored.keypoint_stabilizer is None
    assert restored.turn_guidance is None


def test_state_codec_preserves_keypoint_stabilizer_carry() -> None:
    keypoint = KeypointStabilizerState(
        configuration_code=91,
        stitch_position=torch.randn(21, 3),
        stitch_filter_input=torch.randn(21, 3),
        stitch_filter_derivative=torch.randn(21, 3),
        residual_position=torch.randn(8, 3),
        residual_filter_input=torch.randn(8, 3),
        residual_filter_derivative=torch.randn(8, 3),
    )
    state = replace(_avtr_state(None), keypoint_stabilizer=keypoint)
    restored = state_from_safetensors(state_to_safetensors(state), device="cpu")
    assert restored.keypoint_stabilizer is not None
    assert restored.keypoint_stabilizer.configuration_code == 91
    for name in (
        "stitch_position",
        "stitch_filter_input",
        "stitch_filter_derivative",
        "residual_position",
        "residual_filter_input",
        "residual_filter_derivative",
    ):
        torch.testing.assert_close(
            getattr(restored.keypoint_stabilizer, name),
            getattr(keypoint, name),
        )


def test_state_codec_preserves_turn_guidance_carry() -> None:
    guidance = TurnGuidanceState(
        stable_state_code=1,
        pending_state_code=2,
        pending_chunks=1,
        state_age_chunks=7,
        effective_cfg_other_audio=1.25,
    )
    state = replace(_avtr_state(None), turn_guidance=guidance)
    restored = state_from_safetensors(state_to_safetensors(state), device="cpu")

    assert restored.turn_guidance is not None
    assert restored.turn_guidance.stable_state_code == 1
    assert restored.turn_guidance.pending_state_code == 2
    assert restored.turn_guidance.pending_chunks == 1
    assert restored.turn_guidance.state_age_chunks == 7
    assert restored.turn_guidance.effective_cfg_other_audio == 1.25
