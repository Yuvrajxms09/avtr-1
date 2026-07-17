import torch

from avtr1_renderer.avtr1_motion_generator import (
    AVTR1State,
    state_from_safetensors,
    state_to_safetensors,
)
from avtr1_renderer.motion_stabilizer import MotionStabilizerState


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
        rotation_raw_position=torch.tensor([0.2, 0.3, 0.4]),
        rotation_raw_velocity=torch.tensor([0.02, 0.03, 0.04]),
        expression_position=torch.arange(39, dtype=torch.float32),
        expression_velocity=torch.ones(39),
        expression_raw_position=torch.arange(39, dtype=torch.float32) + 1,
        expression_raw_velocity=torch.full((39,), 2.0),
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
        "rotation_raw_position",
        "rotation_raw_velocity",
        "expression_position",
        "expression_velocity",
        "expression_raw_position",
        "expression_raw_velocity",
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
