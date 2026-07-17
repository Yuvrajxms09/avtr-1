import math
from pathlib import Path

import numpy as np
from scripts.sweep_rotation_stabilization import (
    ROTATION_CONFIGS,
    _expression_configs,
    _json_safe,
    _rotvec_kinematics_degrees,
)


def _x_axis_rotvecs(degrees: list[float]) -> np.ndarray:
    values = np.zeros((len(degrees), 3), dtype=np.float64)
    values[:, 0] = np.radians(degrees)
    return values


def test_rotvec_kinematics_preserve_direction_changes() -> None:
    step, acceleration, jerk = _rotvec_kinematics_degrees(_x_axis_rotvecs([0.0, 1.0, 0.0, -1.0]))

    np.testing.assert_allclose(step, [1.0, 1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(acceleration, [2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(jerk, [2.0], atol=1e-10)


def test_sweep_presets_define_motion_mode_once() -> None:
    expression_configs = _expression_configs(Path("profile.json"))

    assert all(config.arguments.count("--motion-stabilization") <= 1 for config in ROTATION_CONFIGS)
    assert all(
        config.arguments.count("--motion-stabilization") == 1 for config in expression_configs
    )


def test_json_safe_replaces_non_finite_values() -> None:
    assert _json_safe({"nan": math.nan, "values": [math.inf, 1.0]}) == {
        "nan": None,
        "values": [None, 1.0],
    }
