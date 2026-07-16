# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

from pathlib import Path

import cv2
import numpy as np
import pytest

from avtr1_renderer.frame_size import read_native_output_size


def _write_image(path: Path, *, height: int, width: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_read_native_output_size_preserves_even_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    _write_image(image_path, height=650, width=500)

    assert read_native_output_size(image_path) == (650, 500)


def test_read_native_output_size_aligns_odd_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    _write_image(image_path, height=313, width=315)

    assert read_native_output_size(image_path) == (312, 314)


def test_read_native_output_size_rejects_unreadable_image(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Could not load image"):
        read_native_output_size(image_path)


def test_read_native_output_size_rejects_tiny_image(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    _write_image(image_path, height=1, width=2)

    with pytest.raises(ValueError, match="too small"):
        read_native_output_size(image_path)
