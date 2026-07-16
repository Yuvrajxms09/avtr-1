# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Output-frame sizing helpers."""

from __future__ import annotations

from pathlib import Path

import cv2


def read_native_output_size(image_path: str | Path) -> tuple[int, int]:
    """Return the image's ``(height, width)`` aligned for YUV420 encoding.

    YUV420 requires even spatial dimensions. Odd dimensions are rounded down
    by one pixel; no aspect-ratio-changing resize is otherwise introduced.
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    output_size = (height - height % 2, width - width % 2)
    if min(output_size) < 2:
        raise ValueError(
            f"Image is too small for YUV420 output: {image_path} has size {width}x{height}"
        )
    return output_size


__all__ = ["read_native_output_size"]
