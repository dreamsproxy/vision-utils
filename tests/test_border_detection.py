from __future__ import annotations

import numpy as np

from vision_utils.borders import detect_content_bbox


def test_detect_content_bbox_on_black_border() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:15, 7:22] = 255

    assert detect_content_bbox(image) == (5, 7, 15, 22)
