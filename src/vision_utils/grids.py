from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, iter_images, progress, read_image, write_image


def _fit_thumb(image: np.ndarray, thumb_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = float(thumb_size) / float(max(h, w))
    out_w = max(1, round(w * scale))
    out_h = max(1, round(h * scale))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    if resized.shape[2] == 4:
        resized = resized[:, :, :3]
    canvas = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
    y0 = (thumb_size - out_h) // 2
    x0 = (thumb_size - out_w) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return canvas


def make_sample_grid(
    input_dir: Path,
    output_path: Path,
    *,
    exts: tuple[str, ...],
    recursive: bool = True,
    limit: int = 64,
    rows: int | None = None,
    cols: int | None = None,
    thumb_size: int = 128,
    seed: int = 1,
    quiet: bool = False,
) -> Path:
    paths = iter_images(input_dir, exts, recursive=recursive)
    rng = random.Random(seed)
    rng.shuffle(paths)
    selected = paths[: max(1, limit)]
    if not selected:
        raise ValueError(f"No images found in: {input_dir}")

    if rows is None and cols is None:
        cols = math.ceil(math.sqrt(len(selected)))
        rows = math.ceil(len(selected) / cols)
    elif rows is None:
        rows = math.ceil(len(selected) / cols)
    elif cols is None:
        cols = math.ceil(len(selected) / rows)

    canvas = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    for idx, path in enumerate(progress(selected, desc="sample-grid", quiet=quiet)):
        image = read_image(path)
        thumb = _fit_thumb(image, thumb_size)
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break
        y0 = r * thumb_size
        x0 = c * thumb_size
        canvas[y0 : y0 + thumb_size, x0 : x0 + thumb_size] = thumb

    ensure_dir(output_path.parent)
    write_image(output_path, canvas)
    return output_path
