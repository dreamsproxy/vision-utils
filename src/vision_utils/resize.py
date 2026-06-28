from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from .common import ensure_dir, iter_images, output_path_for, progress, read_image, write_image


def _target_size(width: int, height: int, max_side: int | None, exact_width: int | None, exact_height: int | None) -> tuple[int, int]:
    if exact_width is not None and exact_height is not None:
        return exact_width, exact_height
    if max_side is None:
        raise ValueError("Either max_side or exact width/height must be provided.")
    scale = float(max_side) / float(max(width, height))
    if scale <= 0:
        raise ValueError("max_side must be positive.")
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_one(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    max_side: int | None,
    exact_width: int | None,
    exact_height: int | None,
    interpolation: int,
) -> Path:
    image = read_image(input_path)
    h, w = image.shape[:2]
    out_w, out_h = _target_size(w, h, max_side, exact_width, exact_height)
    resized = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
    out_path = output_path_for(input_path, input_root, output_root)
    write_image(out_path, resized)
    return out_path


def resize_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    exts: tuple[str, ...],
    recursive: bool = True,
    workers: int = 4,
    max_side: int | None = None,
    exact_width: int | None = None,
    exact_height: int | None = None,
    interpolation_name: str = "area",
    quiet: bool = False,
) -> list[Path]:
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if interpolation_name not in interpolation_map:
        raise ValueError(f"Unsupported interpolation: {interpolation_name}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    paths = iter_images(input_dir, exts, recursive=recursive)

    def run(path: Path) -> Path:
        return resize_one(
            path,
            input_dir,
            output_dir,
            max_side=max_side,
            exact_width=exact_width,
            exact_height=exact_height,
            interpolation=interpolation_map[interpolation_name],
        )

    worker_count = max(1, workers)
    if worker_count == 1:
        return [run(path) for path in progress(paths, desc="resize", quiet=quiet)]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(progress(pool.map(run, paths), desc="resize", quiet=quiet))
