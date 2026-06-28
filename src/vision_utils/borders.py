from __future__ import annotations

import csv
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import ensure_dir, iter_images, output_path_for, progress, read_image, write_image


@dataclass(frozen=True)
class BorderRecord:
    full_path: str
    dir: str
    filename: str
    y0: int
    x0: int
    y1: int
    x1: int
    cropped: bool


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def detect_content_bbox(
    image: np.ndarray,
    *,
    std_threshold: float = 2.0,
    mean_threshold: float = 8.0,
    min_span: int = 4,
) -> tuple[int, int, int, int] | None:
    """Detect an approximate non-border bounding box.

    The detector combines row/column variance with distance from the median boundary
    intensity. This keeps the method simple and useful for dataset cleanup without
    pretending to solve every pathological border case.
    """
    gray = _gray(image)
    h, w = gray.shape[:2]
    if h < min_span or w < min_span:
        return None

    border_samples = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    border_median = float(np.median(border_samples))

    row_std = gray.std(axis=1)
    col_std = gray.std(axis=0)
    row_mean_delta = np.abs(gray.mean(axis=1) - border_median)
    col_mean_delta = np.abs(gray.mean(axis=0) - border_median)

    row_mask = (row_std > std_threshold) | (row_mean_delta > mean_threshold)
    col_mask = (col_std > std_threshold) | (col_mean_delta > mean_threshold)

    ys = np.flatnonzero(row_mask)
    xs = np.flatnonzero(col_mask)
    if ys.size == 0 or xs.size == 0:
        return None

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    if (y1 - y0) < min_span or (x1 - x0) < min_span:
        return None
    return y0, x0, y1, x1


def inspect_border(path: Path, *, std_threshold: float, mean_threshold: float) -> BorderRecord:
    try:
        image = read_image(path)
        bbox = detect_content_bbox(
            image,
            std_threshold=std_threshold,
            mean_threshold=mean_threshold,
        )
        if bbox is None:
            return BorderRecord(str(path), str(path.parent), path.name, -1, -1, -1, -1, False)
        y0, x0, y1, x1 = bbox
        h, w = image.shape[:2]
        cropped = (y0, x0, y1, x1) != (0, 0, h, w)
        return BorderRecord(str(path), str(path.parent), path.name, y0, x0, y1, x1, cropped)
    except Exception:
        return BorderRecord(str(path), str(path.parent), path.name, -1, -1, -1, -1, False)


def detect_borders(
    input_dir: Path,
    output_csv: Path,
    *,
    exts: tuple[str, ...],
    recursive: bool = True,
    workers: int = 4,
    std_threshold: float = 2.0,
    mean_threshold: float = 8.0,
    quiet: bool = False,
) -> list[BorderRecord]:
    paths = iter_images(input_dir, exts, recursive=recursive)
    worker_count = max(1, workers)

    def run(path: Path) -> BorderRecord:
        return inspect_border(path, std_threshold=std_threshold, mean_threshold=mean_threshold)

    if worker_count == 1:
        records = [run(path) for path in progress(paths, desc="detect-borders", quiet=quiet)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            records = list(progress(pool.map(run, paths), desc="detect-borders", quiet=quiet))

    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["full_path", "dir", "filename", "y0", "x0", "y1", "x1", "cropped"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return records


def crop_borders(
    input_dir: Path,
    output_dir: Path,
    *,
    exts: tuple[str, ...],
    recursive: bool = True,
    workers: int = 4,
    std_threshold: float = 2.0,
    mean_threshold: float = 8.0,
    copy_uncropped: bool = True,
    csv_path: Path | None = None,
    quiet: bool = False,
) -> list[BorderRecord]:
    paths = iter_images(input_dir, exts, recursive=recursive)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    def run(path: Path) -> BorderRecord:
        record = inspect_border(path, std_threshold=std_threshold, mean_threshold=mean_threshold)
        out_path = output_path_for(path, input_dir, output_dir)
        if record.cropped:
            image = read_image(path)
            cropped = image[record.y0 : record.y1, record.x0 : record.x1]
            write_image(out_path, cropped)
        elif copy_uncropped:
            ensure_dir(out_path.parent)
            shutil.copy2(path, out_path)
        return record

    worker_count = max(1, workers)
    if worker_count == 1:
        records = [run(path) for path in progress(paths, desc="crop-borders", quiet=quiet)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            records = list(progress(pool.map(run, paths), desc="crop-borders", quiet=quiet))

    if csv_path is not None:
        ensure_dir(csv_path.parent)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["full_path", "dir", "filename", "y0", "x0", "y1", "x1", "cropped"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
    return records
