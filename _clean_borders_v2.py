#!/usr/bin/env python3
"""
Batch border cleaner with bbox CSV logging.

Default behavior:
  - Detects only edge-connected borders, not arbitrary low-variance rows/cols inside the image.
  - Uses a hybrid detector:
      1. low row/column variance, and
      2. similarity to colors estimated from image corners.
  - Writes cropped images.
  - Writes one CSV row per input image:
      full_path, dir, filename, y0, x0, y1, x1, cropped

If no crop bbox is detected, y0/x0/y1/x1 are written as -1 and cropped is False.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from glob import glob
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

CSV_HEADERS = ["full_path", "dir", "filename", "y0", "x0", "y1", "x1", "cropped"]


def corner_colors_bgr(img: np.ndarray, patch_frac: float = 0.02, patch_min: int = 4) -> np.ndarray:
    """
    Estimate likely border/background colors from the four image corners.

    Returns:
        (4, C) float32 array. For normal cv2 images C=3, BGR order.
    """
    h, w = img.shape[:2]
    ph = min(h, max(patch_min, int(round(h * patch_frac))))
    pw = min(w, max(patch_min, int(round(w * patch_frac))))

    patches = [
        img[:ph, :pw],
        img[:ph, w - pw :],
        img[h - ph :, :pw],
        img[h - ph :, w - pw :],
    ]

    colors = []
    for patch in patches:
        flat = patch.reshape(-1, patch.shape[-1]).astype(np.float32)
        colors.append(np.median(flat, axis=0))

    return np.stack(colors, axis=0).astype(np.float32)


def nearest_corner_color_distance(img: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    Pixel-wise Euclidean distance to the nearest estimated corner color.
    """
    img_f = img.astype(np.float32)
    # Shape: H x W x 4 x C
    diff = img_f[:, :, None, :] - colors[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return np.sqrt(np.min(dist2, axis=-1))


def contiguous_true_from_start(mask: np.ndarray, max_count: int) -> int:
    """
    Count contiguous True values from the beginning of a boolean mask.
    """
    max_count = max(0, min(int(max_count), int(mask.size)))
    count = 0
    for v in mask[:max_count]:
        if bool(v):
            count += 1
        else:
            break
    return count


def contiguous_true_from_end(mask: np.ndarray, max_count: int) -> int:
    """
    Count contiguous True values from the end of a boolean mask.
    """
    max_count = max(0, min(int(max_count), int(mask.size)))
    count = 0
    for v in mask[::-1][:max_count]:
        if bool(v):
            count += 1
        else:
            break
    return count


def detect_border_bbox(
    img: np.ndarray,
    var_thresh: float = 8.0,
    color_thresh: float = 12.0,
    border_frac: float = 0.98,
    max_crop_frac: float = 0.45,
    corner_patch_frac: float = 0.02,
    pad: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect an edge-connected border crop box.

    Returns:
        (y0, x0, y1, x1) using Python slicing convention, or None if no crop is detected.

    Notes:
        - y1/x1 are exclusive crop coordinates.
        - The detector only scans inward from the four outer image edges. This prevents
          internal low-variance separators from being mistaken as crop borders.
    """
    if img is None or img.ndim < 2:
        return None

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None

    # Normalize grayscale to H x W x 1 so row/column variance logic is consistent.
    if img.ndim == 2:
        work = img[:, :, None]
    else:
        work = img

    work_f = work.astype(np.float32)

    # Current method, vectorized: row/column uniformity by variance.
    row_var = np.var(work_f, axis=(1, 2))
    col_var = np.var(work_f, axis=(0, 2))
    row_low_var = row_var <= var_thresh
    col_low_var = col_var <= var_thresh

    # Additional robust signal: similarity to likely border colors sampled from corners.
    colors = corner_colors_bgr(work, patch_frac=corner_patch_frac)
    dist = nearest_corner_color_distance(work, colors)
    near_border_color = dist <= color_thresh

    row_border_color_frac = np.mean(near_border_color, axis=1)
    col_border_color_frac = np.mean(near_border_color, axis=0)

    row_border_like = row_low_var | (row_border_color_frac >= border_frac)
    col_border_like = col_low_var | (col_border_color_frac >= border_frac)

    max_y_crop = max(1, int(round(h * max_crop_frac)))
    max_x_crop = max(1, int(round(w * max_crop_frac)))

    top = contiguous_true_from_start(row_border_like, max_y_crop)
    bottom = contiguous_true_from_end(row_border_like, max_y_crop)
    left = contiguous_true_from_start(col_border_like, max_x_crop)
    right = contiguous_true_from_end(col_border_like, max_x_crop)

    y0 = top
    y1 = h - bottom
    x0 = left
    x1 = w - right

    # Add optional safety padding back into the crop to avoid shaving off content.
    if pad > 0:
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h, y1 + pad)
        x1 = min(w, x1 + pad)

    # Invalid or empty crop.
    if y0 >= y1 or x0 >= x1:
        return None

    # No actual crop detected.
    if y0 == 0 and x0 == 0 and y1 == h and x1 == w:
        return None

    return int(y0), int(x0), int(y1), int(x1)


def output_path_for(src_path: Path, output_dir: Path, input_root: Optional[Path]) -> Path:
    """
    Resolve output path while optionally preserving relative directory structure.
    """
    if input_root is not None:
        try:
            rel = src_path.resolve().relative_to(input_root.resolve())
        except ValueError:
            rel = Path(src_path.name)
    else:
        rel = Path(src_path.name)

    return output_dir / rel


def iter_paths(patterns: Iterable[str], recursive: bool = True) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob(pattern, recursive=recursive))
    return sorted(set(p for p in paths if p.is_file()))


def process_image(
    src_path: Path,
    dst_path: Path,
    var_thresh: float,
    color_thresh: float,
    border_frac: float,
    max_crop_frac: float,
    corner_patch_frac: float,
    pad: int,
    write_uncropped: bool,
) -> dict:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)

    if img is None:
        # Treat unreadable files as no bbox. Do not write an output image.
        return {
            "full_path": str(src_path),
            "dir": str(src_path.parent),
            "filename": src_path.name,
            "y0": -1,
            "x0": -1,
            "y1": -1,
            "x1": -1,
            "cropped": False,
        }

    bbox = detect_border_bbox(
        img,
        var_thresh=var_thresh,
        color_thresh=color_thresh,
        border_frac=border_frac,
        max_crop_frac=max_crop_frac,
        corner_patch_frac=corner_patch_frac,
        pad=pad,
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if bbox is None:
        if write_uncropped:
            cv2.imwrite(str(dst_path), img)
        y0 = x0 = y1 = x1 = -1
        cropped = False
    else:
        y0, x0, y1, x1 = bbox
        cropped_img = img[y0:y1, x0:x1]
        cv2.imwrite(str(dst_path), cropped_img)
        cropped = True

    return {
        "full_path": str(src_path),
        "dir": str(src_path.parent),
        "filename": src_path.name,
        "y0": np.int32(y0).item(),
        "x0": np.int32(x0).item(),
        "y1": np.int32(y1).item(),
        "x1": np.int32(x1).item(),
        "cropped": bool(cropped),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean edge borders from images and write bbox CSV metadata.")
    ap.add_argument(
        "--input-glob",
        nargs="+",
        default=["./some_dataset/*.png"],
        help="Input glob pattern(s). Example: './renamed/**/*.png'",
    )
    ap.add_argument(
        "--output-dir",
        default="./cleaned",
        help="Directory where cleaned images are written.",
    )
    ap.add_argument(
        "--input-root",
        default=None,
        help="Optional root directory used to preserve relative paths under --output-dir.",
    )
    ap.add_argument(
        "--csv",
        default="./border_bboxes.csv",
        help="CSV path for bbox metadata.",
    )
    ap.add_argument("--var-thresh", type=float, default=8.0, help="Low-variance threshold for border rows/cols.")
    ap.add_argument(
        "--color-thresh",
        type=float,
        default=12.0,
        help="Pixel color distance threshold from estimated corner border colors.",
    )
    ap.add_argument(
        "--border-frac",
        type=float,
        default=0.98,
        help="Required row/col fraction near a corner border color to mark as border-like.",
    )
    ap.add_argument(
        "--max-crop-frac",
        type=float,
        default=0.45,
        help="Safety cap: maximum fraction cropped from any one side.",
    )
    ap.add_argument(
        "--corner-patch-frac",
        type=float,
        default=0.02,
        help="Corner patch size as fraction of image dimension for border color estimation.",
    )
    ap.add_argument("--pad", type=int, default=0, help="Pixels to add back around detected content.")
    ap.add_argument(
        "--write-uncropped",
        action="store_true",
        help="Also copy images where no border bbox was detected.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    input_root = Path(args.input_root) if args.input_root else None
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    paths = iter_paths(args.input_glob, recursive=True)
    print(f"Found {len(paths)} image(s).")

    rows = []
    for idx, src_path in enumerate(paths, start=1):
        dst_path = output_path_for(src_path, output_dir, input_root)
        row = process_image(
            src_path=src_path,
            dst_path=dst_path,
            var_thresh=args.var_thresh,
            color_thresh=args.color_thresh,
            border_frac=args.border_frac,
            max_crop_frac=args.max_crop_frac,
            corner_patch_frac=args.corner_patch_frac,
            pad=args.pad,
            write_uncropped=args.write_uncropped,
        )
        rows.append(row)

        if row["cropped"]:
            print(f"[{idx}/{len(paths)}] cropped: {src_path} -> {dst_path} bbox=({row['y0']}, {row['x0']}, {row['y1']}, {row['x1']})")
        else:
            print(f"[{idx}/{len(paths)}] no bbox: {src_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
