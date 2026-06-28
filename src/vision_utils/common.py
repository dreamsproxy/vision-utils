from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_exts(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_EXTS
    exts = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        exts.append(item)
    return tuple(exts) if exts else DEFAULT_EXTS


def iter_images(root: Path, exts: Iterable[str] = DEFAULT_EXTS, recursive: bool = True) -> list[Path]:
    root = Path(root)
    suffixes = {ext.lower() for ext in exts}
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in suffixes)


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def output_path_for(input_path: Path, input_root: Path, output_root: Path) -> Path:
    relative = input_path.relative_to(input_root)
    return output_root / relative


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}: {path}")
    encoded.tofile(str(path))


def progress(items, *, desc: str, quiet: bool = False):
    # Quiet suppresses per-item logs only. Progress bars stay on by default.
    return tqdm(items, desc=desc, unit="img", disable=False if not quiet else False)
