from __future__ import annotations

import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .common import ensure_dir, iter_images, progress, read_image


@dataclass(frozen=True)
class ManifestRecord:
    full_path: str
    dir: str
    filename: str
    sha256: str
    size_bytes: int
    width: int
    height: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path: Path) -> ManifestRecord:
    image = read_image(path)
    h, w = image.shape[:2]
    return ManifestRecord(
        full_path=str(path),
        dir=str(path.parent),
        filename=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        width=int(w),
        height=int(h),
    )


def build_manifest(
    input_dir: Path,
    output_csv: Path,
    *,
    exts: tuple[str, ...],
    recursive: bool = True,
    workers: int = 4,
    quiet: bool = False,
) -> list[ManifestRecord]:
    paths = iter_images(input_dir, exts, recursive=recursive)
    worker_count = max(1, workers)

    if worker_count == 1:
        records = [inspect_file(path) for path in progress(paths, desc="manifest", quiet=quiet)]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            records = list(progress(pool.map(inspect_file, paths), desc="manifest", quiet=quiet))

    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["full_path", "dir", "filename", "sha256", "size_bytes", "width", "height"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return records
