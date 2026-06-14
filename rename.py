#!/usr/bin/env python3
"""
Copy images into a separate output directory using SHA-256 filenames.

Default behavior:
    Hash the decoded pixel array using OpenCV.

    - Similar images with any pixel differences receive different names.
    - Metadata-only differences do not affect the hash.
    - Source files remain untouched.
    - Existing output files are never overwritten.
    - Exact duplicates are preserved with __dup001 suffixes.
    - The script performs a dry run unless --apply is passed.

Optional behavior:
    Use --hash-source file to hash the exact compressed file bytes instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from tqdm import tqdm

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class CopyRecord:
    source: Path
    digest: str
    target: Path | None = None


def path_key(path: Path) -> str:
    """
    Normalize paths for comparisons across operating systems.
    """
    return os.path.normcase(os.path.abspath(path))


def sha256_file_bytes(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Hash the exact compressed bytes stored on disk.

    Metadata changes, compression changes, and encoding changes will alter
    the resulting digest.
    """
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def sha256_decoded_pixels(path: Path) -> str:
    """
    Hash the decoded image pixels.

    Similar images are not considered identical. Any pixel-level difference
    changes the digest.

    Image shape and dtype are included in the digest to avoid ambiguity.
    """
    encoded = np.fromfile(str(path), dtype=np.uint8)

    if encoded.size == 0:
        raise ValueError("File is empty.")

    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("OpenCV could not decode the image.")

    image = np.ascontiguousarray(image)

    hasher = hashlib.sha256()
    hasher.update(b"decoded-pixel-sha256-v1\0")
    hasher.update(str(image.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(",".join(str(value) for value in image.shape).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(image.tobytes())

    return hasher.hexdigest()


def is_inside_directory(path: Path, directory: Path) -> bool:
    """
    Return True when path is inside directory.
    """
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def find_images(
    source_directory: Path,
    output_directory: Path,
    recursive: bool,
) -> list[Path]:
    """
    Locate supported images while excluding the output directory.

    Excluding the output directory prevents copied files from being scanned
    again if the output folder is placed inside the source directory.
    """
    iterator = (
        source_directory.rglob("*")
        if recursive
        else source_directory.iterdir()
    )

    return sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not is_inside_directory(path, output_directory)
        ),
        key=path_key,
    )


def make_available_target(
    output_directory: Path,
    digest: str,
    extension: str,
    reserved_targets: set[str],
) -> Path:
    """
    Find a collision-safe output filename.

    First candidate:
        <sha256>.<extension>

    Additional exact duplicates or existing files:
        <sha256>__dup001.<extension>
        <sha256>__dup002.<extension>
    """
    duplicate_index = 0

    while True:
        if duplicate_index == 0:
            filename = f"{digest}{extension}"
        else:
            filename = f"{digest}__dup{duplicate_index:03d}{extension}"

        candidate = output_directory / filename
        candidate_key = path_key(candidate)

        if candidate_key not in reserved_targets and not candidate.exists():
            reserved_targets.add(candidate_key)
            return candidate

        duplicate_index += 1


def create_copy_plan(
    records: list[CopyRecord],
    output_directory: Path,
) -> None:
    """
    Assign output paths without overwriting files.
    """
    reserved_targets: set[str] = set()

    for record in tqdm(records):
        record.target = make_available_target(
            output_directory=output_directory,
            digest=record.digest,
            extension=record.source.suffix.lower(),
            reserved_targets=reserved_targets,
        )


def apply_copy_plan(records: list[CopyRecord], output_directory: Path) -> None:
    """
    Copy source files to their hash-based output paths.

    shutil.copy2() preserves the original file bytes and filesystem metadata.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    copied_targets: list[Path] = []

    try:
        for record in tqdm(records):
            if record.target is None:
                raise RuntimeError("Internal copy-plan error: missing target path.")

            shutil.copy2(record.source, record.target)
            copied_targets.append(record.target)

    except Exception:
        print(
            "Copying failed. Removing files created during this run.",
            file=sys.stderr,
        )

        for target in tqdm(reversed(copied_targets)):
            try:
                target.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(
                    f"Could not remove partial output: {target} | {cleanup_error}",
                    file=sys.stderr,
                )

        raise


def write_manifest(
    manifest_path: Path,
    records: list[CopyRecord],
    hash_source: str,
) -> None:
    """
    Save the source-to-output mapping as a CSV file.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "source_path",
                "output_path",
                "sha256",
                "hash_source",
            ]
        )

        for record in tqdm(records):
            writer.writerow(
                [
                    str(record.source),
                    str(record.target),
                    record.digest,
                    hash_source,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy images into another directory using full SHA-256 filenames."
        )
    )

    parser.add_argument(
        "source_directory",
        type=Path,
        help="Directory containing the original images.",
    )

    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory that will receive the hash-named image copies.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in nested source directories.",
    )

    parser.add_argument(
        "--hash-source",
        choices=("pixels", "file"),
        default="pixels",
        help=(
            "Hash decoded pixels or exact compressed file bytes. "
            "Default: pixels."
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the files. Without this flag, only preview the operation.",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV file recording the source and output paths.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_directory = args.source_directory.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()

    if not source_directory.exists():
        print(
            f"Error: source directory does not exist: {source_directory}",
            file=sys.stderr,
        )
        return 1

    if not source_directory.is_dir():
        print(
            f"Error: source path is not a directory: {source_directory}",
            file=sys.stderr,
        )
        return 1

    if path_key(source_directory) == path_key(output_directory):
        print(
            "Error: source and output directories must be different.",
            file=sys.stderr,
        )
        return 1

    hash_function: Callable[[Path], str]

    if args.hash_source == "pixels":
        hash_function = sha256_decoded_pixels
    else:
        hash_function = sha256_file_bytes

    images = find_images(
        source_directory=source_directory,
        output_directory=output_directory,
        recursive=args.recursive,
    )

    if not images:
        print("No supported images found.")
        return 0

    records: list[CopyRecord] = []
    failed: list[tuple[Path, str]] = []

    for image_path in tqdm(images):
        try:
            digest = hash_function(image_path)

            records.append(
                CopyRecord(
                    source=image_path,
                    digest=digest,
                )
            )

        except Exception as error:
            failed.append((image_path, str(error)))

    if not records:
        print("No images could be hashed.", file=sys.stderr)

        for path, reason in failed:
            print(f"FAILED: {path} | {reason}", file=sys.stderr)

        return 1

    create_copy_plan(
        records=records,
        output_directory=output_directory,
    )

    if not args.quiet:
        for record in records:
            print(f"COPY: {record.source}")
            print(f"   -> {record.target}")

        for path, reason in failed:
            print(f"FAILED: {path} | {reason}", file=sys.stderr)

    if args.apply:
        apply_copy_plan(
            records=records,
            output_directory=output_directory,
        )

        operation = "Copy completed"
    else:
        operation = "Dry run completed"

    if args.manifest is not None:
        write_manifest(
            manifest_path=args.manifest.expanduser().resolve(),
            records=records,
            hash_source=args.hash_source,
        )

    print()
    print(f"{operation}.")
    print(f"Source directory:  {source_directory}")
    print(f"Output directory:  {output_directory}")
    print(f"Hash source:       {args.hash_source}")
    print(f"Images discovered: {len(images)}")
    print(f"Files planned:     {len(records)}")
    print(f"Failed to decode:  {len(failed)}")

    if not args.apply:
        print()
        print("No files were copied. Add --apply to perform the operation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())