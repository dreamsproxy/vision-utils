import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps


ACCEPTED_EXTS = {".jpeg", ".jpg", ".webp", ".png"}
CONVERT_EXTS = {".jpeg", ".jpg", ".webp"}


def make_unique_path(path: Path) -> Path:
    """
    If path exists, append _1, _2, etc.
    Example:
        image.png -> image_1.png
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def convert_to_png(src_path: Path, dst_path: Path, overwrite: bool = False) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        dst_path = make_unique_path(dst_path)

    with Image.open(src_path) as img:
        img = ImageOps.exif_transpose(img)

        # PNG supports RGB, RGBA, grayscale, etc.
        # Convert palette or uncommon modes safely.
        if img.mode in {"P", "CMYK"}:
            img = img.convert("RGBA")

        img.save(dst_path, format="PNG", optimize=True)

    return f"CONVERTED: {src_path} -> {dst_path}"


def move_png(src_path: Path, dst_path: Path, overwrite: bool = False) -> str:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        dst_path = make_unique_path(dst_path)
    elif dst_path.exists():
        dst_path.unlink()

    shutil.move(str(src_path), str(dst_path))

    return f"MOVED: {src_path} -> {dst_path}"


def process_image(src_path: Path, input_dir: Path, output_dir: Path, overwrite: bool = False) -> str:
    rel_path = src_path.relative_to(input_dir)
    ext = src_path.suffix.lower()

    if ext == ".png":
        dst_path = output_dir / rel_path
        return move_png(src_path, dst_path, overwrite)

    if ext in CONVERT_EXTS:
        dst_path = output_dir / rel_path.with_suffix(".png")
        return convert_to_png(src_path, dst_path, overwrite)

    return f"SKIPPED: {src_path}"


def collect_images(input_dir: Path, output_dir: Path) -> list[Path]:
    images = []

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue

        # Prevent accidentally processing files already inside output_dir
        try:
            path.relative_to(output_dir)
            continue
        except ValueError:
            pass

        if path.suffix.lower() in ACCEPTED_EXTS:
            images.append(path)

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Multi-threaded image to PNG converter."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input images."
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where PNG images will be written or moved."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads. Default: 8"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory."
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir, output_dir)

    print(f"Found {len(images)} image(s).")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print()

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_image,
                image_path,
                input_dir,
                output_dir,
                args.overwrite
            )
            for image_path in images
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                success_count += 1
                print(result)
            except Exception as e:
                error_count += 1
                print(f"ERROR: {e}")

    print()
    print("Done.")
    print(f"Successful: {success_count}")
    print(f"Errors:     {error_count}")


if __name__ == "__main__":
    main()