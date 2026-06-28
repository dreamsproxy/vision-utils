from __future__ import annotations

import argparse
from pathlib import Path

from .borders import crop_borders, detect_borders
from .collage import split_grid_collage
from .common import parse_exts
from .grids import make_sample_grid
from .manifest import build_manifest
from .resize import resize_dataset


def add_common_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exts", default=None, help="Comma-separated extensions. Default: common image formats.")
    parser.add_argument("--non-recursive", action="store_true", help="Only process the top-level input directory.")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for IO-bound jobs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress non-progress logs. Progress bars remain enabled.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-utils",
        description="Practical computer vision dataset utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect-borders", help="Detect likely borders and write a CSV report.")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_csv", type=Path)
    p.add_argument("--std-threshold", type=float, default=2.0)
    p.add_argument("--mean-threshold", type=float, default=8.0)
    add_common_image_args(p)

    p = sub.add_parser("crop-borders", help="Crop detected borders into an output directory.")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV report path.")
    p.add_argument("--std-threshold", type=float, default=2.0)
    p.add_argument("--mean-threshold", type=float, default=8.0)
    p.add_argument("--no-copy-uncropped", action="store_true", help="Do not copy images where no crop is detected.")
    add_common_image_args(p)

    p = sub.add_parser("manifest", help="Write SHA256 and image-dimension manifest CSV.")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_csv", type=Path)
    add_common_image_args(p)

    p = sub.add_parser("resize", help="Resize a dataset while preserving directory structure.")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--max-side", type=int, default=None, help="Resize longest side to this value.")
    p.add_argument("--width", type=int, default=None, help="Exact output width. Requires --height.")
    p.add_argument("--height", type=int, default=None, help="Exact output height. Requires --width.")
    p.add_argument("--interpolation", default="area", choices=["nearest", "linear", "area", "cubic", "lanczos"])
    add_common_image_args(p)

    p = sub.add_parser("sample-grid", help="Create a quick visual grid from a dataset.")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_path", type=Path)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--thumb-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    add_common_image_args(p)

    p = sub.add_parser("split-collage", help="Split a regular grid collage into individual tiles.")
    p.add_argument("input_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--prefix", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "detect-borders":
        detect_borders(
            args.input_dir,
            args.output_csv,
            exts=parse_exts(args.exts),
            recursive=not args.non_recursive,
            workers=args.workers,
            std_threshold=args.std_threshold,
            mean_threshold=args.mean_threshold,
            quiet=args.quiet,
        )
        return 0

    if args.command == "crop-borders":
        crop_borders(
            args.input_dir,
            args.output_dir,
            exts=parse_exts(args.exts),
            recursive=not args.non_recursive,
            workers=args.workers,
            std_threshold=args.std_threshold,
            mean_threshold=args.mean_threshold,
            copy_uncropped=not args.no_copy_uncropped,
            csv_path=args.csv,
            quiet=args.quiet,
        )
        return 0

    if args.command == "manifest":
        build_manifest(
            args.input_dir,
            args.output_csv,
            exts=parse_exts(args.exts),
            recursive=not args.non_recursive,
            workers=args.workers,
            quiet=args.quiet,
        )
        return 0

    if args.command == "resize":
        resize_dataset(
            args.input_dir,
            args.output_dir,
            exts=parse_exts(args.exts),
            recursive=not args.non_recursive,
            workers=args.workers,
            max_side=args.max_side,
            exact_width=args.width,
            exact_height=args.height,
            interpolation_name=args.interpolation,
            quiet=args.quiet,
        )
        return 0

    if args.command == "sample-grid":
        make_sample_grid(
            args.input_dir,
            args.output_path,
            exts=parse_exts(args.exts),
            recursive=not args.non_recursive,
            limit=args.limit,
            rows=args.rows,
            cols=args.cols,
            thumb_size=args.thumb_size,
            seed=args.seed,
            quiet=args.quiet,
        )
        return 0

    if args.command == "split-collage":
        split_grid_collage(
            args.input_path,
            args.output_dir,
            rows=args.rows,
            cols=args.cols,
            prefix=args.prefix,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
