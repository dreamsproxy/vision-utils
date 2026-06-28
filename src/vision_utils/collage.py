from __future__ import annotations

from pathlib import Path

from .common import ensure_dir, read_image, write_image


def split_grid_collage(
    input_path: Path,
    output_dir: Path,
    *,
    rows: int,
    cols: int,
    prefix: str | None = None,
) -> list[Path]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")

    image = read_image(input_path)
    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    if cell_h <= 0 or cell_w <= 0:
        raise ValueError("rows/cols produce empty cells for this image.")

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    stem = prefix or input_path.stem
    paths: list[Path] = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = h if r == rows - 1 else (r + 1) * cell_h
            x1 = w if c == cols - 1 else (c + 1) * cell_w
            tile = image[y0:y1, x0:x1]
            out_path = output_dir / f"{stem}_r{r:02d}_c{c:02d}{input_path.suffix.lower()}"
            write_image(out_path, tile)
            paths.append(out_path)
    return paths
