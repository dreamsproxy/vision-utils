# vision-utils

Practical computer vision dataset utilities for dataset preparation, image cropping, collage splitting, resizing, manifest generation, quick visual inspection, and experiment support.

## Alpha 0.1 scope

This alpha intentionally focuses on boring, useful dataset operations:

- detect likely borders and write crop boxes to CSV
- crop borders while preserving directory structure
- generate SHA256 image manifests with dimensions
- resize image datasets by longest side or exact dimensions
- create random sample grids for visual QA
- split regular grid collages into individual tiles

The package is CLI-first and designed for local research workflows.

## Install from a local checkout

```bash
git clone https://github.com/dreamsproxy/vision-utils.git
cd vision-utils
python -m pip install -e .
```

## Commands

### Detect borders

```bash
vision-utils detect-borders /path/to/images reports/borders.csv \
  --workers 4 \
  --std-threshold 2.0 \
  --mean-threshold 8.0
```

CSV columns:

```text
full_path,dir,filename,y0,x0,y1,x1,cropped
```

If no crop box is detected, coordinates are `-1` and `cropped` is `False`.

### Crop borders

```bash
vision-utils crop-borders /path/to/images /path/to/cropped \
  --csv reports/crop_report.csv \
  --workers 4
```

By default, images with no detected crop are copied unchanged. Use `--no-copy-uncropped` to only emit cropped images.

### Build checksum manifest

```bash
vision-utils manifest /path/to/images reports/manifest.csv --workers 4
```

CSV columns:

```text
full_path,dir,filename,sha256,size_bytes,width,height
```

### Resize dataset

Resize by longest side:

```bash
vision-utils resize /path/to/images /path/to/resized --max-side 256
```

Resize to exact dimensions:

```bash
vision-utils resize /path/to/images /path/to/resized --width 256 --height 256
```

### Create a sample grid

```bash
vision-utils sample-grid /path/to/images reports/sample_grid.png \
  --limit 64 \
  --thumb-size 128 \
  --seed 1
```

### Split a regular grid collage

```bash
vision-utils split-collage sample_grid.png split_tiles --rows 8 --cols 8
```

## Shared arguments

- `--exts png,jpg,webp` limits file extensions.
- `--non-recursive` only processes direct children of the input directory.
- `--quiet` suppresses non-progress logs, but progress bars remain enabled.
- Directory-preserving operations mirror the input tree under the output directory.

## Notes

- Border detection is heuristic. It is useful for dataset cleanup, not a formal segmentation algorithm.
- The border detector combines row/column variance with intensity distance from the median boundary color.
- Image IO is OpenCV-based and uses `imdecode` / `imencode` for better path compatibility.

## Planned next steps

- add pytest smoke tests with synthetic images
- add mask preview utilities
- add safer dry-run reports
- add side-by-side before/after QA grids
- add CLI examples for AFGAN dataset preparation
