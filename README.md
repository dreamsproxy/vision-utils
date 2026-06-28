# vision-utils

A collection of practical image processing utilities for dataset preparation, image cropping, collage splitting, masking, visualization, annotation workflows, and computer vision experiments.

## Alpha 0.1 scope

The first alpha is intentionally CLI-first and dataset-focused. The goal is to make common computer vision preparation tasks repeatable, scriptable, and easy to audit.

Implemented commands:

```bash
vision-utils detect-borders INPUT_DIR OUTPUT_CSV
vision-utils crop-borders INPUT_DIR OUTPUT_DIR --csv crop_report.csv
vision-utils manifest INPUT_DIR OUTPUT_CSV
vision-utils resize INPUT_DIR OUTPUT_DIR --longest-side 256
vision-utils sample-grid INPUT_DIR sample_grid.png --count 64 --tile-size 128
```

## Install from source

```bash
git clone https://github.com/dreamsproxy/vision-utils.git
cd vision-utils
python -m pip install -e .
```

## Commands

### Detect borders

Detect probable border regions and write a CSV report.

```bash
vision-utils detect-borders ./images ./border_report.csv --workers 4
```

CSV columns:

```text
full_path,dir,filename,y0,x0,y1,x1,cropped
```

If no usable crop is detected, `y0`, `x0`, `y1`, and `x1` are set to `-1`.

### Crop borders

Crop detected borders while preserving the input directory structure.

```bash
vision-utils crop-borders ./images ./cropped --csv ./crop_report.csv --workers 4
```

By default, files without a detected crop are copied unchanged. Use `--no-copy-uncropped` to skip them.

### Manifest

Generate a checksum and image-dimension manifest.

```bash
vision-utils manifest ./images ./manifest.csv
```

CSV columns:

```text
full_path,dir,filename,sha256,size_bytes,width,height
```

### Resize dataset

Resize a dataset while preserving directory structure.

```bash
vision-utils resize ./images ./resized_256 --longest-side 256
vision-utils resize ./images ./resized_exact --width 256 --height 256
```

### Sample grid

Create a deterministic sample grid for quick visual inspection.

```bash
vision-utils sample-grid ./images ./sample_grid.png --count 64 --tile-size 128 --seed 1
```

## Design rules

- Prefer boring, inspectable utilities over clever automation.
- Preserve directory structures by default.
- Emit CSV reports for auditability.
- Keep progress bars visible even when `--quiet` is used.
- Keep image operations OpenCV-based for speed and compatibility with existing workflows.

## Near-term backlog

- Collage splitting.
- Mask preview overlays.
- Dataset comparison reports.
- Basic tests with generated toy images.
- Optional JSON sidecar summaries.
