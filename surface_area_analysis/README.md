# Alphashape with Holes

A pipeline for computing captured tissue area from spatial transcriptomics data, using alphashape with internal cavity detection.

## What it does

Given cell centroids from a SpatialData object, this notebook:
1. Selects a shared **alpha** value across samples (automated candidates + visual confirmation on H&E).
2. Selects a shared **grid size** across samples (sweep + per-sample sensitivity figures).
3. Computes the captured tissue area as an outer alphashape polygon **minus** internal cavities (donut polygon), with validation checks.

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd alphashape-with-holes

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

## Inputs

Before running, edit the paths in **Section 0 → Inputs** of the notebook:

- `ZARR_PATH` — path to the SpatialData `.zarr` store (CellCharter clusters).
- `SF_CSV`    — scale factors CSV (`sample_id, scalef, mpp`).
- `OUTDIR`    — where to save output figures and CSVs.

## How to run

Open `notebooks/20260502_alphawitholes.ipynb` in Jupyter and run cells top-to-bottom. The pipeline has three sections:

- **Section 1** — Alpha selection. Run, then visually pick the best alpha from the H&E overlays.
- **Section 2** — Grid size selection. Run, then visually pick the best grid size from the sensitivity figures.
- **Section 3** — Paste chosen `ALPHA` and `GRID_SIZE_UM` into the manual parameters cell, then run to compute final areas + validations.

## Outputs

- Per-sample H&E overlay PNGs (alpha candidates, grid sweep, final polygon).
- Summary CSVs of areas per sample.
- Validation 1: inequality check (`area_grid <= area_with_holes <= area_alphashape_only`).
- Validation 2: H&E anatomical overlay.

## Notes

- Hardcoded local paths in the notebook should be replaced with your own before running.
- Large outputs and `.zarr` stores are gitignored — do not commit raw data.
