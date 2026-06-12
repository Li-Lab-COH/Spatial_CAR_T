#!/usr/bin/env python
"""Render per-tissue cell-type pages, ONE tissue per subprocess, then assemble a PDF.

Why a standalone script (vs. the notebook): the Jupyter kernel keeps the full
2M-cell `zdata` resident while rendering, so kernel + worker can together trip the
node's OOM killer. Here the long-lived driver stays tiny (just a list of tissue
names) and each tissue is rendered in a fresh, short-lived subprocess, so any
memory leaked by spatialdata-plot / dask / matplotlib is reclaimed by the OS when
that process exits.

Two modes (the driver re-executes THIS file per tissue):
  driver:  python run_celltype_figures.py
  worker:  python run_celltype_figures.py --tissue NoTx_1_1 --out /path/NoTx_1_1.png

Resumable: tissues whose PNG already exists are skipped, so a re-run continues
where a crash left off. Run it in tmux on an interactive node, or via sbatch.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG  -- edit here, or override any of these with CLI flags.
# ---------------------------------------------------------------------------
ZARR_FILE = "/coh_labs/yunroseli/Jona/CAR-T/data/zarr/fullDataset/processing_Zarr/1_C2l_annotations_400"
FIGURES_FOLDER = "/home/janzules/spatial/CAR-T/figures"
FINAL_PDF_NAME = "c2l_label_perm_all_tissues_celltypes_present.pdf"

LABEL_COL = "c2l_label_perm"
TISSUE_COL = "tissue"
TABLE_NAME = "segmentation_counts"

DPI = 200
NCOLS = 4
PANEL_SIZE = 6.0
IMAGE_SCALE = None  # None = auto-pick pyramid level to match panel size

PANEL_COLOR = "#0033CC"  # strong blue, used for every panel

# Fixed cell-type display order (lineage-grouped), excluding 'Unknown'.
CELL_TYPE_ORDER = [
    # Non-immune / structural
    "Cancer_cell", "Erythrocyte", "Endothelial", "Fibroblast",
    # Monocytes
    "Monocyte", "Classical_Mono", "Nonclassical_Mono",
    # Macrophages
    "Macrophage", "M1_like_Mac", "M2_like_Mac", "Intermediate_Mac",
    # Neutrophils
    "Neutrophil", "N1_like_Neu", "N2_like_Neu",
    # Dendritic cells
    "DC", "cDC", "pDC",
    # T cells
    "Tcell", "CD4_T", "CD8_T", "Treg",
    # NK / NKT
    "NK", "NKT",
    # B cells
    "B",
]

# Treatment escalation: control -> chemo+nonspecific -> chemo+specific ->
# +radiation(nonspecific) -> +radiation(specific).
TREATMENT_RANK = {"NoTx": 0, "CyT72": 1, "CyPSCA": 2, "RTCyT72": 3, "RTCyPSCA": 4}


def tissue_sort_key(tissue):
    """Order tissues by treatment -> tumor location -> replicate."""
    parts = str(tissue).split("_")
    treatment = parts[0]
    tumor_loc = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    replicate = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (TREATMENT_RANK.get(treatment, 99), tumor_loc, replicate)


# ---------------------------------------------------------------------------
# WORKER: render a single tissue to a PNG, then the process exits.
# ---------------------------------------------------------------------------
def render_one_tissue(tissue, out_png, dpi, ncols, panel_size, image_scale):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import spatialdata as spd
    import spatialdata_plot  # noqa: F401  registers .pl accessor

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    zdata = spd.read_zarr(ZARR_FILE)

    img_key = f"{tissue}_hires_tissue_image"
    shape_key = f"{tissue}_cell_boundaries"
    if img_key not in zdata.images:
        print(f"Skipping {tissue}: missing image key {img_key}")
        return 2
    if shape_key not in zdata.shapes:
        print(f"Skipping {tissue}: missing shape key {shape_key}")
        return 2

    obs = zdata.tables[TABLE_NAME].obs
    mask = (obs[TISSUE_COL].astype(str) == tissue).to_numpy()
    labels_here = obs.loc[mask, LABEL_COL]
    present = [ct for ct in CELL_TYPE_ORDER if (labels_here == ct).any()]
    if not present:
        print(f"Skipping {tissue}: no non-Unknown cell types found")
        return 2

    # Subset to this tissue only -> render_shapes joins ~tens of thousands of
    # rows instead of the full 2M-cell table.
    sdata_t = zdata.subset([img_key, shape_key], filter_tables=True)

    nrows = math.ceil(len(present) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_size, nrows * panel_size),
        dpi=dpi, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"{tissue}\nCell types present: {len(present)}", fontsize=22)

    for ax, cell_type in zip(axes, present):
        print(f"  {tissue}: {cell_type}")
        (
            sdata_t.pl
            .render_images(img_key, scale=image_scale, norm=Normalize(0, 255), alpha=0.85)
            .pl.render_shapes(
                shape_key, color=LABEL_COL, groups=[cell_type],
                palette=[PANEL_COLOR], table_name=TABLE_NAME,
                fill_alpha=1, outline_alpha=0.95, outline_width=0.35,
                method="matplotlib",
            )
            .pl.show(ax=ax, coordinate_systems=["downscale_to_hires"], dpi=dpi, colorbar=False)
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for leg in list(fig.legends):
            leg.remove()
        ax.set_title(cell_type, fontsize=14)
        ax.set_axis_off()

    for ax in axes[len(present):]:
        ax.set_axis_off()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png} ({len(present)} cell types)")
    return 0


# ---------------------------------------------------------------------------
# DRIVER: list tissues, spawn one worker per tissue, assemble final PDF.
# ---------------------------------------------------------------------------
def list_tissues():
    import spatialdata as spd
    zdata = spd.read_zarr(ZARR_FILE)
    ts = zdata.tables[TABLE_NAME].obs[TISSUE_COL].astype(str).unique()
    return sorted((str(t) for t in ts), key=tissue_sort_key)


def assemble_pdf(png_paths, final_pdf, dpi):
    import gc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(final_pdf) as pdf:
        for p in png_paths:
            arr = mpimg.imread(p)
            h, w = arr.shape[0], arr.shape[1]
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(arr)
            ax.set_axis_off()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            del arr
            gc.collect()


def driver(dpi, ncols, panel_size, image_scale):
    figs = Path(FIGURES_FOLDER)
    png_dir = figs / "_per_tissue"
    png_dir.mkdir(parents=True, exist_ok=True)

    tissues = list_tissues()
    n = len(tissues)
    print(f"{n} tissues to render -> {png_dir}")

    png_paths = []
    for i, tissue in enumerate(tissues):
        out_png = png_dir / f"{i:02d}_{tissue}.png"
        if out_png.exists():
            print(f"[{i + 1}/{n}] {tissue}: already rendered, skipping")
            png_paths.append(out_png)
            continue

        print(f"[{i + 1}/{n}] {tissue}: rendering in subprocess ...", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--tissue", tissue, "--out", str(out_png),
            "--dpi", str(dpi), "--ncols", str(ncols),
            "--panel-size", str(panel_size),
            "--image-scale", "None" if image_scale is None else str(image_scale),
        ]
        rc = subprocess.run(cmd).returncode
        if rc == 0 and out_png.exists():
            png_paths.append(out_png)
        elif rc == 2:
            print(f"   -> skipped ({tissue})")
        else:
            print(f"   -> ERROR rendering {tissue} (exit {rc}); continuing")

    final_pdf = figs / FINAL_PDF_NAME
    print(f"\nAssembling {len(png_paths)} pages -> {final_pdf}")
    assemble_pdf(png_paths, final_pdf, dpi)
    print(f"Done.\nFinal PDF: {final_pdf}")
    return final_pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tissue", default=None, help="worker mode: render this tissue")
    ap.add_argument("--out", default=None, help="worker mode: output PNG path")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--ncols", type=int, default=NCOLS)
    ap.add_argument("--panel-size", type=float, default=PANEL_SIZE)
    ap.add_argument("--image-scale", default="None")
    args = ap.parse_args()

    image_scale = None if args.image_scale == "None" else args.image_scale

    if args.tissue is not None:
        if not args.out:
            ap.error("--out is required in worker mode (with --tissue)")
        try:
            return render_one_tissue(
                args.tissue, args.out, args.dpi, args.ncols,
                args.panel_size, image_scale,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            return 1

    driver(args.dpi, args.ncols, args.panel_size, image_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
