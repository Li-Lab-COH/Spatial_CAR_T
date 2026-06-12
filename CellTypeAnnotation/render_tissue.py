#!/usr/bin/env python
"""Render ONE tissue's cell-type page to a PNG, then exit.

Run as a short-lived subprocess (one per tissue) so that any memory leaked by
spatialdata-plot / dask / matplotlib during rendering is reclaimed by the OS
when the process exits. The driver notebook calls this once per tissue and then
concatenates the per-tissue PNGs into a single PDF.

Exit codes:
    0  success (PNG written)
    2  skipped (missing image/shape key, or no non-Unknown cell types)
    1  error (traceback printed)
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless; no display needed in the worker

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import spatialdata as spd
import spatialdata_plot  # noqa: F401  # registers the .pl accessor


# Fixed cell-type display order (lineage-grouped), excluding 'Unknown'.
CELL_TYPE_ORDER = [
    # Non-immune / structural
    "Cancer_cell",
    "Erythrocyte",
    "Endothelial",
    "Fibroblast",
    # Monocytes
    "Monocyte",
    "Classical_Mono",
    "Nonclassical_Mono",
    # Macrophages
    "Macrophage",
    "M1_like_Mac",
    "M2_like_Mac",
    "Intermediate_Mac",
    # Neutrophils
    "Neutrophil",
    "N1_like_Neu",
    "N2_like_Neu",
    # Dendritic cells
    "DC",
    "cDC",
    "pDC",
    # T cells
    "Tcell",
    "CD4_T",
    "CD8_T",
    "Treg",
    # NK / NKT
    "NK",
    "NKT",
    # B cells
    "B",
]

# Treatment escalation order: control -> chemo+nonspecific CAR-T ->
# chemo+specific CAR-T -> +radiation(nonspecific) -> +radiation(specific).
TREATMENT_RANK = {
    "NoTx": 0,
    "CyT72": 1,
    "CyPSCA": 2,
    "RTCyT72": 3,
    "RTCyPSCA": 4,
}

# Single high-contrast color used for every panel.
PANEL_COLOR = "#0033CC"  # strong blue


def tissue_sort_key(tissue):
    """Sort tissues by treatment -> tumor location -> replicate.

    Name convention: [treatment]_[tumor_location]_[replicate]
    e.g. 'RTCyPSCA_1_2' = RT+Cy+PSCA, directly-irradiated tumor (1), replicate 2.
    Tumor location 1 = directly irradiated; 2 = contralateral (abscopal).
    """
    parts = str(tissue).split("_")
    treatment = parts[0]
    tumor_loc = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    replicate = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return (TREATMENT_RANK.get(treatment, 99), tumor_loc, replicate)


def render_tissue(
    zarr_file,
    tissue,
    out_png,
    label_col="c2l_label_perm",
    tissue_col="tissue",
    table_name="segmentation_counts",
    dpi_sel=200,
    ncols=4,
    panel_size=6.0,
    image_scale=None,
):
    zdata = spd.read_zarr(zarr_file)

    img_key = f"{tissue}_hires_tissue_image"
    shape_key = f"{tissue}_cell_boundaries"

    if img_key not in zdata.images:
        print(f"Skipping {tissue}: missing image key {img_key}")
        return 2
    if shape_key not in zdata.shapes:
        print(f"Skipping {tissue}: missing shape key {shape_key}")
        return 2

    obs = zdata.tables[table_name].obs
    tissue_mask = (obs[tissue_col].astype(str) == tissue).to_numpy()
    labels_here = obs.loc[tissue_mask, label_col]

    present_cell_types = [
        ct for ct in CELL_TYPE_ORDER if (labels_here == ct).any()
    ]

    if len(present_cell_types) == 0:
        print(f"Skipping {tissue}: no non-Unknown cell types found")
        return 2

    # Subset to just this tissue's elements so render_shapes joins ~tens of
    # thousands of rows, not the whole 2M-cell table.
    sdata_t = zdata.subset([img_key, shape_key], filter_tables=True)

    n_panels = len(present_cell_types)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * panel_size, nrows * panel_size),
        dpi=dpi_sel,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(
        f"{tissue}\nCell types present: {len(present_cell_types)}",
        fontsize=22,
    )

    for ax, cell_type in zip(axes, present_cell_types):
        print(f"  {tissue}: {cell_type}")
        (
            sdata_t.pl
            .render_images(
                img_key,
                scale=image_scale,
                norm=Normalize(0, 255),
                alpha=0.85,
            )
            .pl.render_shapes(
                shape_key,
                color=label_col,
                groups=[cell_type],
                palette=[PANEL_COLOR],
                table_name=table_name,
                fill_alpha=1,
                outline_alpha=0.95,
                outline_width=0.35,
                method="matplotlib",
            )
            .pl.show(
                ax=ax,
                coordinate_systems=["downscale_to_hires"],
                dpi=dpi_sel,
                colorbar=False,
            )
        )
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for leg in list(fig.legends):
            leg.remove()
        ax.set_title(cell_type, fontsize=14)
        ax.set_axis_off()

    for ax in axes[len(present_cell_types):]:
        ax.set_axis_off()

    fig.savefig(out_png, dpi=dpi_sel)
    plt.close(fig)
    print(f"Wrote {out_png} ({n_panels} cell types)")
    return 0


def main():
    p = argparse.ArgumentParser(description="Render one tissue's cell-type page.")
    p.add_argument("--zarr", required=True)
    p.add_argument("--tissue", required=True)
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--label-col", default="c2l_label_perm")
    p.add_argument("--tissue-col", default="tissue")
    p.add_argument("--table-name", default="segmentation_counts")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--panel-size", type=float, default=6.0)
    p.add_argument("--image-scale", default="None",
                   help="multiscale level name, or 'None' for auto")
    args = p.parse_args()

    image_scale = None if args.image_scale == "None" else args.image_scale

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        code = render_tissue(
            zarr_file=args.zarr,
            tissue=args.tissue,
            out_png=args.out,
            label_col=args.label_col,
            tissue_col=args.tissue_col,
            table_name=args.table_name,
            dpi_sel=args.dpi,
            ncols=args.ncols,
            panel_size=args.panel_size,
            image_scale=image_scale,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
