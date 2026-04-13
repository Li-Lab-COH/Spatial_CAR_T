#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cell2location spatial mapping (standalone script).
Mirrors Spatial_Mapping.ipynb exactly.

Inputs:
  <proj>/results/cell2location/C2L_inputs/spatial_sdata_c2l_model.h5ad
  <proj>/results/cell2location/C2L_inputs/sc_with_signatures_<level>.h5ad

Outputs:
  Model:         <proj>/results/cell2location/c2l_<level>/
  Predictions:   <proj>/results/cell2location/C2L_inputs/predicted_cells_<level>.h5ad
  Figures:       <proj>/figures/cell2location/mapping/
"""

from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # batch-safe, no display required
import matplotlib.pyplot as plt

import scanpy as sc
import torch
import scvi
import cell2location


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def call_and_save_plots(func, fig_dir: Path, name_prefix: str, dpi: int = 200) -> int:
    """
    Call func() and save every figure it produces via plt.show().

    cell2location plotting functions (plot_QC, plot_spatial_QC_across_batches)
    call plt.show() internally after each figure, which clears the figure before
    any external save can happen.  This helper intercepts those calls, saves the
    current figure first, then lets the original plt.show() run.

    Returns the number of figures saved.
    """
    original_show = plt.show
    saved_count = [0]

    def intercepted_show(*args, **kwargs):
        fig = plt.gcf()
        if fig.get_axes():
            out = fig_dir / f"{name_prefix}_{saved_count[0]}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"  [{now_str()}] Saved: {out}")
            saved_count[0] += 1
        original_show(*args, **kwargs)

    plt.show = intercepted_show
    try:
        func()
    finally:
        plt.show = original_show

    return saved_count[0]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run cell2location spatial mapping.")

    # Paths
    ap.add_argument("--proj-folder", type=str,
                    default="/coh_labs/yunroseli/Jona/CAR-T",
                    help="Project root (contains results/ and figures/).")
    ap.add_argument("--level", type=int, default=1, choices=[0, 1, 2],
                    help="Annotation level index: 0=lvl1, 1=lvl2, 2=lvl3  (default: 1).")
    ap.add_argument("--spatial-file", type=str,
                    default="spatial_sdata_c2l_model.h5ad",
                    help="Spatial AnnData filename inside C2L_inputs/.")
    ap.add_argument("--batch-key", type=str, default="TMA",
                    help="obs column used as batch_key for Cell2location.setup_anndata.")

    # Model hyperparameters
    ap.add_argument("--N-cells-per-location", type=float, default=1.5)
    ap.add_argument("--detection-alpha", type=float, default=200.0)

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--train-batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--num-particles", type=int, default=1)
    ap.add_argument("--accelerator", type=str, default="gpu",
                    help="PyTorch Lightning accelerator: gpu | cpu | mps.")
    ap.add_argument("--num-workers", type=int, default=7,
                    help="DataLoader workers (match to --cpus-per-task minus 1).")

    # Export hyperparameters
    ap.add_argument("--export-num-samples", type=int, default=1000,
                    help="Posterior samples for export_posterior.")
    ap.add_argument("--export-batch-size", type=int, default=16384,
                    help="Batch size for export_posterior.")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    # --- Level / naming (mirrors notebook) ---
    levels = ["cell_type_lvl1", "cell_type_lvl2", "cell_type_lvl3"]
    level_short = {"cell_type_lvl1": "lvl1", "cell_type_lvl2": "lvl2", "cell_type_lvl3": "lvl3"}
    current_level = levels[args.level]
    lvl_name_short = level_short[current_level]

    # --- Paths (mirrors notebook) ---
    proj_folder          = Path(args.proj_folder)
    c2l_folder           = proj_folder / "results/cell2location"
    input_folder         = c2l_folder / "C2L_inputs"
    figures_folder       = proj_folder / "figures/cell2location/mapping"
    run_name             = c2l_folder / f"c2l_{lvl_name_short}"
    spatial_loc          = input_folder / args.spatial_file
    trained_sc_data_file = input_folder / f"sc_with_signatures_{lvl_name_short}.h5ad"
    predicted_cells_file = input_folder / f"predicted_cells_{lvl_name_short}.h5ad"

    for d in [figures_folder, run_name]:
        ensure_dir(d)

    # --- Metadata ---
    run_meta = {
        "timestamp":            now_str(),
        "proj_folder":          str(proj_folder),
        "level":                args.level,
        "lvl_name_short":       lvl_name_short,
        "spatial_loc":          str(spatial_loc),
        "trained_sc_data_file": str(trained_sc_data_file),
        "run_name":             str(run_name),
        "N_cells_per_location": args.N_cells_per_location,
        "detection_alpha":      args.detection_alpha,
        "epochs":               args.epochs,
        "train_batch_size":     args.train_batch_size,
        "lr":                   args.lr,
        "num_particles":        args.num_particles,
        "export_num_samples":   args.export_num_samples,
        "export_batch_size":    args.export_batch_size,
        "torch_version":        torch.__version__,
        "cuda_available":       torch.cuda.is_available(),
        "cuda_version_torch":   torch.version.cuda,
        "gpu_name":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scvi_version":         scvi.__version__,
        "cell2location_version": cell2location.__version__,
    }
    with open(run_name / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[{now_str()}] Run metadata saved to {run_name / 'run_metadata.json'}")

    # --- Load data ---
    print(f"[{now_str()}] Loading spatial:             {spatial_loc}")
    if not spatial_loc.exists():
        raise FileNotFoundError(f"Spatial file not found: {spatial_loc}")
    print(f"[{now_str()}] Loading reference signatures: {trained_sc_data_file}")
    if not trained_sc_data_file.exists():
        raise FileNotFoundError(f"Reference signatures not found: {trained_sc_data_file}")

    adata_vis = sc.read_h5ad(spatial_loc)
    adata_ref = sc.read_h5ad(trained_sc_data_file)

    # --- Reference signatures (inf_aver) ---
    if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = adata_ref.var[
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]

    # --- Gene intersection ---
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    print(f"[{now_str()}] Shared genes: {len(intersect)}")
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver  = inf_aver.loc[intersect, :].copy()
    print(f"[{now_str()}] Spatial: {adata_vis.shape}  |  Reference signatures: {inf_aver.shape}")

    # --- Setup anndata ---
    if args.batch_key not in adata_vis.obs.columns:
        raise KeyError(
            f"batch_key '{args.batch_key}' not found in adata_vis.obs. "
            f"Available columns: {list(adata_vis.obs.columns)[:25]}"
        )
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key=args.batch_key)

    # --- Build model ---
    print(f"[{now_str()}] Building Cell2location model...")
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        N_cells_per_location=args.N_cells_per_location,
        detection_alpha=args.detection_alpha,
    )
    mod.view_anndata_setup()

    # --- Train ---
    print(f"[{now_str()}] Training: epochs={args.epochs}, batch_size={args.train_batch_size}, "
          f"lr={args.lr}, accelerator={args.accelerator}")
    torch.cuda.empty_cache()
    mod.train(
        max_epochs=args.epochs,
        batch_size=args.train_batch_size,
        train_size=1,
        lr=args.lr,
        num_particles=args.num_particles,
        accelerator=args.accelerator,
        datasplitter_kwargs={
            "drop_dataset_tail": True,
            "drop_last": False,
            "num_workers": args.num_workers,
        },
    )

    # --- Training history plot ---
    print(f"[{now_str()}] Saving training history plot...")
    fig, ax = plt.subplots(figsize=(10, 4))
    mod.plot_history(5, ax=ax)
    ax.legend(labels=["full data training"])
    out = figures_folder / f"training_history_{lvl_name_short}_{args.epochs}epochs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{now_str()}] Saved: {out}")

    # --- Save model (before export_posterior — safer if export crashes) ---
    print(f"[{now_str()}] Saving model to: {run_name}")
    mod.save(str(run_name), overwrite=True)

    # --- Export posterior ---
    print(f"[{now_str()}] Exporting posterior: num_samples={args.export_num_samples}, "
          f"batch_size={args.export_batch_size}")
    adata_vis = mod.export_posterior(
        adata_vis,
        sample_kwargs={
            "num_samples": args.export_num_samples,
            "batch_size": args.export_batch_size,
        },
    )

    # --- QC plots ---
    # plot_QC and plot_spatial_QC_across_batches both call plt.show() internally,
    # which clears figures before a normal savefig can run.
    # call_and_save_plots intercepts those plt.show() calls to save first.
    print(f"[{now_str()}] Saving QC plots...")
    try:
        n = call_and_save_plots(
            mod.plot_QC,
            figures_folder,
            f"QC_{lvl_name_short}_{args.epochs}epochs",
        )
        print(f"  plot_QC saved {n} figure(s).")
    except Exception as e:
        print(f"[{now_str()}] WARNING: mod.plot_QC() failed: {e}", file=sys.stderr)

    try:
        n = call_and_save_plots(
            mod.plot_spatial_QC_across_batches,
            figures_folder,
            f"spatial_QC_{lvl_name_short}_{args.epochs}epochs",
        )
        print(f"  plot_spatial_QC_across_batches saved {n} figure(s).")
    except Exception as e:
        print(f"[{now_str()}] WARNING: mod.plot_spatial_QC_across_batches() failed: {e}", file=sys.stderr)

    # --- Save output AnnData ---
    print(f"[{now_str()}] Writing predictions: {predicted_cells_file}")
    adata_vis.write(predicted_cells_file)

    print(f"[{now_str()}] DONE.")


if __name__ == "__main__":
    main()
