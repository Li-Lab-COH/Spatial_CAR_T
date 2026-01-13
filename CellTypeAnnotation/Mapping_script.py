#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cell2location spatial mapping (standalone script).

Inputs:
- Cleaned spatial anndata:  cell2location_inputs/spatial_sdata_c2l_model_cleaned.h5ad
- Reference with signatures: reference_signatures/sc_with_signatures.h5ad

Outputs:
- Model folder: <run_name>/
- Annotated spatial anndata: <run_name>/sp_<epoch_num>_epochs.h5ad
- Figures: /home/janzules/spatial/CAR-T/data/cell2location/figures/*.png

Notes:
- Uses matplotlib Agg backend (no display required).
- Saves model BEFORE export_posterior to reduce risk of losing training if export crashes.
- export_posterior can be memory-heavy; default export batch size is conservative.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")  # batch-safe
import matplotlib.pyplot as plt

import scanpy as sc
from scipy import sparse

import torch
import scvi
import cell2location


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outpath: Path, dpi: int = 200) -> None:
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run cell2location spatial mapping on cleaned spatial AnnData using trained reference signatures."
    )

    ap.add_argument(
        "--proj-folder",
        type=str,
        default="/home/janzules/spatial/CAR-T/data/cell2location",
        help="Project folder containing cell2location_inputs/ and reference_signatures/",
    )
    ap.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Override input folder (defaults to <proj-folder>/cell2location_inputs).",
    )
    ap.add_argument(
        "--spatial-clean-file",
        type=str,
        default="spatial_sdata_c2l_model_cleaned.h5ad",
        help="Filename inside input_folder for cleaned spatial AnnData.",
    )
    ap.add_argument(
        "--ref-signatures-file",
        type=str,
        default="/home/janzules/spatial/CAR-T/data/cell2location/reference_signatures/sc_with_signatures.h5ad",
        help="Path to the reference AnnData with exported signatures (sc_with_signatures.h5ad).",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output run folder. Default: <proj-folder>/cell2location_map",
    )

    # Model hyperparameters
    ap.add_argument("--N-cells-per-location", type=float, default=1.5)
    ap.add_argument("--detection-alpha", type=float, default=20.0)

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=450)
    ap.add_argument("--train-batch-size", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--num-particles", type=int, default=1)

    # Export hyperparameters (export is where you saw crashes)
    ap.add_argument(
        "--export-num-samples",
        type=int,
        default=300,
        help="Posterior samples for export_posterior. Lower is faster/safer; 200–500 is typical.",
    )
    ap.add_argument(
        "--export-batch-size",
        type=int,
        default=4096,
        help="Batch size for export_posterior over spatial cells. Smaller reduces GPU memory spikes.",
    )
    ap.add_argument(
        "--export-q50",
        action="store_true",
        help="Also export q50 using use_quantiles=True (additional pass).",
    )

    # Which column to use as batch in Cell2location.setup_anndata
    ap.add_argument("--batch-key", type=str, default="TMA", help="obs column used as batch_key for spatial mapping.")
    
    # ap.add_argument("--file_name")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Performance/precision settings
    torch.set_float32_matmul_precision("high")

    proj_folder = Path(args.proj_folder)
    input_folder = Path(args.input_folder) if args.input_folder else (proj_folder / "cell2location_inputs")
    ensure_dir(input_folder)

    fig_dir = Path("/home/janzules/spatial/CAR-T/data/cell2location/figures")
    ensure_dir(fig_dir)

    run_name = Path(args.run_name) if args.run_name else (proj_folder / "cell2location_map")
    ensure_dir(run_name)

    spatial_clean_loc = input_folder / args.spatial_clean_file
    ref_sig_loc = Path(args.ref_signatures_file)

    # Basic run metadata
    run_meta = {
        "timestamp": now_str(),
        "proj_folder": str(proj_folder),
        "input_folder": str(input_folder),
        "spatial_clean_loc": str(spatial_clean_loc),
        "ref_sig_loc": str(ref_sig_loc),
        "run_name": str(run_name),
        "N_cells_per_location": args.N_cells_per_location,
        "detection_alpha": args.detection_alpha,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "lr": args.lr,
        "num_particles": args.num_particles,
        "export_num_samples": args.export_num_samples,
        "export_batch_size": args.export_batch_size,
        "export_q50": args.export_q50,
        "batch_key": args.batch_key,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version_torch": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scvi_version": scvi.__version__,
        "cell2location_version": cell2location.__version__,
    }

    # Save metadata early
    with open(run_name / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[{now_str()}] Loading spatial: {spatial_clean_loc}")
    if not spatial_clean_loc.exists():
        raise FileNotFoundError(f"Cleaned spatial file not found: {spatial_clean_loc}")

    print(f"[{now_str()}] Loading reference signatures: {ref_sig_loc}")
    if not ref_sig_loc.exists():
        raise FileNotFoundError(f"Reference signatures file not found: {ref_sig_loc}")

    adata_vis = sc.read_h5ad(spatial_clean_loc)
    adata_ref = sc.read_h5ad(ref_sig_loc)

    # Make sure X is sparse CSR float32 (helps performance/memory)
    if sparse.issparse(adata_vis.X):
        adata_vis.X = adata_vis.X.tocsr()
        adata_vis.X.data = adata_vis.X.data.astype(np.float32, copy=False)
    else:
        adata_vis.X = np.asarray(adata_vis.X, dtype=np.float32)

    # -------------------------
    # START: requested script start point
    # -------------------------
    # export estimated expression in each cluster
    if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    else:
        inf_aver = adata_ref.var[
            [f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns["mod"]["factor_names"]]
        ].copy()
    inf_aver.columns = adata_ref.uns["mod"]["factor_names"]
    # -------------------------
    # END: requested script start point
    # -------------------------

    # Save a tiny preview table (for logs / sanity check)
    inf_preview = inf_aver.iloc[:5, :5]
    inf_preview.to_csv(run_name / "inf_aver_preview.csv")
    print(f"[{now_str()}] inf_aver shape: {inf_aver.shape}")

    # Find shared genes and subset
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    print(f"[{now_str()}] Shared genes (intersect): {len(intersect)}")

    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # Setup anndata for mapping
    if args.batch_key not in adata_vis.obs.columns:
        raise KeyError(
            f"batch_key '{args.batch_key}' not in adata_vis.obs. "
            f"Available columns include: {list(adata_vis.obs.columns)[:25]} ..."
        )

    cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key=args.batch_key)

    # Create and train model
    print(f"[{now_str()}] Building Cell2location model...")
    mod = cell2location.models.Cell2location(
        adata_vis,
        cell_state_df=inf_aver,
        N_cells_per_location=args.N_cells_per_location,
        detection_alpha=args.detection_alpha,
    )

    # Save setup summary
    setup_txt = (run_name / "anndata_setup.txt")
    with open(setup_txt, "w") as f:
        # view_anndata_setup prints to stdout; we capture minimal summary ourselves
        f.write(f"n_obs={adata_vis.n_obs}\n")
        f.write(f"n_vars={adata_vis.n_vars}\n")
        f.write(f"batch_key={args.batch_key}\n")
        f.write(f"batches={adata_vis.obs[args.batch_key].astype(str).unique().tolist()}\n")
        f.write(f"factor_names={adata_ref.uns['mod']['factor_names']}\n")

    # Train
    torch.cuda.empty_cache()
    print(f"[{now_str()}] Training: epochs={args.epochs}, train_batch_size={args.train_batch_size}")
    mod.train(
        max_epochs=args.epochs,
        batch_size=args.train_batch_size,
        train_size=1,
        lr=args.lr,
        num_particles=args.num_particles,
	datasplitter_kwargs={"num_workers": 7},
    )

    # Plot ELBO history
    print(f"[{now_str()}] Saving training history plot...")
    fig, ax = plt.subplots(figsize=(7, 4))
    mod.plot_history(iter_start=0, iter_end=-1, ax=ax)
    ax.set_title("Cell2location training history (-ELBO)")
    save_fig(fig, fig_dir / f"c2l_train_history_{args.epochs}epochs_bs{args.train_batch_size}.png")

    # IMPORTANT: save model BEFORE posterior export (export can crash)
    print(f"[{now_str()}] Saving trained model to: {run_name}")
    mod.save(str(run_name), overwrite=True)

    # Export posterior (cell abundance)
    # This is the step that was crashing for you in Jupyter; keep export batch size conservative.
    print(
        f"[{now_str()}] Exporting posterior: num_samples={args.export_num_samples}, "
        f"export_batch_size={args.export_batch_size}"
    )

    adata_vis = mod.export_posterior(
        adata_vis,
        add_to_obsm=["means", "stds", "q05", "q95"],
        sample_kwargs={
            "num_samples": args.export_num_samples,
            "batch_size": args.export_batch_size,
        },
    )

    # Optional: export q50 in a second pass using quantile-only mode
    if args.export_q50:
        print(f"[{now_str()}] Exporting q50 (quantile-only pass)...")
        adata_vis = mod.export_posterior(
            adata_vis,
            use_quantiles=True,
            add_to_obsm=["q50"],
            sample_kwargs={"batch_size": args.export_batch_size},
        )

    # QC plots
    print(f"[{now_str()}] Saving QC plots...")
    try:
        fig = mod.plot_QC()
        # plot_QC may return None or a figure depending on version; handle both
        if fig is not None:
            save_fig(fig, fig_dir / f"c2l_QC_{args.epochs}epochs.png")
        else:
            plt.gcf().savefig(fig_dir / f"c2l_QC_{args.epochs}epochs.png", dpi=200, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"[{now_str()}] WARNING: mod.plot_QC() failed: {e}", file=sys.stderr)

    try:
        fig = mod.plot_spatial_QC_across_batches()
        if fig is not None:
            save_fig(fig, fig_dir / f"c2l_spatial_QC_across_batches_{args.epochs}epochs.png")
        else:
            plt.gcf().savefig(
                fig_dir / f"c2l_spatial_QC_across_batches_{args.epochs}epochs.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
    except Exception as e:
        print(f"[{now_str()}] WARNING: mod.plot_spatial_QC_across_batches() failed: {e}", file=sys.stderr)

    # Save annotated anndata
    alpha_tag = f"{args.detection_alpha:g}" # :g = general format, removes trailing zero and decimals that are not needed
    out_h5ad = Path(str(run_name)) / f"sp_{args.epochs}_epoch_{alpha_tag}_alpha.h5ad"
    print(f"[{now_str()}] Writing output AnnData: {out_h5ad}")
    adata_vis.write(out_h5ad)

    print(f"[{now_str()}] DONE.")


if __name__ == "__main__":
    main()
