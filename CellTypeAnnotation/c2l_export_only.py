#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cell2location export-only script.

Goal:
- Load an already-trained spatial Cell2location model (model.pt in a run folder)
- Recreate the same adata_vis used during training (same shared genes + order)
- Run export_posterior to compute:
    - means, stds, q05, q95 (sampling pass)
    - q50 (second pass, quantile-only)
- Save figures into: /home/janzules/spatial/CAR-T/data/cell2location/figures
- Save final anndata: <run_name>/sp_<epoch_num>_epochs.h5ad

This script does NOT train.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc
import torch
import cell2location
import gc

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Export posterior from an existing cell2location spatial model (no training).")

    ap.add_argument(
        "--proj-folder",
        type=str,
        default="/home/janzules/spatial/CAR-T/data/cell2location",
        help="Project folder containing cell2location_inputs/ and reference_signatures/",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default="/home/janzules/spatial/CAR-T/data/cell2location/cell2location_map",
        help="Run folder containing model.pt (output of mod.save).",
    )
    ap.add_argument(
        "--epoch-num",
        type=int,
        default=450,
        help="Used only for output filename sp_<epoch_num>_epochs.h5ad (keep consistent with training run).",
    )
    ap.add_argument(
        "--batch-key",
        type=str,
        default="TMA",
        help="Must match what was used when training the spatial model.",
    )

    ap.add_argument(
        "--spatial-clean-file",
        type=str,
        default="cell2location_inputs/spatial_sdata_c2l_model_cleaned.h5ad",
        help="Relative to --proj-folder, or an absolute path.",
    )
    ap.add_argument(
        "--ref-signatures-file",
        type=str,
        default="reference_signatures/sc_with_signatures.h5ad",
        help="Relative to --proj-folder, or an absolute path.",
    )

    # Export settings
    ap.add_argument("--export-num-samples", type=int, default=300)
    ap.add_argument("--export-batch-size", type=int, default=4096)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    proj = Path(args.proj_folder)
    run_name = Path(args.run_name)
    epoch_num = args.epoch_num

    fig_dir = proj / "figures"
    ensure_dir(fig_dir)

    # Resolve paths
    spatial_path = Path(args.spatial_clean_file)
    if not spatial_path.is_absolute():
        spatial_path = proj / spatial_path

    ref_path = Path(args.ref_signatures_file)
    if not ref_path.is_absolute():
        ref_path = proj / ref_path

    model_pt = run_name / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found at: {model_pt}")

    print(f"[{now()}] Using run folder: {run_name}")
    print(f"[{now()}] Loading spatial cleaned adata: {spatial_path}")
    print(f"[{now()}] Loading reference signatures adata: {ref_path}")

    adata_vis = sc.read_h5ad(spatial_path)
    adata_ref = sc.read_h5ad(ref_path)

    # -------------------------
    # Rebuild inf_aver exactly like your mapping script
    # -------------------------
    if "mod" not in adata_ref.uns or "factor_names" not in adata_ref.uns["mod"]:
        raise KeyError("adata_ref.uns['mod']['factor_names'] missing. Did you load sc_with_signatures.h5ad?")

    factor_names = list(adata_ref.uns["mod"]["factor_names"])

    if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
        inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
            [f"means_per_cluster_mu_fg_{i}" for i in factor_names]
        ].copy()
    else:
        inf_aver = adata_ref.var[
            [f"means_per_cluster_mu_fg_{i}" for i in factor_names]
        ].copy()

    inf_aver.columns = factor_names
    inf_aver.iloc[:5, :5].to_csv(run_name / "inf_aver_preview_export_only.csv")

    # -------------------------
    # Match the gene intersection + order used previously
    # IMPORTANT: np.intersect1d returns sorted order (deterministic)
    # -------------------------
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    print(f"[{now()}] Shared genes: {len(intersect)}")

    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # Confirm batch key exists (must match training)
    if args.batch_key not in adata_vis.obs.columns:
        raise KeyError(
            f"batch_key '{args.batch_key}' not in adata_vis.obs. "
            f"Available: {list(adata_vis.obs.columns)[:30]} ..."
        )

    # Clearing residual GPU allocations
    torch.cuda.empty_cache()
    gc.collect()
    # -------------------------
    # Load trained model (no training)
    # -------------------------
    print(f"[{now()}] Loading trained Cell2location model from: {run_name}")
    # Important: do NOT call setup_anndata here; load validates against saved setup
    mod = cell2location.models.Cell2location.load(
    str(run_name),
    adata=adata_vis,
    accelerator="cpu",
    device=1,
    )
    # Now move ONLY the model to GPU for posterior sampling
    try:
        mod.to_device("cuda")
    except Exception:
        # fallback for older scvi/cell2location versions
        mod.module.to("cuda")
    # -------------------------
    # Export posterior pass 1: means/stds/q05/q95 (sampling)
    # -------------------------
    print(
        f"[{now()}] Export pass 1: means/stds/q05/q95 "
        f"(num_samples={args.export_num_samples}, batch_size={args.export_batch_size})"
    )
    torch.cuda.empty_cache()
    gc.collect()

    adata_vis = mod.export_posterior(
        adata_vis,
        add_to_obsm=["means", "stds", "q05", "q95"],
        sample_kwargs={
            "num_samples": args.export_num_samples,
            "batch_size": args.export_batch_size,
            "use_gpu": True,
        },
    )

    # Quick ELBO plot if history exists (sometimes present depending on version)
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        mod.plot_history(ax=ax)
        ax.set_title("cell2location training history (-ELBO)")
        fig.savefig(fig_dir / f"c2l_history_loadedmodel_{epoch_num}ep.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[{now()}] NOTE: plot_history skipped: {e}", file=sys.stderr)

    # QC plots
    try:
        mod.plot_QC()
        save_fig(fig_dir / f"c2l_QC_loadedmodel_{epoch_num}ep.png")
    except Exception as e:
        print(f"[{now()}] NOTE: plot_QC skipped: {e}", file=sys.stderr)

    try:
        mod.plot_spatial_QC_across_batches()
        save_fig(fig_dir / f"c2l_spatial_QC_batches_loadedmodel_{epoch_num}ep.png")
    except Exception as e:
        print(f"[{now()}] NOTE: plot_spatial_QC_across_batches skipped: {e}", file=sys.stderr)

    # -------------------------
    # Export posterior pass 2: q50 only (quantile-only)
    # -------------------------
    print(f"[{now()}] Export pass 2: q50 only (use_quantiles=True)")
    torch.cuda.empty_cache()
    gc.collect()
    adata_vis = mod.export_posterior(
        adata_vis,
        use_quantiles=True,
        add_to_obsm=["q50"],
        sample_kwargs={
            "batch_size": args.export_batch_size,
            "use_gpu": True,
        },
    )

    # Save output AnnData into the run folder, consistent naming
    out_h5ad = run_name / f"sp_{epoch_num}_epochs_export_only.h5ad"
    print(f"[{now()}] Writing output: {out_h5ad}")
    adata_vis.write(out_h5ad)

    # Save a tiny JSON with what keys were exported (handy sanity check)
    exported = {
        "timestamp": now(),
        "out_h5ad": str(out_h5ad),
        "obsm_keys": list(adata_vis.obsm.keys()),
        "export_num_samples": args.export_num_samples,
        "export_batch_size": args.export_batch_size,
    }
    with open(run_name / "export_only_metadata.json", "w") as f:
        json.dump(exported, f, indent=2)

    print(f"[{now()}] DONE.")


if __name__ == "__main__":
    main()

