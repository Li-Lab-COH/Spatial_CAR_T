#!/usr/bin/env python

# ============================================================
# Resume cell2location posterior export from saved model
# ============================================================
#
# This script assumes:
# 1. cell2location training completed successfully.
# 2. The trained model was saved with:
#       mod.save(f"{run_name}", overwrite=True)
# 3. The posterior export failed or timed out.
#
# This script does NOT retrain the model.
# It reloads the saved model, runs export_posterior(), saves adata_vis,
# then attempts QC only after the posterior-exported AnnData is saved.
# ============================================================

# %%
import os
from pathlib import Path
import gc
import socket
import subprocess

# Avoid accidentally using user-site packages on HPC
os.environ.setdefault("PYTHONNOUSERSITE", "True")

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use("Agg")  # required for non-interactive .py jobs on HPC
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch
import scvi
import cell2location

rcParams["pdf.fonttype"] = 42


# %%
# ============================================================
# Helpers
# ============================================================

try:
    import psutil

    def mem_report(label: str):
        proc = psutil.Process(os.getpid())
        rss_gb = proc.memory_info().rss / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
        print(
            f"[MEM] {label}: RSS={rss_gb:.2f} GB | "
            f"available={avail_gb:.2f} GB | total={total_gb:.2f} GB",
            flush=True,
        )

except ImportError:
    def mem_report(label: str):
        print(f"[MEM] {label}: psutil not installed; skipping memory report.", flush=True)


def print_header(title: str):
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


# %%
# ============================================================
# CPU / environment setup
# ============================================================

print_header("Environment")

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

# For posterior export, too many DataLoader workers can add overhead/memory pressure.
# Keep workers modest and let PyTorch use CPU threads for computation.
n_workers = min(4, max(1, n_cpus // 4))

torch.set_num_threads(n_cpus)
scvi.settings.num_threads = n_cpus
scvi.settings.dl_num_workers = n_workers

print(f"hostname: {socket.gethostname()}", flush=True)
print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)
print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK')}", flush=True)
print(f"SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE')}", flush=True)

print(f"Using {n_cpus} CPU threads", flush=True)
print(f"Using {n_workers} DataLoader workers", flush=True)

print(f"cell2location: {getattr(cell2location, '__version__', 'unknown')}", flush=True)
print(f"scvi-tools: {scvi.__version__}", flush=True)
print(f"torch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

jobid = os.environ.get("SLURM_JOB_ID")
if jobid:
    print("\nSLURM job status:", flush=True)
    print(
        subprocess.getoutput(
            f"squeue -j {jobid} -o '%.18i %.9P %.8u %.2t %.10M %.10l %.10L %.20R'"
        ),
        flush=True,
    )

mem_report("startup")


# %%
# ============================================================
# Data / parameter section
# ============================================================

print_header("Parameters")

N_cells_loc = 1
dec_alpha = 100

epoch_num = 200

# This is posterior export batch size, not training batch size.
# Higher is faster if memory allows.
batch_num = 8192

# Rescue posterior sample number.
# Lower = faster, but q05/q95 estimates are noisier.
export_num = 120 #500 for final submission use this

level = 1
levels = ["cell_type_lvl1", "cell_type_lvl2", "cell_type_lvl3"]

level_short = {
    "cell_type_lvl1": "lvl1",
    "cell_type_lvl2": "lvl2",
    "cell_type_lvl3": "lvl3",
}

current_level = levels[level]
lvl_name_short = level_short[current_level]

print(f"N_cells_loc: {N_cells_loc}", flush=True)
print(f"dec_alpha: {dec_alpha}", flush=True)
print(f"epoch_num: {epoch_num}", flush=True)
print(f"batch_num / posterior export batch size: {batch_num}", flush=True)
print(f"export_num / posterior samples: {export_num}", flush=True)
print(f"current_level: {current_level}", flush=True)
print(f"lvl_name_short: {lvl_name_short}", flush=True)


# %%
# ============================================================
# Locations
# ============================================================

print_header("Locations")

proj_folder = Path("/coh_labs/yunroseli/Jona/CAR-T/")
input_folder = proj_folder / "data/zarr/fullDataset/annotating_references"
c2l_folder = input_folder / "c2l_run_output"

spatial_loc = input_folder / "spatial_sdata_c2l_model.h5ad"
trained_sc_data_file = input_folder / f"sc_with_signatures_{lvl_name_short}.h5ad"

run_name = c2l_folder / (
    f"c2l_{lvl_name_short}_epochs_{epoch_num}_"
    f"Ncells{N_cells_loc}_decalpha_{dec_alpha}"
)

# Include export settings so rescue exports do not overwrite older/final outputs.
predicted_cells_file = input_folder / (
    f"predicted_cells_{lvl_name_short}_epochs_{epoch_num}_"
    f"Ncells{N_cells_loc}_decalpha_{dec_alpha}_"
    f"posterior{export_num}_bs{batch_num}.h5ad"
)

qc_file = predicted_cells_file.with_suffix(".qc.png")

paths = {
    "proj_folder": proj_folder,
    "input_folder": input_folder,
    "c2l_folder": c2l_folder,
    "spatial_loc": spatial_loc,
    "trained_sc_data_file": trained_sc_data_file,
    "run_name_saved_model": run_name,
    "model_pt": run_name / "model.pt",
    "predicted_cells_file_output": predicted_cells_file,
    "qc_file_output": qc_file,
}

for name, path in paths.items():
    exists = Path(path).exists()
    print(f"{'✓' if exists else '✗'} {name}: {path}", flush=True)

if not spatial_loc.exists():
    raise FileNotFoundError(f"Could not find spatial AnnData: {spatial_loc}")

if not trained_sc_data_file.exists():
    raise FileNotFoundError(f"Could not find reference AnnData: {trained_sc_data_file}")

if not run_name.exists():
    raise FileNotFoundError(f"Could not find saved model directory: {run_name}")

if not (run_name / "model.pt").exists():
    raise FileNotFoundError(f"Could not find model.pt in: {run_name}")

print(
    f"model.pt size: {(run_name / 'model.pt').stat().st_size / 1e9:.2f} GB",
    flush=True,
)


# %%
# ============================================================
# Load data
# ============================================================

print_header("Load AnnData objects")

mem_report("before loading AnnData")

print(f"Loading spatial AnnData from:\n{spatial_loc}", flush=True)
adata_vis = sc.read_h5ad(spatial_loc)
print(f"adata_vis shape before filtering: {adata_vis.shape}", flush=True)
mem_report("after loading adata_vis")

print(f"Loading reference AnnData from:\n{trained_sc_data_file}", flush=True)
adata_ref = sc.read_h5ad(trained_sc_data_file)
print(f"adata_ref shape: {adata_ref.shape}", flush=True)
mem_report("after loading adata_ref")


# %%
# ============================================================
# Rebuild reference signature matrix
# ============================================================

print_header("Build reference signature matrix")

factor_names = list(adata_ref.uns["mod"]["factor_names"])
print(f"Number of cell2location factors / cell types: {len(factor_names)}", flush=True)

if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][
        [f"means_per_cluster_mu_fg_{i}" for i in factor_names]
    ].copy()
else:
    inf_aver = adata_ref.var[
        [f"means_per_cluster_mu_fg_{i}" for i in factor_names]
    ].copy()

inf_aver.columns = factor_names

print(f"inf_aver shape before gene intersection: {inf_aver.shape}", flush=True)
print("inf_aver preview:", flush=True)
print(inf_aver.iloc[:5, :5], flush=True)

mem_report("after building inf_aver")


# %%
# ============================================================
# Match genes
# ============================================================

print_header("Gene intersection")

intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)

print(f"Number of intersecting genes: {len(intersect)}", flush=True)

adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

gc.collect()
mem_report("after gene subsetting")

print(f"The amount of genes in visium data: {adata_vis.shape[1]}", flush=True)
print(f"The amount of genes in reference signatures: {inf_aver.shape[0]}", flush=True)

# Important:
# Do not manually call setup_anndata here for the load test.
# The saved model should carry its registry, and load() validates adata against it.
#
# If load fails with a registry error later, then try adding:
# cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key="TMA")


# %%
# ============================================================
# Load saved cell2location model
# ============================================================

print_header("Load saved cell2location model")

mem_report("before model load")

print(f"\nReconstructing saved cell2location model manually from:\n{run_name}", flush=True)

model_file = Path(run_name) / "model.pt"

ckpt = torch.load(
    model_file,
    map_location="cpu",
    weights_only=False,
)

attr = ckpt["attr_dict"]

print("Checkpoint loaded.", flush=True)
print("Checkpoint keys:", list(ckpt.keys()), flush=True)
print("attr_dict keys:", list(attr.keys()), flush=True)
print("is_trained_:", attr.get("is_trained_", "MISSING"), flush=True)

# Use the exact saved gene order from the trained model.
saved_genes = pd.Index(ckpt["var_names"].astype(str))

missing_genes = saved_genes.difference(adata_vis.var_names)
if len(missing_genes) > 0:
    raise ValueError(
        f"{len(missing_genes)} saved model genes are missing from adata_vis. "
        f"First few missing genes: {list(missing_genes[:10])}"
    )

# Use the exact cell_state_df saved inside the model checkpoint.
cell_state_df_saved = attr["init_params_"]["non_kwargs"]["cell_state_df"].copy()

# Reorder both spatial data and reference signatures to match the saved model.
adata_vis = adata_vis[:, saved_genes].copy()
cell_state_df_saved = cell_state_df_saved.loc[saved_genes, :].copy()

print(f"adata_vis shape after saved-gene ordering: {adata_vis.shape}", flush=True)
print(f"cell_state_df_saved shape: {cell_state_df_saved.shape}", flush=True)

# Register AnnData exactly as in training.
cell2location.models.Cell2location.setup_anndata(
    adata=adata_vis,
    batch_key="TMA"
)

# Recover the exact model kwargs used during training.
model_kwargs = attr["init_params_"]["kwargs"]["model_kwargs"].copy()

print("Model kwargs from checkpoint:", model_kwargs, flush=True)

# Recreate the model architecture.
mod = cell2location.models.Cell2location(
    adata_vis,
    cell_state_df=cell_state_df_saved,
    **model_kwargs,
)

# Load the trained PyTorch module parameters and restore the Pyro ParamStore.
# In cell2location/scvi Pyro models, the posterior guide parameters are not
# ordinary torch module weights. They live in Pyro's ParamStore.
import pyro
from collections import OrderedDict

full_state = ckpt["model_state_dict"]

# Separate standard torch module state from Pyro guide / ParamStore state.
module_state = OrderedDict(
    (k, v)
    for k, v in full_state.items()
    if (k != "pyro_param_store") and (not k.startswith("_guide."))
)

print(f"Full checkpoint state keys: {len(full_state)}", flush=True)
print(f"Module-state keys to load into mod.module: {len(module_state)}", flush=True)

load_result = mod.module.load_state_dict(
    module_state,
    strict=False,
)

print("Torch module state loaded.", flush=True)
print("Missing keys:", load_result.missing_keys, flush=True)
print("Unexpected keys:", load_result.unexpected_keys, flush=True)

if len(load_result.missing_keys) > 0:
    raise RuntimeError(
        "Manual model restore had missing module keys. "
        "Do not proceed to export until this is resolved. "
        f"Missing keys: {load_result.missing_keys}"
    )

if len(load_result.unexpected_keys) > 0:
    raise RuntimeError(
        "Manual model restore had unexpected module keys after filtering. "
        "Do not proceed to export until this is resolved. "
        f"Unexpected keys: {load_result.unexpected_keys}"
    )

# Restore Pyro's global parameter store.
if "pyro_param_store" not in full_state:
    raise RuntimeError(
        "Checkpoint does not contain 'pyro_param_store'. "
        "Cannot safely restore the variational posterior guide parameters."
    )

pyro.clear_param_store()
pyro.get_param_store().set_state(full_state["pyro_param_store"])

param_store_state = pyro.get_param_store().get_state()
param_names = list(param_store_state["params"].keys())

print("Pyro ParamStore restored.", flush=True)
print(f"Number of Pyro params: {len(param_names)}", flush=True)
print("First 10 Pyro params:", param_names[:10], flush=True)

expected_substrings = [
    "w_sf",
    "detection_y_s",
    "n_s_cells_per_location",
]

for substring in expected_substrings:
    matches = [p for p in param_names if substring in p]
    print(f"ParamStore params containing '{substring}': {len(matches)}", flush=True)
    print(matches[:5], flush=True)

# Restore useful trained-model metadata.
for key in [
    "factor_names_",
    "detection_mean_",
    "cell_state_df_",
    "n_factors_",
    "history_",
    "is_trained_",
    "train_indices_",
    "test_indices_",
    "validation_indices_",
    "run_id_",
    "run_name_",
]:
    if key in attr:
        setattr(mod, key, attr[key])

mod.is_trained_ = True
mod.module.eval()

print("Manual model reconstruction successful.", flush=True)
print("mod.is_trained_:", getattr(mod, "is_trained_", "MISSING"), flush=True)

try:
    mod.view_anndata_setup()
except Exception as e:
    print("WARNING: mod.view_anndata_setup() failed after manual restore.", flush=True)
    print(repr(e), flush=True)
mem_report("after model load")


# %%
# ============================================================
# Export posterior
# ============================================================

print_header("Export posterior")

print(f"export_num / num_samples: {export_num}", flush=True)
print(f"batch_num / batch_size: {batch_num}", flush=True)
print("use_gpu: False", flush=True)

mem_report("before export_posterior")

adata_vis = mod.export_posterior(
    adata_vis,
    add_to_obsm=["means", "q05", "q95"],
    sample_kwargs={
        "num_samples": export_num,
        "batch_size": batch_num
    },
)

print("Posterior export finished.", flush=True)

gc.collect()
mem_report("after export_posterior")

posterior_keys = [k for k in adata_vis.obsm.keys() if "cell_abundance" in k]
print("Posterior abundance keys in adata_vis.obsm:", flush=True)
for key in posterior_keys:
    try:
        print(f"  - {key}: {adata_vis.obsm[key].shape}", flush=True)
    except Exception:
        print(f"  - {key}", flush=True)


# %%
# ============================================================
# Save immediately after posterior export
# ============================================================

print_header("Save posterior-exported AnnData")

adata_vis.write(predicted_cells_file)

print(f"Saved posterior-exported AnnData to:\n{predicted_cells_file}", flush=True)
mem_report("after saving AnnData")


# %%
# ============================================================
# QC plot after saving
# ============================================================

print_header("QC plot")

try:
    mod.plot_QC()
    plt.tight_layout()
    plt.savefig(qc_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved QC plot to:\n{qc_file}", flush=True)

except Exception as e:
    print("WARNING: mod.plot_QC() failed, but posterior AnnData was already saved.", flush=True)
    print(repr(e), flush=True)


# %%
# ============================================================
# Done
# ============================================================

print_header("Done")

print(f"Final AnnData: {predicted_cells_file}", flush=True)
print(f"QC plot: {qc_file}", flush=True)
mem_report("final")
