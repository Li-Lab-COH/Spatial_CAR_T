#!/bin/bash
#SBATCH --job-name=cuda_check    # Job name
#SBATCH --output=./slurmOutput/cuda_check.log
#SBATCH --error=./slurmOutput/cuda_check.err
# #SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
# #SBATCH --mail-user=janzules@coh.org     # Where to send mail  
#SBATCH -p gpu-a100                        # or gpu-v100
#SBATCH --gres=gpu:1                       # Number of GPU Units
#SBATCH -N 1-1 
#SBATCH --ntasks-per-node=8  
#SBATCH --mem=80G
#SBATCH --time=01:00:00

# Modules
module load Mamba
module load cuda12.3/toolkit/12.3.2   # matches pytorch-cuda 12.4 reasonably well

# Env
export PYTHONNOUSERSITE="boo"
mamba activate spatial_gpu_env

module avail cuda
echo "----"
nvidia-smi
echo "----"
which nvcc || echo "nvcc not in PATH"
nvcc --version || true

python - << 'PYCODE'
import torch
import numpy as np
import anndata as ad
import scvi
import cell2location
import cellcharter

print("=== Versions ===")
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))

print("scvi-tools:", scvi.__version__)
print("cell2location:", getattr(cell2location, "__version__", "no __version__ attr"))
print("cellcharter:", getattr(cellcharter, "__version__", "no __version__ attr"))

print("\n=== Quick functional checks ===")

# 1. Build a tiny synthetic AnnData by hand
n_cells = 256
n_genes = 500
X = np.random.poisson(1.0, (n_cells, n_genes)).astype("float32")
adata = ad.AnnData(X)
print("Synthetic AnnData n_obs, n_vars:", adata.n_obs, adata.n_vars)

# 2. Train a tiny SCVI model (just to touch the GPU)
from scvi.model import SCVI
SCVI.setup_anndata(adata)
model = SCVI(adata, n_latent=5)
model.train(max_epochs=1)
print("SCVI training ran.")

# 3. cell2location smoke: import model class and instantiate a minimal object
from cell2location.models import Cell2location
import pandas as pd

n_cell_types = 5
gene_names = adata.var_names
cell_types = [f"type_{i}" for i in range(n_cell_types)]

# cell_state_df must be a DataFrame: rows = cell types, cols = genes
cell_state_df = pd.DataFrame(
    np.abs(np.random.normal(size=(n_cell_types, n_genes))).astype("float32"),
    index=cell_types,
    columns=gene_names,
)

print("Cell2location model import OK; creating a dummy instance...")
c2l_model = Cell2location(
    adata,
    cell_state_df=cell_state_df,
    N_cells_per_location=10,
    detection_alpha=20,
)
print("Cell2location dummy model instantiated OK.")

# 4. CellCharter smoke: import Cluster and instantiate with toy latent
from cellcharter.tl import Cluster
print("CellCharter Cluster import OK; running a dummy clustering call...")

# Dummy embedding: use SCVI latent space as input to Cluster
latent = model.get_latent_representation()
cluster_labels = Cluster(n_clusters=3).fit_predict(latent)
print("CellCharter dummy clustering OK. Label distribution:", np.bincount(cluster_labels))

print("\nAll quick checks completed without errors.")
PYCODE