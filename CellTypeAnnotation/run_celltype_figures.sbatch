#!/bin/bash
#SBATCH --job-name=celltype_figs
#SBATCH --output=celltype_figs_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
# #SBATCH --partition=<your_partition>     # uncomment + set if your cluster needs it
# #SBATCH --account=<your_account>         # uncomment + set if required
#
# No GPU needed: this is matplotlib/spatialdata rendering only.
#
# Submit with:   sbatch run_celltype_figures.sbatch
# Watch with:    tail -f celltype_figs_<jobid>.log
# Resumable:     re-submitting skips tissues whose PNG already exists.

set -euo pipefail

# --- Activate the same environment the notebook kernel uses ---
source ~/.bashrc
# Use whichever applies to your setup:
mamba activate spatial_gpu_py311 || conda activate spatial_gpu_py311

cd "$(dirname "$0")"

python run_celltype_figures.py
