#!/bin/bash
#SBATCH --job-name=c2l_export
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -N 1-1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=./slurmOutput/c2l_export.log
#SBATCH --error=./slurmOutput/c2l_export.err

set -eo pipefail

module load Mamba
export PYTHONNOUSERSITE="True"
mamba activate spatial_gpu_py311
module load cuda12.3/toolkit/12.3.2

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

# Helps avoid CUDA allocator fragmentation issues (can matter during export_posterior)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

# ---- export-only call (no training) ----
python -u /home/janzules/spatial/CAR-T/code/CellTypeAnnotation/c2l_export_only.py \
  --proj-folder /home/janzules/spatial/CAR-T/data/cell2location \
  --run-name /home/janzules/spatial/CAR-T/data/cell2location/cell2location_map \
  --epoch-num 450 \
  --export-num-samples 300 \
  --export-batch-size 4096

