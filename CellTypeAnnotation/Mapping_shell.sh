#!/bin/bash
#SBATCH --job-name=c2l_map
#SBATCH -n 8
#SBATCH -N 1-1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=./slurmOutput/c2l_map_%j.log
#SBATCH --error=./slurmOutput/c2l_map_%j.err

set -euo pipefail

module load Mamba
export PYTHONNOUSERSITE="True"
mamba activate spatial_gpu_py311
module load cuda12.3/toolkit/12.3.2

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

# Helps avoid CUDA allocator fragmentation issues (can matter during export_posterior)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

PROJ="/home/janzules/spatial/CAR-T/data/cell2location"
SCRIPT="${PROJ}/cell2location_spatial_map.py"

python -u "${SCRIPT}" \
  --proj-folder "${PROJ}" \
  --epochs 450 \
  --train-batch-size 16384 \
  --export-num-samples 300 \
  --export-batch-size 4096 \
  --export-q50
