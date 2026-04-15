#!/bin/bash
# =========================================================
# SLURM configuration
# To change GPU count or memory at submission time:
#   sbatch --gres=gpu:2 --mem=300G Mapping_shell.sh
# =========================================================
#SBATCH --job-name=c2l_map_v100
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH -N 1-1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:3
#SBATCH --mem=240G
#SBATCH --time=20:00:00
#SBATCH --output=./slurmOutput/v100_c2l_map_%j.log
#SBATCH --error=./slurmOutput/v100_c2l_map_%j.err

set -eo pipefail

# =========================================================
# Configurable run parameters (edit these before submitting)
# =========================================================
LEVEL=1                    # 0=lvl1, 1=lvl2, 2=lvl3
EPOCHS=200
BATCH_SIZE=16384
DETECTION_ALPHA=200
N_CELLS_PER_LOCATION=1.5
EXPORT_NUM_SAMPLES=200 #1000
EXPORT_BATCH_SIZE=1024 #16384
NUM_WORKERS=23             # Should be cpus-per-task minus 1
ACCELERATOR="gpu"
# =========================================================

PROJ="/coh_labs/yunroseli/Jona/CAR-T"
SCRIPT="/home/janzules/spatial/CAR-T/code/CellTypeAnnotation/Mapping_script.py"

mkdir -p ./slurmOutput

module load Mamba
export PYTHONNOUSERSITE="True"
mamba activate spatial_gpu_py311
module load cuda12.3/toolkit/12.3.2

nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

python -u "${SCRIPT}" \
  --proj-folder            "${PROJ}" \
  --level                  "${LEVEL}" \
  --epochs                 "${EPOCHS}" \
  --train-batch-size       "${BATCH_SIZE}" \
  --detection-alpha        "${DETECTION_ALPHA}" \
  --N-cells-per-location   "${N_CELLS_PER_LOCATION}" \
  --export-num-samples     "${EXPORT_NUM_SAMPLES}" \
  --export-batch-size      "${EXPORT_BATCH_SIZE}" \
  --num-workers            "${NUM_WORKERS}" \
  --accelerator            "${ACCELERATOR}"
