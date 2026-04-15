#!/bin/bash
# =========================================================
# sweep_submit.sh — submit a grid of cell2location mapping
# jobs, one SLURM job per parameter combination.
#
# Each job calls Mapping_script_test.py, which writes all
# outputs (model, predictions, figures) to uniquely-named
# directories so runs never overwrite each other.
#
# Usage:
#   cd /home/janzules/spatial/CAR-T/code/CellTypeAnnotation
#   chmod +x sweep_submit.sh
#   ./sweep_submit.sh            # dry-run (shows sbatch commands)
#   ./sweep_submit.sh --submit   # actually submit to SLURM
# =========================================================

set -euo pipefail

# ---- Parse flags -------------------------------------------------------
SUBMIT=false
for arg in "$@"; do
  [[ "$arg" == "--submit" ]] && SUBMIT=true
done

# ---- Fixed parameters (same for every job) -----------------------------
LEVEL=1
EPOCHS=200
BATCH_SIZE=16384
NUM_WORKERS=23
ACCELERATOR="gpu"

PROJ="/coh_labs/yunroseli/Jona/CAR-T"
SCRIPT="/home/janzules/spatial/CAR-T/code/CellTypeAnnotation/Mapping_script_test.py"

# ---- Sweep arrays — edit these to define the parameter grid ------------
DETECTION_ALPHAS=(200)
N_CELLS_PER_LOCATIONS=(1.5)
EXPORT_NUM_SAMPLES_LIST=(1000 500 200)  # descending from original; find max that fits in memory
EXPORT_BATCH_SIZES=(16384 4096 1024)   # descending from original; lower = less VRAM per batch
# ------------------------------------------------------------------------

submitted=0

for ALPHA in "${DETECTION_ALPHAS[@]}"; do
  for NCELLS in "${N_CELLS_PER_LOCATIONS[@]}"; do
    for SAMPLES in "${EXPORT_NUM_SAMPLES_LIST[@]}"; do
      for BSZ in "${EXPORT_BATCH_SIZES[@]}"; do

        JOB_TAG="a${ALPHA}_n${NCELLS}_ep${EPOCHS}_s${SAMPLES}_bs${BSZ}"
        JOB_NAME="c2l_${JOB_TAG}"

        CMD=(sbatch
          --job-name="${JOB_NAME}"
          --output="./slurmOutput/test_${JOB_TAG}_%j.log"
          --error="./slurmOutput/test_${JOB_TAG}_%j.err"
          --wrap="
            set -eo pipefail
            mkdir -p ./slurmOutput
            module load Mamba
            export PYTHONNOUSERSITE=True
            mamba activate spatial_gpu_py311
            module load cuda12.3/toolkit/12.3.2
            nvidia-smi
            python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')\"
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export MPLBACKEND=Agg
            python -u '${SCRIPT}' \
              --proj-folder            '${PROJ}' \
              --level                  '${LEVEL}' \
              --epochs                 '${EPOCHS}' \
              --train-batch-size       '${BATCH_SIZE}' \
              --detection-alpha        '${ALPHA}' \
              --N-cells-per-location   '${NCELLS}' \
              --export-num-samples     '${SAMPLES}' \
              --export-batch-size      '${BSZ}' \
              --num-workers            '${NUM_WORKERS}' \
              --accelerator            '${ACCELERATOR}'
          "
          # ---- SLURM resource requests (override on command line if needed) ----
          -n 1
          --cpus-per-task=24
          -N 1-1
          -p gpu-v100
          --gres=gpu:3
          --mem=240G
          --time=20:00:00
        )

        if $SUBMIT; then
          "${CMD[@]}"
          echo "Submitted SLURM job: ${JOB_NAME}"
          (( submitted++ )) || true
        else
          echo "[DRY RUN] Would submit: ${JOB_NAME}"
          echo "  export-num-samples=${SAMPLES}  export-batch-size=${BSZ}"
          echo "  detection-alpha=${ALPHA}  N-cells-per-location=${NCELLS}"
          echo ""
        fi

      done
    done
  done
done

if $SUBMIT; then
  echo ""
  echo "Total jobs submitted: ${submitted}"
else
  echo "--- Dry run complete. Re-run with --submit to actually submit. ---"
fi
