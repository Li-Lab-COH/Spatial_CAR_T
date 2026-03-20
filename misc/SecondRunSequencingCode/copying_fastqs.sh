#!/bin/bash
#SBATCH --job-name=copy_fastqs
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=./slurmOutput/fastq_copy/%x-%A_%a_F07839.out
#SBATCH --error=./slurmOutput/fastq_copy/%x-%A_%a_F07839.err

sample_id="F07839"
DEST_DIR="/home/janzules/spatial/raw_data/for_spaceranger/${sample_id}"

# ensure dirs exist
mkdir -p "$DEST_DIR" 

# one file per array task
F07839=(
  "/home/janzules/spatial/raw_data/Rose_Li_VisiumHD/Fastq/BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07839_22WJCYLT3_S1_L005_R1_001.fastq.gz"
  "/home/janzules/spatial/raw_data/Rose_Li_VisiumHD/Fastq/BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07839_22WJCYLT3_S1_L005_R2_001.fastq.gz"
  "/home/janzules/spatial/raw_data/20250924_LH00295_0289_B235MNTLT4/BANOSSM_SSM0015/BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07839_235MNTLT4_TGATCAAAGG_L006_R1_001.fastq.gz"
  "/home/janzules/spatial/raw_data/20250924_LH00295_0289_B235MNTLT4/BANOSSM_SSM0015/BANOSSM_SSM0015_1_PR_Whole_C1_VISHD_F07839_235MNTLT4_TGATCAAAGG_L006_R2_001.fastq.gz"
)

FILE="${F07839[$SLURM_ARRAY_TASK_ID]}"

rsync -av --partial --progress \
  "$FILE" "$DEST_DIR"/
