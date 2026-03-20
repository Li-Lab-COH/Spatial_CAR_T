#!/bin/bash
#SBATCH --job-name=humanSpaceranger       
#SBATCH --output=./slurmOutput/human/spaceranger_%a.out 
#SBATCH --error=./slurmOutput/human/spaceranger_%a.err  
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=8               
#SBATCH --mem=64G                       
#SBATCH --time=24:00:00
#SBATCH --array=0-7                  

# Human or mouse
species="human" #mouse, human

# Hard coded based on species
if [[ $species == "mouse" ]]; then
    TRANSCRIPTOME="/home/janzules/spatial/references/refdata-gex-mm10-2020-A"  
    PROBE_SET="/home/janzules/spatial/software/spaceranger-4.0.1/probe_sets/Visium_Mouse_Transcriptome_Probe_Set_v2.0_mm10-2020-A.csv"  # Probe set for FFPE
    # Local Test
    # job_manifest="/Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/second_run_images/code/job_manifest_mice.csv"
    job_manifest="/home/janzules/spatial/second_run_images/code/job_manifest_mice.csv"
    OUTPUT_suf="/home/janzules/spatial/second_run_output/mouse"  # Directory where outputs will be saved
elif [[ $species == "human" ]]; then
    TRANSCRIPTOME="/home/janzules/spatial/references/refdata-gex-GRCh38-2020-A"  # Reference transcriptome for mouse samples
    PROBE_SET="/home/janzules/spatial/software/spaceranger-4.0.1/probe_sets/Visium_Human_Transcriptome_Probe_Set_v2.0_GRCh38-2020-A.csv"  # Probe set for FFPE
    job_manifest="/home/janzules/spatial/second_run_images/code/job_manifest_human.csv"
    OUTPUT_suf="/home/janzules/spatial/second_run_output/human"
else
    echo "Messed up on labeling human or mouse"
    exit 1
fi


# Hard coded variables
SPACERANGER_PATH="/home/janzules/spatial/software/spaceranger-4.0.1/"  # Path to Space Ranger installation


# === Pull variables from manifest with Python ===
# Row index to use (0-based)
ROW_IDX=$SLURM_ARRAY_TASK_ID
# local test
# ROW_IDX=0

# Variable names must match headers (case-insensitive) in the CSV:
# headers: tgen_id, loupe_alignment, cyt_image, hne, slide, area, fastqs
# The Python prints lines like VAR='value', safe for eval.
set -o errexit -o pipefail

# Resolve script directory so we can call pross_manifest.py reliably if this script is invoked from elsewhere
# local test
# SCRIPT_DIR="/Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/second_run_images/code/"
SCRIPT_DIR="/home/janzules/spatial/second_run_images/code/"

##"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture and eval the assignments
assignments="$(
  python3 "${SCRIPT_DIR}/pross_manifest.py" \
    "$job_manifest" "$ROW_IDX" \
    TGEN_ID LOUPE_ALIGNMENT CYT_IMAGE HNE SLIDE AREA FASTQS
)"
# If you want to see exactly what came back, uncomment:
echo "$assignments" >&2

# Apply them to the current shell
eval "$assignments"

# Folder output creation
OUTPUT_DIR="${OUTPUT_suf}/${TGEN_ID}"
mkdir -p "$OUTPUT_DIR"

# === Testing echo region of each variable ===
echo "#====================================#"
echo "TGEN_ID=$TGEN_ID"
echo "LOUPE_ALIGNMENT=$LOUPE_ALIGNMENT"
echo "CYT_IMAGE=$CYT_IMAGE"
echo "HNE=$HNE"
echo "SLIDE=$SLIDE"
echo "AREA=$AREA"
echo "FASTQS=$FASTQS"
echo ""
echo "#====================================#"


# Run Space Ranger count pipeline
"${SPACERANGER_PATH}"/spaceranger count \
    --id="${TGEN_ID}" \
    --transcriptome="${TRANSCRIPTOME}" \
    --probe-set="${PROBE_SET}" \
    --fastqs="${FASTQS}" \
    --cytaimage="${CYT_IMAGE}" \
    --image="${HNE}" \
    --loupe-alignment="${LOUPE_ALIGNMENT}" \
    --slide="${SLIDE}" \
    --area="${AREA}" \
    --nucleus-segmentation=true \
    --localcores=8 \
    --localmem=64 \
    --output-dir="${OUTPUT_DIR}" \
    --create-bam=false

    # Set an IF statement for the spots where there are multiple folders and where lanes need to be defined
