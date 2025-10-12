#!/bin/bash
#SBATCH --job-name=spaceranger_ffpe       
#SBATCH --output=spaceranger_ffpe.out 
#SBATCH --error=spaceranger_ffpe.err  
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=32               
#SBATCH --mem=128G                       
#SBATCH --time=48:00:00                  

# Human or mouse run?
species="mouse" #mouse, human

if [[ $species -eq "mouse" ]]; then
    TRANSCRIPTOME="/path/to/refdata"  # Reference transcriptome for mouse samples
    PROBE_SET="/path/to/human_v2_probeset.csv"  # Probe set for FFPE
    job_manifest="/path/to/csv"
    OUTPUT_DIR="/path/to/output_directory"  # Directory where outputs will be saved
elif [[ $species -eq "human" ]]; then
    TRANSCRIPTOME="/path/to/refdata"  # Reference transcriptome for mouse samples
    PROBE_SET="/path/to/human_v2_probeset.csv"  # Probe set for FFPE
    job_manifest="/path/to/csv"
    OUTPUT_DIR="/path/to/output_directory"  # Directory where outputs will be saved
else
    echo "Messed up on labeling human or mouse"
    exit 1
fi


# Hard coded variables
SPACERANGER_PATH="/path/to/spaceranger"  # Path to Space Ranger installation







col_index = 0 # Job array number
items=$(
python3 - <<'PY' "$csv" "$col_idx" #single quote is important to take everything is sent in as stdin
import csv, sys
csv_path, idx = sys.argv[1], int(sys.argv[2])
with open(csv_path, newline='') as f:
    r = csv.reader(f)
    next(r, None)  # skip header; remove if no header
    out=[]
    for row in r:
        if idx < len(row) and row[idx]:
            out += [x for x in row[idx].split(';') if x]
print(",".join(out))
PY #This 
)
echo "$items"





FASTQ_DIR="/path/to/fastq_directory"    # Directory containing FASTQ files
IMAGE="/path/to/tissue_image.jpg"       # Path to H&E stained image
SLIDE="V11J26-127"                      # Slide serial number
AREA="B1"                               # Capture area on the slide


# Grabbing fastq files

fastq_files = 
#Testing_3

# Run Space Ranger count pipeline
${SPACERANGER_PATH}/spaceranger count \
    --id=spaceranger_ffpe_analysis \
    --transcriptome=${TRANSCRIPTOME} \
    --probe-set=${PROBE_SET} \
    --fastqs=${FASTQ_DIR} \
    --image=${IMAGE} \
    --slide=${SLIDE} \
    --area=${AREA} \
    --reorient-images=true \
    --localcores=32 \
    --localmem=128
