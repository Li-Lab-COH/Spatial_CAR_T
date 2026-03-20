#!/bin/bash
#SBATCH --job-name=spaceranger_testrun      # Job name
#SBATCH --output=./slurmOutput/spaceranger_testrun.out    # Standard output log
#SBATCH --error=./slurmOutput/spaceranger_testrun.err     # Standard error log
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=32                   # Number of CPU cores per task
#SBATCH --mem=128G                           # Memory allocation
#SBATCH --time=03:00:00                     # Time limit (hh:mm:ss)

# Echo the CPU and memory allocation
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"

# Optionally, print memory information of the node
echo "Node Memory Info:"
free -h

# Run the SpaceRanger test run
/home/janzules/spatial/software/spaceranger-4.0.1/spaceranger testrun \
  --id=verify_install \
  --localcores "${SLURM_CPUS_PER_TASK:-1}" \
  --localmem "${LOCALMEM_GB:-8}" \
  --disable-ui