#!/bin/bash
#SBATCH --job-name=cuda_check    # Job name
#SBATCH --output=./slurmOutput/cuda_check.log
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=janzules@coh.org     # Where to send mail  
#SBATCH -n 1                        # Number of cores
#SBATCH -p gpu                        # gpu queue
#SBATCH --gres=gpu:1                  # Number of GPU Units
#SBATCH --mem=4G                     # Amount of memory in GB
#SBATCH --time=00:02:00               # Time limit hrs:min:sec


module avail cuda
echo "----"
nvidia-smi
echo "----"
which nvcc || echo "nvcc not in PATH"
nvcc --version || true
