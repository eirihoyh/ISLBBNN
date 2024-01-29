#!/bin/bash
#SBATCH --ntasks=8               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=f_lcomplex  # sensible name for the job
#SBATCH --mem=8G                 # Default memory per CPU is 3GB.
# #SBATCH --partition=gpu          # Use the gpu partition
# #SBATCH --gres=gpu:1             # Reserve one GPU
#SBATCH --partition=smallmem
#SBATCH --mail-user=eirik.hoyheim@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/users/eirihoyh/linear_problem/log/flow_complexity_%j.out
#SBATCH --error=/mnt/users/eirihoyh/linear_problem/log/flow_complexity_%j.err

# If you want to load module
module purge                # Clean all modules
# module load samtools  # Load the Samtools software
# module list                 # List loaded modules

module load Miniconda3
eval "$(conda shell.bash hook)"

# Activate environment
conda activate skip_con

## Below you can put your scripts
python flow_compute_complexity.py
