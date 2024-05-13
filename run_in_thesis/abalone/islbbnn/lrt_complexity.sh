#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=l_compA  # sensible name for the job
#SBATCH --mem=4G                 # Default memory per CPU is 3GB.
# #SBATCH --partition=gpu          # Use the gpu partition
# #SBATCH --gres=gpu:1             # Reserve one GPU
#SBATCH --partition=smallmem
#SBATCH --mail-user=eirik.hoyheim@nmbu.no # Email me when job is done.
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/users/eirihoyh/abalone/islbbnn/log/abalone_lrt_comp_%j.out
#SBATCH --error=/mnt/users/eirihoyh/abalone/islbbnn/log/abalone_lrt_comp_%j.err

# If you want to load module
module purge                # Clean all modules
# module load samtools  # Load the Samtools software
# module list                 # List loaded modules

module load Miniconda3
eval "$(conda shell.bash hook)"

# Activate environment
conda activate skip_con

## Below you can put your scripts
python lrt_complexity.py

