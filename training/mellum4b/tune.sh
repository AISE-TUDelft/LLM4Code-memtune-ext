#!/bin/bash
#SBATCH --job-name=me4b2g24b_test
#SBATCH --partition=gpu-a100
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000MB
#SBATCH --gpus-per-task=1
#SBATCH --account=research-eemcs-st

# Deployment purposes
# This script is used to deploy run .py files on the cluster
cd /home/alialkaswan/scratch/mem-quant/training/mellum4b

# Load modules:
module load cuda/12.5 ## cuda/11.6
module load miniconda3

export HF_HOME="/scratch/alialkaswan/.cache"

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate memenv
python3 train.py
conda deactivate
