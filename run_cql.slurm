#!/bin/bash
#SBATCH --job-name=mimic_final_sweep
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --propagate=NONE

set -eo pipefail

cd /home/2023/agrois1/comp579/Mimic-Sepsis-Reinforcement

# 1. Use the working module names we verified earlier
module load cuda/cuda-12.6

# 2. Use the proper way to activate conda in a script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comp579_env

# 3. Execute the master sweep
python -m experiments.run_all "$@"