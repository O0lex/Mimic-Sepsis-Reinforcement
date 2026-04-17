#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --error=logs/%N-%j.err
#SBATCH --account=fall2024-comp551

set -eo pipefail

cd /home/2023/agrois1/comp579/Mimic-Sepsis-Reinforcement

# We rely on the PyTorch cu121 wheel now, no module load required
# module load cuda/cuda-12.6  <-- Removed
module load miniconda/miniconda-fall2024
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comp579_env

echo "Host: $(hostname)"
nvidia-smi || true

python "$@"