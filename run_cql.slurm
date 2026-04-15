#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --account=fall2024-comp551

set -euo pipefail

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comp579_env

echo "Host: $(hostname)"
nvidia-smi || true

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch run_cql.slurm <python_script.py> [script_args...]"
  exit 1
fi

PY_SCRIPT="$1"
shift

python "$PY_SCRIPT" "$@"