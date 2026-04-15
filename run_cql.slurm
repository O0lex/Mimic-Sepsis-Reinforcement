#!/bin/bash
#SBATCH --job-name=cql_sepsis_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

set -euo pipefail

module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

# Optional: activate your conda environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate comp579_env

# Optional diagnostics.
echo "Host: $(hostname)"
echo "CUDA devices:"
nvidia-smi || true

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch run_cql.slurm <python_script.py> [script_args...]"
    echo "Example (CQL minimal):"
    echo "  sbatch run_cql.slurm src/train/train_cql.py --dataset-h5 outputs/data/mimic_mdp_minimal_dataset.h5 --dataset-npz outputs/data/mimic_mdp_minimal_raw.npz --out-dir outputs/models/minimal/cql --alpha 1.0 --n-steps 1000 --device cuda:0"
    exit 1
fi

PY_SCRIPT="$1"
shift

python "$PY_SCRIPT" "$@"

echo "SLURM job complete."