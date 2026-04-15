#!/bin/bash
#SBATCH --job-name=cql_sepsis_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         
#SBATCH --mem=32G                 
#SBATCH --time=06:00:00           
#SBATCH --output=logs/%j-cql.out
#SBATCH --error=logs/%j-cql.err
#SBATCH --account=YOUR_ACTUAL_COMP579_ACCOUNT  # <--- FIX THIS

set -euo pipefail

# 1. Load Environment
module load cuda/cuda-12.6
module load miniconda/miniconda-fall2024

# 2. Activate your specific environment
conda activate comp579_env

# 3. Create log directory if it doesn't exist
mkdir -p logs

# 4. Run CQL on one or both feature profiles.
#    Usage examples:
#    sbatch run_cql.slurm
#    PROFILE=full sbatch run_cql.slurm
#    PROFILE=minimal ALPHA=1.0 N_STEPS=2000 sbatch run_cql.slurm
PROFILE="${PROFILE:-both}"   # minimal | full | both
ALPHA="${ALPHA:-5.0}"
N_STEPS="${N_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-cuda:0}"

run_profile () {
    local profile="$1"
    local dataset_h5="outputs/data/mimic_mdp_${profile}_dataset.h5"
    local dataset_npz="outputs/data/mimic_mdp_${profile}_raw.npz"
    local out_dir="outputs/models/${profile}/cql"

    if [[ ! -f "${dataset_h5}" && ! -f "${dataset_npz}" ]]; then
        echo "Missing dataset for profile '${profile}'."
        echo "Expected one of: ${dataset_h5} or ${dataset_npz}"
        echo "Run preprocessing/build first, for example:"
        echo "python experiments/run_all.py --use-mimic-profiles --in-parquet data/raw/mimic_sepsis_final.parquet --profile ${profile} --skip-train"
        exit 1
    fi

    python -m src.train.train_cql \
        --dataset-h5 "${dataset_h5}" \
        --dataset-npz "${dataset_npz}" \
        --out-dir "${out_dir}" \
        --alpha "${ALPHA}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}"
}

case "${PROFILE}" in
    minimal)
        run_profile minimal
        ;;
    full)
        run_profile full
        ;;
    both)
        run_profile minimal
        run_profile full
        ;;
    *)
        echo "Invalid PROFILE='${PROFILE}'. Use one of: minimal, full, both"
        exit 2
        ;;
esac

echo "CQL SLURM job complete."