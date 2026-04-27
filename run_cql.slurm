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

# --- ENVIRONMENT SETUP ---
module load miniconda/fall-2025  # This provides the shared conda base
module load cuda/cuda-11.8  # Matches the version in your nvidia-smi output
export D3RLPY_DEVICE="cuda:0"
export PYTHONPATH=$PYTHONPATH:$(pwd)


export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH"


# 3. Execute the master sweep with your agent's new logic
ARCH_MODE="${ARCH_MODE:-both}"
REPORT_ARCH_OVERRIDE="${REPORT_ARCH_OVERRIDE:-}"

case "$ARCH_MODE" in
    both)    CQL_ARCHS="mlp,dueling"; DEFAULT_REPORT_ARCH="mlp" ;;
    mlp)     CQL_ARCHS="mlp";         DEFAULT_REPORT_ARCH="mlp" ;;
    dueling) CQL_ARCHS="dueling";     DEFAULT_REPORT_ARCH="dueling" ;;
    *) echo "Error: Unsupported ARCH_MODE"; exit 2 ;;
esac

# Pass all arguments from sbatch to the python script
python -m experiments.run_all \
    --cql-architectures "$CQL_ARCHS" \
    --report-arch "${REPORT_ARCH_OVERRIDE:-$DEFAULT_REPORT_ARCH}" \
    "$@"