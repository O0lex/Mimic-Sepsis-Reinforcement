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

module load cuda/cuda-12.6
module load miniconda/fall-2025  # This provides the shared conda base

# 2. Use the system-wide environment (No quota cost!)
# Replace 'comp579_env' with the shared name or the path provided by your TA
# If they didn't give a name, you can often use 'base' or the path:
source /etc/profile.d/conda.sh
conda activate base 


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