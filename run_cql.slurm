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

# 4. Run Training
python src/models/train_cql.py \
    --data_path "data/processed/mimic_mdp_final.npz" \
    --save_dir "experiments/cql_v1" \
    --epochs 100 \
    --batch_size 256 \
    --alpha 5.0