# COMP579 Final Project: Sepsis Offline RL

This repository contains a scripts-first offline reinforcement learning pipeline for sepsis treatment on MIMIC-derived data. The tracked code focuses on three layers:

- data extraction and MDP construction for the mock, demo FHIR, and benchmark tracks
- BC and CQL training, including the dueling Q-function variant used in sweeps
- weighted importance sampling, sweep aggregation, and report figure generation

Current state:
-------------
- The main workflow is in `src/` and is run by `experiments/run_all.py`.

How to run
----------
From the project root, with the Python environment already installed:

```powershell
python -m src.data.mock_dataset
python -m src.data.build_mdp_dataset --force
python -m src.train.train_bc --n-steps 200
python -m src.train.train_cql --alpha 1.0 --model-arch mlp --n-steps 200
python -m src.ope.wis_eval
python experiments/run_all.py
```

Where outputs go
----------------
- `outputs/` for datasets, models, and OPE tables
- `d3rlpy_logs/` for per-run training logs
- `report/generated/figures/` for the PNGs referenced by the LaTeX report


