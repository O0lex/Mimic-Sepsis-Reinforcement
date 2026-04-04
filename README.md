# COMP579 Final Project: Sepsis Offline RL

This folder contains a scripts-first implementation scaffold for:
- MDP dataset construction (mock now, MIMIC-IV later)
- Offline RL training with d3rlpy (BC + CQL)
- Off-policy evaluation with weighted importance sampling (WIS)

## 0) Collaboration Quick Setup

1. Create a new GitHub repository (empty, no template).
2. Clone it or connect this folder as the remote.
3. Commit only code/config/docs first. Do not commit `outputs/`, `d3rlpy_logs/`, or raw data files.
4. Use branch workflow:
	- `main` for stable state
	- `feature/data-pipeline` for extraction changes
	- `feature/training-ope` for BC/CQL/OPE changes

Suggested owner split:
- Alex: data extraction + MDP design + dataset summaries
- Yusuf: BC/CQL training + OPE + plotting
- Both: report writing + final validation

## 1) Environment Setup

From workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r finalProj/requirements.txt
```

Conda alternative:

```powershell
conda create -n comp579-final python=3.11 -y
conda activate comp579-final
pip install -r finalProj/requirements.txt
```

## 2) Quick Start (Mock Pipeline)

```powershell
cd finalProj
python -m src.data.mock_dataset
python -m src.data.build_mdp_dataset --force
python -m src.train.train_bc --n-steps 500
python -m src.train.train_cql --alpha 1.0 --n-steps 500
python -m src.ope.wis_eval
```

Or run all at once:

```powershell
cd finalProj
python experiments/run_all.py
```

## 3) Mini MIMIC-IV FHIR Demo Data Pipeline

If you have the demo FHIR folder in `finalProj/data`, run:

```powershell
cd finalProj
python -m src.data.extract_sepsis_cohort --fhir-dir data/mimic-iv-clinical-database-demo-on-fhir-2.0/mimic-fhir
python -m src.data.build_mdp_dataset --in-npz outputs/data/mimic_fhir_mdp_raw.npz --out-h5 outputs/data/mimic_fhir_mdp_dataset.h5 --force
python -m src.train.train_bc --dataset-h5 outputs/data/mimic_fhir_mdp_dataset.h5 --dataset-npz outputs/data/mimic_fhir_mdp_raw.npz --n-steps 200
python -m src.train.train_cql --dataset-h5 outputs/data/mimic_fhir_mdp_dataset.h5 --dataset-npz outputs/data/mimic_fhir_mdp_raw.npz --alpha 1.0 --n-steps 200
python -m src.ope.wis_eval --in-csv outputs/data/mimic_fhir_ope_table.csv --out-json outputs/ope/mimic_fhir_wis_summary.json
```

Or end-to-end:

```powershell
cd finalProj
python experiments/run_all.py --use-fhir
```

## 4) ICU-Sepsis Open Benchmark Pipeline

This benchmark is useful when full credentialed MIMIC-IV access is pending.

```powershell
cd finalProj
python -m src.data.extract_icu_sepsis_dataset --env-id Sepsis/ICU-Sepsis-v2
python -m src.data.build_mdp_dataset --in-npz outputs/data/icu_sepsis_mdp_raw.npz --out-h5 outputs/data/icu_sepsis_mdp_dataset.h5 --force
python -m src.train.train_bc --dataset-h5 outputs/data/icu_sepsis_mdp_dataset.h5 --dataset-npz outputs/data/icu_sepsis_mdp_raw.npz --n-steps 200
python -m src.train.train_cql --dataset-h5 outputs/data/icu_sepsis_mdp_dataset.h5 --dataset-npz outputs/data/icu_sepsis_mdp_raw.npz --alpha 1.0 --n-steps 200
python -m src.ope.wis_eval --in-csv outputs/data/icu_sepsis_ope_table.csv --out-json outputs/ope/icu_sepsis_wis_summary.json
```

Or end-to-end:

```powershell
cd finalProj
python experiments/run_all.py --use-icu-sepsis
```

## 5) Files To Replace For Real MIMIC-IV Data

- `src/data/mock_dataset.py`: replace with true MIMIC-IV cohort extraction and trajectory build.
- `src/data/build_mdp_dataset.py`: keep export logic, switch input from mock arrays to real features/actions/rewards.
- `src/ope/wis_eval.py`: connect to policy action probabilities from BC/CQL inference on held-out trajectories.

## 6) Output Artifacts

Generated under `finalProj/outputs`:
- `outputs/data/mock_mdp_raw.npz`
- `outputs/data/mock_mdp_dataset.h5`
- `outputs/models/bc/`
- `outputs/models/cql/alpha_*/`
- `outputs/ope/wis_summary.json`
- `outputs/data/mimic_fhir_mdp_raw.npz`
- `outputs/data/mimic_fhir_mdp_dataset.h5`
- `outputs/ope/mimic_fhir_wis_summary.json`
- `outputs/data/icu_sepsis_mdp_raw.npz`
- `outputs/data/icu_sepsis_mdp_dataset.h5`
- `outputs/ope/icu_sepsis_wis_summary.json`

## 7) Next Implementation Step

Implement `src/data/extract_sepsis_cohort.py` for MIMIC-IV credentialed SQL extraction once access is approved.

## 8) Initialize Git And Push

From `finalProj`:

```powershell
git init
git add .
git commit -m "Initial offline RL sepsis pipeline scaffold"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If using GitHub CLI:

```powershell
gh repo create <your-repo> --private --source . --remote origin --push
```
