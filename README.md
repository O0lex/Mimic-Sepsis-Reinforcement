# COMP579 Final Project: Sepsis Offline RL

This folder contains a scripts-first implementation scaffold for:
- MDP dataset construction (mock now, MIMIC-IV later)
- Offline RL training with d3rlpy (BC + CQL)
- Off-policy evaluation with weighted importance sampling (WIS)

For a detailed file-by-file implementation guide, see [TECHNICAL_README.md](TECHNICAL_README.md).

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

This now auto-generates report artifacts (JSON, LaTeX table, and figures) under `report/generated/`.

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

## 5) Full MIMIC-IV BigQuery Sepsis-3 Pipeline

This mode uses the `mimiciv_derived.sepsis3` table as the Sepsis-3 cohort source and builds RL arrays directly from BigQuery tables.

Before running:

1. Authenticate with Google Cloud (`gcloud auth application-default login`) or set `GOOGLE_APPLICATION_CREDENTIALS`.
2. Ensure your account has BigQuery read access to MIMIC-IV datasets.

```powershell
cd finalProj
python experiments/run_all.py --use-bigquery --gcp-project-id <your-gcp-project-id> --max-stays 500
```

Notes:

- Vasopressors are normalized to NE-equivalent dose in extraction.
- WIS can use BC-model behavior probabilities (`mu = P_BC(a_logged | s)`) via model-based mode.
- End-to-end runs auto-generate paper-style metrics and figures in `report/generated/figures/`.

### Large-table preprocessing with two feature profiles

If you downloaded the merged BigQuery table to parquet (e.g., `data/raw/mimic_sepsis_final.parquet`), build both state profiles with:

```powershell
cd finalProj
python -m src.data.preprocess --in-parquet data/raw/mimic_sepsis_final.parquet --out-dir outputs/data
```

This produces:

- Minimal profile NPZ: `outputs/data/mimic_mdp_minimal_raw.npz`
- Full profile NPZ: `outputs/data/mimic_mdp_full_raw.npz`
- Profile summary: `outputs/data/mimic_mdp_profiles_summary.json`

To export d3rlpy datasets:

```powershell
python -m src.data.build_mdp_dataset --in-npz outputs/data/mimic_mdp_minimal_raw.npz --out-h5 outputs/data/mimic_mdp_minimal_dataset.h5 --force
python -m src.data.build_mdp_dataset --in-npz outputs/data/mimic_mdp_full_raw.npz --out-h5 outputs/data/mimic_mdp_full_dataset.h5 --force
```

To train on each profile:

```powershell
python -m src.train.train_bc --dataset-h5 outputs/data/mimic_mdp_minimal_dataset.h5 --dataset-npz outputs/data/mimic_mdp_minimal_raw.npz
python -m src.train.train_cql --dataset-h5 outputs/data/mimic_mdp_minimal_dataset.h5 --dataset-npz outputs/data/mimic_mdp_minimal_raw.npz --alpha 1.0

python -m src.train.train_bc --dataset-h5 outputs/data/mimic_mdp_full_dataset.h5 --dataset-npz outputs/data/mimic_mdp_full_raw.npz
python -m src.train.train_cql --dataset-h5 outputs/data/mimic_mdp_full_dataset.h5 --dataset-npz outputs/data/mimic_mdp_full_raw.npz --alpha 1.0
```

Or run everything (preprocess, build dataset, train, WIS, report assets) from one command:

```powershell
python experiments/run_all.py --use-mimic-profiles --in-parquet data/raw/mimic_sepsis_final.parquet --profile both
```

Useful variants:

```powershell
# Minimal profile only
python experiments/run_all.py --use-mimic-profiles --in-parquet data/raw/mimic_sepsis_final.parquet --profile minimal

# Full profile only
python experiments/run_all.py --use-mimic-profiles --in-parquet data/raw/mimic_sepsis_final.parquet --profile full

# Build + evaluate only (skip training)
python experiments/run_all.py --use-mimic-profiles --in-parquet data/raw/mimic_sepsis_final.parquet --profile both --skip-train
```

## 6) Files To Replace For Real MIMIC-IV Data

- `src/data/mock_dataset.py`: replace with true MIMIC-IV cohort extraction and trajectory build.
- `src/data/build_mdp_dataset.py`: keep export logic, switch input from mock arrays to real features/actions/rewards.
- `src/ope/wis_eval.py`: connect to policy action probabilities from BC/CQL inference on held-out trajectories.

## 7) Output Artifacts

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
- `outputs/data/mimic_bq_mdp_raw.npz`
- `outputs/data/mimic_bq_mdp_dataset.h5`
- `outputs/ope/mimic_bq_wis_summary.json`

Generated under `finalProj/report/generated`:
- `report/generated/report_metrics.json`
- `report/generated/results_table.tex`
- `report/generated/figures/fig2_training_objective_curves.png`
- `report/generated/figures/fig4_feature_importance_rf_proxy.png`
- `report/generated/figures/fig5_action_distribution_comparison.png`
- `report/generated/figures/fig6_return_survival_relationship.png`
- `report/generated/figures/fig7_final_survival_comparison.png`

## 8) Next Implementation Step

Run the BigQuery pipeline on your full authorized MIMIC-IV cohort and validate table-specific schema assumptions for your project's mirror.

## 9) Initialize Git And Push

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
