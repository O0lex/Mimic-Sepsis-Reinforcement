# Technical README: Sepsis Offline RL Project

This document describes the implemented codebase at the script level: what each file does, how data flows through the pipeline, what features and rewards are used, how models are trained, and how results are evaluated and reported.

The project currently supports three data tracks:

1. Public MIMIC-IV FHIR demo data
2. ICU-Sepsis open Gymnasium benchmark
3. Synthetic mock data for smoke tests and fallback validation

The intended final target remains credentialed MIMIC-IV data, but the current codebase is already organized so the same modeling and reporting stack can run on all three tracks.

## 1. Repository Layout

The implementation lives under `finalProj/` with the following high-level structure:

- `src/common/`: shared helpers for seeding and JSON I/O.
- `src/data/`: dataset extraction and MDP construction scripts.
- `src/train/`: BC and CQL training entry points.
- `src/ope/`: off-policy evaluation and report-asset generation.
- `experiments/`: orchestration script to run full pipelines.
- `report/`: draft report and generated reporting assets.
- `configs/`: default experiment configuration.
- `outputs/`: generated artifacts from runs.

## 2. Shared Utilities

### `src/common/seed.py`

Provides `set_seed(seed)` to seed:

- Python `random`
- NumPy
- PyTorch if available

It also sets CuDNN deterministic flags when PyTorch is present.

### `src/common/io.py`

Provides:

- `ensure_parent(path)` to create parent directories
- `write_json(path, payload)` to write pretty-printed JSON artifacts

These helpers are used by extraction, training, OPE, and reporting scripts.

## 3. Data Pipelines

### 3.1 Public MIMIC-IV FHIR Demo Pipeline

#### `src/data/extract_sepsis_cohort.py`

This is the main clinical extraction script for the public demo FHIR dataset located at:

`data/mimic-iv-clinical-database-demo-on-fhir-2.0/mimic-fhir/`

#### Input files used

- `Patient.ndjson`
- `Condition.ndjson`
- `EncounterICU.ndjson`
- `ObservationLabevents.ndjson`
- `MedicationAdministrationICU.ndjson`

#### Cohort logic

The extractor identifies sepsis-related patients and encounters using:

- textual matches on condition display strings: `sepsis`, `septic`, `septicemia`
- ICD-like prefixes: `A40`, `A41`, `R652`, `038`, `9959`

It then builds the ICU cohort from `EncounterICU` entries whose patient or encounter matches the sepsis condition set.

Because FHIR references do not always line up cleanly across resources, the script maps observations and medication events into ICU windows using:

- patient ID
- event timestamp
- ICU encounter start/end times

#### Feature construction

The current state vector uses the top `state-dim=16` most frequent numeric lab codes from `ObservationLabevents`.

Implementation details:

- observations are streamed line-by-line from NDJSON
- only numeric values from `valueQuantity.value` are retained
- the code frequency table determines the 16 retained lab codes
- each retained code becomes one state feature
- hourly updates are used to create a sequence per encounter

The resulting state is a 16-dimensional float vector.

#### Action construction

Actions are discretized from treatment intensity along two axes:

- IV fluid amount in mL
- vasopressor intensity/rate

The discretization uses `action-bins=5`, yielding a grid-based action index.

The script recognizes vasopressor-like medication names including:

- norepinephrine
- phenylephrine
- vasopressin
- epinephrine
- dopamine
- dobutamine

Fluid-like medication names are treated as volume-related when they match tokens such as `nacl`, `dextrose`, `albumin`, `ringer`, or `fluid`.

#### Reward construction

The reward is terminal-only and survival-oriented:

- `+1` if the patient is not marked deceased within a 30-day window after ICU end
- `-1` if death is recorded within that window
- `0` for intermediate time steps

This is a deliberately simple offline-RL reward proxy appropriate for the current project stage and consistent with a survival objective.

#### Splits and leakage control

The extractor performs a patient-level split:

- 70% train
- 15% validation
- 15% test

Behavior-policy probabilities for OPE are estimated from the train split only using Laplace-smoothed action frequencies.

#### Outputs

The script writes:

- `outputs/data/mimic_fhir_mdp_raw.npz`
- `outputs/data/mimic_fhir_ope_table.csv`
- `outputs/data/mimic_fhir_summary.json`

The `.npz` includes:

- `observations`
- `actions`
- `rewards`
- `terminals`
- `episode_terminals`
- `episode_ids`
- `patient_ids`
- `split`
- `behavior_probs`
- `bc_probs`
- `cql_probs`

The OPE CSV mirrors the episode-level training/evaluation table used by WIS.

#### Current demo results

- cohort ICU encounters: 30
- sepsis patients: 16
- transitions: 1320
- state dimension: 16

### 3.2 ICU-Sepsis Open Benchmark Pipeline

#### `src/data/extract_icu_sepsis_dataset.py`

This script uses the open Gymnasium environment:

- `Sepsis/ICU-Sepsis-v2`

It is intended as the no-approval benchmark track for method validation.

#### Environment interface

Observed API characteristics:

- observation space: `Discrete(716)`
- action space: `Discrete(25)`

The extractor converts each discrete state into a 716-dimensional one-hot vector for d3rlpy compatibility.

#### Behavior policy generation

The benchmark extractor constructs a synthetic logged behavior policy over environment actions using a state-conditioned softmax preference matrix.

This is used to populate:

- `mu_prob` for behavior-policy probabilities
- `bc_prob` and `cql_prob` proxy probabilities for OPE compatibility

#### Reward structure

The environment reward is used directly, with episode return computed by summing step rewards.

#### Splits

The benchmark script also uses episode-level split logic:

- train
- validation
- test

#### Outputs

The script writes:

- `outputs/data/icu_sepsis_mdp_raw.npz`
- `outputs/data/icu_sepsis_ope_table.csv`
- `outputs/data/icu_sepsis_summary.json`

#### Current benchmark results

- episodes: 50
- transitions: 359
- state dimension: 716
- actions: 25
- average episode return: 0.72

### 3.3 Synthetic Smoke-Test Pipeline

#### `src/data/mock_dataset.py`

This script generates synthetic offline trajectories with configurable episode count, horizon, state dimension, and action count.

Purpose:

- verify the pipeline without clinical data
- provide a fallback path when data access is unavailable
- exercise d3rlpy conversion, BC/CQL training, and WIS evaluation

Current defaults:

- 500 episodes
- state dimension 16
- 25 actions

Outputs:

- `outputs/data/mock_mdp_raw.npz`
- `outputs/data/mock_ope_table.csv`
- `outputs/data/mock_summary.json`

### 3.4 Dataset Conversion

#### `src/data/build_mdp_dataset.py`

This script converts the raw `.npz` arrays into a d3rlpy-compatible `MDPDataset` saved as `.h5`.

Implementation details:

- supports current d3rlpy API signatures
- handles `timeouts` vs older terminal argument differences
- writes a dataset summary JSON for reproducibility

Current behavior:

- `mock_mdp_raw.npz` -> `mock_mdp_dataset.h5`
- `mimic_fhir_mdp_raw.npz` -> `mimic_fhir_mdp_dataset.h5`
- `icu_sepsis_mdp_raw.npz` -> `icu_sepsis_mdp_dataset.h5`

## 4. Offline RL Training

### `src/train/train_bc.py`

Trains a discrete Behavior Cloning baseline using d3rlpy.

Key details:

- uses `DiscreteBCConfig`
- supports compatibility loading from either `.h5` or `.npz`
- if `.h5` loading is incompatible, falls back to raw arrays
- saves the model under `outputs/models/bc/bc_model.d3`
- writes `train_metrics.json`

### `src/train/train_cql.py`

Trains a discrete Conservative Q-Learning model using d3rlpy.

Key details:

- uses `DiscreteCQLConfig`
- supports multiple API signatures for the conservative penalty parameter:
  - `alpha`
  - `conservative_weight`
- model is saved under `outputs/models/cql/alpha_<value>/cql_model.d3`
- writes per-run `train_metrics.json`

### Training details shared by both scripts

- seed control through `src/common/seed.py`
- short smoke-test runs are supported via `--n-steps`
- evaluation intervals are configurable via `--eval-interval`
- data loading is split into a reusable helper that supports both clinical and benchmark datasets

## 5. Off-Policy Evaluation

### `src/ope/wis_eval.py`

Implements Weighted Importance Sampling (WIS) on the episode-level OPE table.

Main logic:

- groups by `episode_id`
- computes per-episode importance weights as the product of per-step policy ratios
- optionally clips per-step ratios with `--clip`
- bootstraps episodes to form 95% confidence intervals

Default evaluation behavior:

- if a `split` column exists, evaluation runs on the `test` split by default
- reports mean return and bootstrap CI for BC and CQL

Current outputs:

- `outputs/ope/mimic_fhir_wis_summary.json`
- `outputs/ope/icu_sepsis_wis_summary.json`
- `outputs/ope/wis_summary.json` for mock data

### Current demo FHIR WIS results

- behavior return: -0.5
- BC WIS mean: -0.315, CI [-1.0, 0.587]
- CQL WIS mean: -0.424, CI [-1.0, 0.976]

### Current ICU-Sepsis WIS results

- behavior return: 0.75
- BC WIS mean: 0.772, CI [0.429, 1.000]
- CQL WIS mean: 0.786, CI [0.498, 1.000]

## 6. Reporting Utilities

### `src/ope/build_report_assets.py`

This script merges summary JSON files into report-ready artifacts.

It reads:

- `outputs/data/mimic_fhir_summary.json`
- `outputs/ope/mimic_fhir_wis_summary.json`
- `outputs/data/icu_sepsis_summary.json` (if present)
- `outputs/ope/icu_sepsis_wis_summary.json` (if present)

It writes:

- `report/generated/report_metrics.json`
- `report/generated/results_table.tex`
- `report/generated/figures/fig2_training_objective_curves.png`
- `report/generated/figures/fig4_feature_importance_rf_proxy.png`
- `report/generated/figures/fig5_action_distribution_comparison.png`
- `report/generated/figures/fig6_return_survival_relationship.png`
- `report/generated/figures/fig7_final_survival_comparison.png`

Purpose:

- keep the report reproducible
- avoid manual copying of numbers into LaTeX
- maintain one source of truth for results
- generate paper-style figures directly from training/evaluation outputs

### `report/report.tex`

Draft paper source containing:

- motivation
- related work
- data and cohort construction
- methods
- implementation/reproducibility notes
- results for public FHIR demo data
- open-benchmark validation on ICU-Sepsis
- limitations and planned final experiments
- contributions section

## 7. Experiment Orchestration

### `experiments/run_all.py`

Single entry point for full pipelines.

Supported modes:

- default: synthetic mock data
- `--use-fhir`: public MIMIC-IV FHIR demo pipeline
- `--use-icu-sepsis`: ICU-Sepsis benchmark pipeline

It performs:

1. extraction
2. dataset export
3. BC training
4. CQL training with multiple alpha values
5. WIS evaluation

6. Report asset generation (JSON + LaTeX table + figures)

This makes it the best command for end-to-end smoke tests and reproducibility.

## 8. Configuration

### `configs/default.yaml`

Contains project defaults for:

- seed
- mock data settings
- training settings
- CQL alpha sweep values

## 9. Generated Outputs

Main generated folders:

- `outputs/data/`: raw `.npz`, `.csv`, dataset `.h5`, summaries
- `outputs/models/`: BC and CQL checkpoints
- `outputs/ope/`: WIS summaries
- `d3rlpy_logs/`: d3rlpy training logs
- `report/generated/`: report-ready summaries and tables

## 10. Methodological Summary

The implemented pipeline is rigorous in the following ways:

1. Patient-level or episode-level splitting prevents leakage.
2. Behavior-policy probabilities are estimated from training data only.
3. Training, evaluation, and report generation are script-driven and reproducible.
4. WIS uses bootstrap confidence intervals rather than point estimates only.
5. Multiple data tracks are supported to compare:
   - open benchmark validity
   - public demo clinical structure
   - future credentialed MIMIC-IV deployment

The remaining caveat is that the public demo FHIR and ICU-Sepsis benchmark tracks are proxies. They validate the RL pipeline and reporting discipline, but they are not substitutes for the full credentialed MIMIC-IV study.

## 11. Relationship to Course Content and Prior Papers

This project directly reflects COMP 579 material on:

- MDP formulation
- off-policy RL
- batch/offline RL
- policy evaluation
- value-based deep RL
- uncertainty and variance in off-policy estimation

The implementation is aligned with the project proposal references:

- Wu et al. (2023): sepsis treatment with deep RL and human expertise
- Kumar et al. (2020): Conservative Q-Learning
- Johnson et al. (2023): MIMIC-IV dataset description
- Levine et al. (2020): offline RL tutorial/review
- Wang et al. (2023): diffusion policy representation for offline RL

The current code specifically supports a staged narrative:

1. validate the RL method on ICU-Sepsis
2. validate clinical-data extraction on the public MIMIC demo
3. transfer the exact pipeline to credentialed MIMIC-IV when access is available

## 12. Recommended Run Commands

Mock smoke test:

```powershell
cd finalProj
python experiments/run_all.py
```

Public MIMIC FHIR demo:

```powershell
cd finalProj
python experiments/run_all.py --use-fhir
```

ICU-Sepsis benchmark:

```powershell
cd finalProj
python experiments/run_all.py --use-icu-sepsis
```

Report asset generation:

```powershell
cd finalProj
python -m src.ope.build_report_assets
```

## 13. Current Status and Known Limitations

Implemented and working:

- public MIMIC FHIR demo extraction
- ICU-Sepsis benchmark extraction
- d3rlpy dataset export
- BC training
- CQL training
- WIS evaluation
- report metric generation

Known limitations:

- final credentialed MIMIC-IV access is still pending
- proxy policy probabilities are used in the public demo and benchmark tracks
- full clinical claims should wait for the real MIMIC-IV run

## 14. File Index

### Core scripts

- `src/data/extract_sepsis_cohort.py`
- `src/data/extract_icu_sepsis_dataset.py`
- `src/data/mock_dataset.py`
- `src/data/build_mdp_dataset.py`
- `src/train/train_bc.py`
- `src/train/train_cql.py`
- `src/ope/wis_eval.py`
- `src/ope/build_report_assets.py`
- `experiments/run_all.py`

### Reporting

- `report/report.tex`
- `report/generated/report_metrics.json`
- `report/generated/results_table.tex`

### Support files

- `src/common/io.py`
- `src/common/seed.py`
- `configs/default.yaml`
- `README.md`
- `requirements.txt`
