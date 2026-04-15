from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed

FLUID_HINTS = ("saline", "ringer", "albumin", "dextrose", "fluid", "lactated")


@dataclass
class StayInfo:
    patient_id: str
    hadm_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    dod: pd.Timestamp | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Sepsis-3 ICU cohort from MIMIC-IV BigQuery tables and build RL-ready arrays."
    )
    parser.add_argument("--gcp-project-id", type=str, required=True)
    parser.add_argument("--bq-location", type=str, default="US")
    parser.add_argument("--mimic-project", type=str, default="physionet-data")
    parser.add_argument("--derived-dataset", type=str, default="mimiciv_derived")
    parser.add_argument("--icu-dataset", type=str, default="mimiciv_icu")
    parser.add_argument("--hosp-dataset", type=str, default="mimiciv_hosp")
    parser.add_argument("--max-stays", type=int, default=500)
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/data/mimic_bq_mdp_raw.npz"))
    parser.add_argument("--out-ope-csv", type=Path, default=Path("outputs/data/mimic_bq_ope_table.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/data/mimic_bq_summary.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-bins", type=int, default=5)
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def _bq_client(project_id: str, location: str):
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "BigQuery support requires google-cloud-bigquery. Install with: pip install google-cloud-bigquery db-dtypes"
        ) from exc

    return bigquery.Client(project=project_id, location=location), bigquery


def _get_table_columns(client, table_id: str) -> set[str]:
    table = client.get_table(table_id)
    return {field.name.lower() for field in table.schema}


def _pick_column(columns: set[str], candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c.lower() in columns:
            return c
    if required:
        raise RuntimeError(f"Unable to find required column. Candidates: {candidates}")
    return None


def _to_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _rate_to_ne_equivalent(
    label: str,
    rate: float | None,
    rate_uom: str,
    amount: float | None,
    amount_uom: str,
    patient_weight: float | None,
) -> float:
    label_l = (label or "").lower()
    rate_uom_l = (rate_uom or "").lower()
    amount_uom_l = (amount_uom or "").lower()

    v_rate = float(rate) if rate is not None and not pd.isna(rate) else None
    v_amount = float(amount) if amount is not None and not pd.isna(amount) else None
    weight = float(patient_weight) if patient_weight is not None and not pd.isna(patient_weight) else None

    ne_mcg_min = 0.0
    epi_mcg_min = 0.0
    vaso_units_min = 0.0
    dopamine_mcg_kg_min = 0.0

    if "norepinephrine" in label_l or "noradrenaline" in label_l:
        if v_rate is not None and "mcg/min" in rate_uom_l:
            ne_mcg_min = max(0.0, v_rate)
        elif v_rate is not None and "mcg/kg/min" in rate_uom_l and weight and weight > 0:
            ne_mcg_min = max(0.0, v_rate * weight)
        elif v_amount is not None and "mcg" in amount_uom_l:
            ne_mcg_min = max(0.0, v_amount)

    if "epinephrine" in label_l and "norepinephrine" not in label_l:
        if v_rate is not None and "mcg/min" in rate_uom_l:
            epi_mcg_min = max(0.0, v_rate)
        elif v_rate is not None and "mcg/kg/min" in rate_uom_l and weight and weight > 0:
            epi_mcg_min = max(0.0, v_rate * weight)
        elif v_amount is not None and "mcg" in amount_uom_l:
            epi_mcg_min = max(0.0, v_amount)

    if "vasopressin" in label_l:
        if v_rate is not None and "unit/min" in rate_uom_l:
            vaso_units_min = max(0.0, v_rate)
        elif v_amount is not None and "unit" in amount_uom_l:
            vaso_units_min = max(0.0, v_amount)

    if "dopamine" in label_l:
        if v_rate is not None and "mcg/kg/min" in rate_uom_l:
            dopamine_mcg_kg_min = max(0.0, v_rate)
        elif v_rate is not None and "mcg/min" in rate_uom_l and weight and weight > 0:
            dopamine_mcg_kg_min = max(0.0, v_rate / weight)

    # Standardized NE-equivalent conversion requested for this project setup.
    ne_equivalent = ne_mcg_min + epi_mcg_min + (vaso_units_min * 2.5) + (dopamine_mcg_kg_min / 2.0)
    return float(max(0.0, ne_equivalent))


def _fluid_ml_from_event(label: str, rate: float | None, rate_uom: str, amount: float | None, amount_uom: str) -> float:
    label_l = (label or "").lower()
    rate_uom_l = (rate_uom or "").lower()
    amount_uom_l = (amount_uom or "").lower()

    if not any(tok in label_l for tok in FLUID_HINTS):
        return 0.0

    v_rate = float(rate) if rate is not None and not pd.isna(rate) else None
    v_amount = float(amount) if amount is not None and not pd.isna(amount) else None

    if v_amount is not None and ("ml" in amount_uom_l or "millil" in amount_uom_l):
        return max(0.0, v_amount)
    if v_rate is not None and ("ml/hour" in rate_uom_l or "ml/hr" in rate_uom_l):
        return max(0.0, v_rate)
    if v_rate is not None and "ml/min" in rate_uom_l:
        return max(0.0, v_rate * 60.0)
    return 0.0


def _discretize_action(fluid: float, vaso_ne: float, fluid_bins: np.ndarray, vaso_bins: np.ndarray) -> int:
    fi = int(np.digitize(fluid, fluid_bins, right=False))
    vi = int(np.digitize(vaso_ne, vaso_bins, right=False))
    fi = min(max(fi, 0), len(fluid_bins))
    vi = min(max(vi, 0), len(vaso_bins))
    return fi * (len(vaso_bins) + 1) + vi


def _build_arrays(
    cohort: dict[str, StayInfo],
    obs_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    top_codes: list[str],
    action_bins: int,
    max_hours: int,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, np.ndarray]:
    set_seed(seed)

    code_to_idx = {c: i for i, c in enumerate(top_codes)}
    obs_by_stay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obs_rows:
        obs_by_stay[row["stay_id"]].append(row)

    act_by_stay: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_fluids: list[float] = []
    all_vaso: list[float] = []
    for row in action_rows:
        act_by_stay[row["stay_id"]].append(row)
        all_fluids.append(float(row["fluid_ml"]))
        all_vaso.append(float(row["vasopressor_ne"]))

    if not all_fluids:
        all_fluids = [0.0]
    if not all_vaso:
        all_vaso = [0.0]

    q = np.linspace(0.2, 0.8, max(1, action_bins - 1))
    fluid_bins = np.unique(np.quantile(np.asarray(all_fluids, dtype=np.float32), q))
    vaso_bins = np.unique(np.quantile(np.asarray(all_vaso, dtype=np.float32), q))

    observations = []
    actions = []
    rewards = []
    terminals = []
    episode_terminals = []
    episode_ids = []
    patient_ids = []

    for ep_idx, (stay_id, stay) in enumerate(cohort.items()):
        total_hours = max(1, int(np.ceil((stay.end - stay.start).total_seconds() / 3600.0)))
        horizon = min(max_hours, total_hours)

        feature_state = np.zeros((len(top_codes),), dtype=np.float32)
        stay_obs = sorted(obs_by_stay.get(stay_id, []), key=lambda x: x["time"])
        stay_actions = sorted(act_by_stay.get(stay_id, []), key=lambda x: x["time"])
        obs_ptr = 0

        for t in range(horizon):
            t0 = stay.start + pd.Timedelta(hours=t)
            t1 = t0 + pd.Timedelta(hours=1)

            while obs_ptr < len(stay_obs) and stay_obs[obs_ptr]["time"] < t1:
                row = stay_obs[obs_ptr]
                idx = code_to_idx.get(row["code"])
                if idx is not None:
                    feature_state[idx] = row["value"]
                obs_ptr += 1

            fluid = 0.0
            vaso = 0.0
            for a in stay_actions:
                if t0 <= a["time"] < t1:
                    fluid += float(a["fluid_ml"])
                    vaso = max(vaso, float(a["vasopressor_ne"]))

            action_idx = _discretize_action(fluid, vaso, fluid_bins, vaso_bins)
            done = 1.0 if t == horizon - 1 else 0.0

            reward = 0.0
            if done > 0:
                if stay.dod is not None and stay.dod <= (stay.end + pd.Timedelta(days=30)):
                    reward = -1.0
                else:
                    reward = 1.0

            observations.append(feature_state.copy())
            actions.append(action_idx)
            rewards.append(reward)
            terminals.append(done)
            episode_terminals.append(done)
            episode_ids.append(ep_idx)
            patient_ids.append(stay.patient_id)

    observations_np = np.asarray(observations, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.int64)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    terminals_np = np.asarray(terminals, dtype=np.float32)
    episode_terminals_np = np.asarray(episode_terminals, dtype=np.float32)
    episode_ids_np = np.asarray(episode_ids, dtype=np.int64)
    patient_ids_np = np.asarray(patient_ids)

    unique_patients = np.unique(patient_ids_np)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_patients)

    n_train = int(np.floor(train_frac * shuffled.shape[0]))
    n_val = int(np.floor(val_frac * shuffled.shape[0]))
    n_train = max(1, min(n_train, shuffled.shape[0]))
    n_val = max(0, min(n_val, shuffled.shape[0] - n_train))

    train_patients = set(shuffled[:n_train].tolist())
    val_patients = set(shuffled[n_train : n_train + n_val].tolist())

    split_np = np.empty(patient_ids_np.shape[0], dtype="<U8")
    for i, pid in enumerate(patient_ids_np):
        if pid in train_patients:
            split_np[i] = "train"
        elif pid in val_patients:
            split_np[i] = "val"
        else:
            split_np[i] = "test"

    n_actions = int(actions_np.max() + 1) if actions_np.size else action_bins * action_bins
    train_mask = split_np == "train"
    train_actions = actions_np[train_mask]
    laplace = 1.0
    counts = np.bincount(train_actions, minlength=n_actions).astype(np.float64) + laplace
    probs = counts / np.clip(counts.sum(), 1.0, None)
    mu = probs[np.clip(actions_np, 0, n_actions - 1)]

    bc_probs = 0.9 * mu + 0.1 * (1.0 / max(n_actions, 1))
    cql_bias = np.where(actions_np < n_actions // 2, 1.1, 0.9)
    cql_probs = np.clip(mu * cql_bias, 1e-5, 1.0)

    return {
        "observations": observations_np,
        "actions": actions_np,
        "rewards": rewards_np,
        "terminals": terminals_np,
        "episode_terminals": episode_terminals_np,
        "episode_ids": episode_ids_np,
        "patient_ids": patient_ids_np,
        "split": split_np,
        "behavior_probs": np.asarray(mu, dtype=np.float32),
        "bc_probs": np.asarray(bc_probs, dtype=np.float32),
        "cql_probs": np.asarray(cql_probs, dtype=np.float32),
    }


def main() -> None:
    args = _parse_args()
    client, bigquery = _bq_client(project_id=args.gcp_project_id, location=args.bq_location)

    sepsis3_table = f"{args.mimic_project}.{args.derived_dataset}.sepsis3"
    sepsis3_cols = _get_table_columns(client, sepsis3_table)
    stay_col = _pick_column(sepsis3_cols, ["stay_id", "icustay_id"])
    subject_col = _pick_column(sepsis3_cols, ["subject_id"])
    hadm_col = _pick_column(sepsis3_cols, ["hadm_id"], required=False)
    sepsis_flag_col = _pick_column(sepsis3_cols, ["sepsis3", "sepsis3_bool", "sepsis3_flag"])

    hadm_expr = f"CAST(s.{hadm_col} AS INT64)" if hadm_col else "i.hadm_id"

    cohort_query = f"""
    SELECT
      CAST(s.{subject_col} AS INT64) AS subject_id,
      {hadm_expr} AS hadm_id,
      CAST(s.{stay_col} AS INT64) AS stay_id,
      i.intime,
      i.outtime,
      p.dod
    FROM `{sepsis3_table}` s
    JOIN `{args.mimic_project}.{args.icu_dataset}.icustays` i
      ON CAST(s.{stay_col} AS INT64) = i.stay_id
    LEFT JOIN `{args.mimic_project}.{args.hosp_dataset}.patients` p
      ON CAST(s.{subject_col} AS INT64) = p.subject_id
    WHERE CAST(s.{sepsis_flag_col} AS INT64) = 1
    ORDER BY i.intime
    LIMIT @max_stays
    """

    cohort_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("max_stays", "INT64", args.max_stays)]
    )
    cohort_df = client.query(cohort_query, job_config=cohort_cfg).result().to_dataframe(create_bqstorage_client=False)

    if cohort_df.empty:
        raise RuntimeError("No Sepsis-3 ICU stays found from mimiciv_derived.sepsis3 with current configuration.")

    cohort: dict[str, StayInfo] = {}
    for row in cohort_df.itertuples(index=False):
        stay_id = str(int(row.stay_id))
        start = _to_ts(row.intime)
        end = _to_ts(row.outtime)
        if start is None or end is None or end <= start:
            continue
        cohort[stay_id] = StayInfo(
            patient_id=str(int(row.subject_id)),
            hadm_id=str(int(row.hadm_id)) if not pd.isna(row.hadm_id) else "",
            start=start,
            end=end,
            dod=_to_ts(row.dod),
        )

    if not cohort:
        raise RuntimeError("Sepsis-3 query returned rows, but none had valid ICU interval metadata.")

    hadm_ids = sorted({int(v.hadm_id) for v in cohort.values() if v.hadm_id})
    stay_ids = sorted({int(k) for k in cohort.keys()})

    if not hadm_ids:
        raise RuntimeError("Cohort has no hadm_id values, unable to pull laboratory features.")

    top_lab_query = f"""
    SELECT CAST(l.itemid AS STRING) AS code, COUNT(*) AS n
    FROM `{args.mimic_project}.{args.hosp_dataset}.labevents` l
    WHERE l.valuenum IS NOT NULL
      AND l.hadm_id IN UNNEST(@hadm_ids)
    GROUP BY code
    ORDER BY n DESC
    LIMIT @state_dim
    """
    top_lab_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ScalarQueryParameter("state_dim", "INT64", args.state_dim),
        ]
    )
    top_codes_df = client.query(top_lab_query, job_config=top_lab_cfg).result().to_dataframe(create_bqstorage_client=False)
    top_codes = top_codes_df["code"].astype(str).tolist()
    if not top_codes:
        raise RuntimeError("No lab item IDs were found for the Sepsis-3 cohort.")

    lab_query = f"""
    SELECT
      CAST(l.hadm_id AS INT64) AS hadm_id,
      l.charttime,
      CAST(l.itemid AS STRING) AS code,
      CAST(l.valuenum AS FLOAT64) AS value
    FROM `{args.mimic_project}.{args.hosp_dataset}.labevents` l
    WHERE l.valuenum IS NOT NULL
      AND l.hadm_id IN UNNEST(@hadm_ids)
      AND CAST(l.itemid AS STRING) IN UNNEST(@top_codes)
    """
    lab_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ArrayQueryParameter("top_codes", "STRING", top_codes),
        ]
    )
    labs_df = client.query(lab_query, job_config=lab_cfg).result().to_dataframe(create_bqstorage_client=False)

    hadm_windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for sid, st in cohort.items():
        if st.hadm_id:
            hadm_windows[st.hadm_id].append((sid, st.start, st.end))

    obs_rows: list[dict[str, Any]] = []
    code_counts: Counter[str] = Counter()
    for row in labs_df.itertuples(index=False):
        hadm = str(int(row.hadm_id)) if not pd.isna(row.hadm_id) else ""
        ts = _to_ts(row.charttime)
        if not hadm or ts is None:
            continue
        for stay_id, start, end in hadm_windows.get(hadm, []):
            if start <= ts <= end:
                code = str(row.code)
                obs_rows.append(
                    {
                        "stay_id": stay_id,
                        "time": ts,
                        "code": code,
                        "value": float(row.value),
                    }
                )
                code_counts[code] += 1
                break

    if not obs_rows:
        raise RuntimeError("No numeric lab observations could be mapped into Sepsis-3 ICU windows.")

    action_query = f"""
    SELECT
      CAST(ie.stay_id AS INT64) AS stay_id,
      ie.starttime,
      LOWER(COALESCE(di.label, '')) AS label,
      CAST(ie.rate AS FLOAT64) AS rate,
      LOWER(COALESCE(ie.rateuom, '')) AS rateuom,
      CAST(ie.amount AS FLOAT64) AS amount,
      LOWER(COALESCE(ie.amountuom, '')) AS amountuom,
      CAST(ie.patientweight AS FLOAT64) AS patientweight
    FROM `{args.mimic_project}.{args.icu_dataset}.inputevents` ie
    LEFT JOIN `{args.mimic_project}.{args.icu_dataset}.d_items` di
      ON ie.itemid = di.itemid
    WHERE ie.stay_id IN UNNEST(@stay_ids)
    """
    action_cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
    )
    action_df = client.query(action_query, job_config=action_cfg).result().to_dataframe(create_bqstorage_client=False)

    action_rows: list[dict[str, Any]] = []
    for row in action_df.itertuples(index=False):
        stay_id = str(int(row.stay_id)) if not pd.isna(row.stay_id) else ""
        if stay_id not in cohort:
            continue

        ts = _to_ts(row.starttime)
        if ts is None:
            continue

        fluid_ml = _fluid_ml_from_event(
            label=str(row.label),
            rate=row.rate,
            rate_uom=str(row.rateuom),
            amount=row.amount,
            amount_uom=str(row.amountuom),
        )
        vaso_ne = _rate_to_ne_equivalent(
            label=str(row.label),
            rate=row.rate,
            rate_uom=str(row.rateuom),
            amount=row.amount,
            amount_uom=str(row.amountuom),
            patient_weight=row.patientweight,
        )

        action_rows.append(
            {
                "stay_id": stay_id,
                "time": ts,
                "fluid_ml": fluid_ml,
                "vasopressor_ne": vaso_ne,
            }
        )

    top_codes = [code for code, _ in code_counts.most_common(max(1, args.state_dim))]
    arrays = _build_arrays(
        cohort=cohort,
        obs_rows=obs_rows,
        action_rows=action_rows,
        top_codes=top_codes,
        action_bins=args.action_bins,
        max_hours=args.max_hours,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **arrays)

    ope_df = pd.DataFrame(
        {
            "episode_id": arrays["episode_ids"],
            "reward": arrays["rewards"],
            "action": arrays["actions"],
            "mu_prob": arrays["behavior_probs"],
            "bc_prob": arrays["bc_probs"],
            "cql_prob": arrays["cql_probs"],
            "patient_id": arrays["patient_ids"],
            "split": arrays["split"],
        }
    )
    args.out_ope_csv.parent.mkdir(parents=True, exist_ok=True)
    ope_df.to_csv(args.out_ope_csv, index=False)

    summary = {
        "source": "bigquery_mimiciv_derived_sepsis3",
        "gcp_project_id": args.gcp_project_id,
        "mimic_project": args.mimic_project,
        "cohort_icu_stays": int(len(cohort)),
        "cohort_sepsis_patients": int(len({v.patient_id for v in cohort.values()})),
        "n_transitions": int(arrays["observations"].shape[0]),
        "state_dim": int(arrays["observations"].shape[1]),
        "observed_actions": int(np.unique(arrays["actions"]).shape[0]),
        "mean_reward": float(np.mean(arrays["rewards"])),
        "split_counts": {
            "train": int(np.sum(arrays["split"] == "train")),
            "val": int(np.sum(arrays["split"] == "val")),
            "test": int(np.sum(arrays["split"] == "test")),
        },
        "output_npz": str(args.out_npz),
        "output_ope_csv": str(args.out_ope_csv),
        "top_feature_codes": top_codes,
        "vasopressor_normalization": "NE-equivalent: NE + EPI + 2.5*Vasopressin(units/min) + 0.5*Dopamine(mcg/kg/min)",
    }

    write_json(args.summary_json, summary)

    print("BigQuery Sepsis-3 extraction complete")
    print(summary)


if __name__ == "__main__":
    main()
