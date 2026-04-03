from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed

SEPSIS_PATTERNS = (
    "sepsis",
    "septic",
    "septicemia",
)
SEPSIS_CODE_PREFIXES = (
    "A40",
    "A41",
    "R652",
    "038",
    "9959",
)
VASPRESSOR_NAMES = (
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "epinephrine",
    "dopamine",
    "dobutamine",
)


@dataclass
class EncounterInfo:
    patient_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mini MIMIC-IV FHIR sepsis cohort and build RL-ready arrays.")
    parser.add_argument(
        "--fhir-dir",
        type=Path,
        default=Path("data/mimic-iv-clinical-database-demo-on-fhir-2.0/mimic-fhir"),
    )
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/data/mimic_fhir_mdp_raw.npz"))
    parser.add_argument("--out-ope-csv", type=Path, default=Path("outputs/data/mimic_fhir_ope_table.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--action-bins", type=int, default=5)
    parser.add_argument("--max-hours", type=int, default=48)
    parser.add_argument("--max-observations", type=int, default=0)
    return parser.parse_args()


def _iter_ndjson(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _ref_id(resource_ref: str | None) -> str | None:
    if not resource_ref:
        return None
    if "/" in resource_ref:
        return resource_ref.split("/")[-1]
    return resource_ref


def _extract_code_display(resource: dict[str, Any]) -> tuple[str, str]:
    coding = ((resource.get("code") or {}).get("coding") or [{}])[0]
    code = str(coding.get("code") or "").strip()
    display = str(coding.get("display") or "").strip().lower()
    return code, display


def _to_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _load_patients(patient_path: Path) -> dict[str, pd.Timestamp | None]:
    deceased = {}
    for row in _iter_ndjson(patient_path):
        pid = row.get("id")
        deceased_ts = _to_timestamp(row.get("deceasedDateTime"))
        if pid:
            deceased[pid] = deceased_ts
    return deceased


def _load_icu_encounters(enc_path: Path) -> dict[str, EncounterInfo]:
    encounters: dict[str, EncounterInfo] = {}
    for row in _iter_ndjson(enc_path):
        eid = row.get("id")
        patient_id = _ref_id((row.get("subject") or {}).get("reference"))
        period = row.get("period") or {}
        start = _to_timestamp(period.get("start"))
        end = _to_timestamp(period.get("end"))
        if not eid or not patient_id or start is None or end is None:
            continue
        if end <= start:
            continue
        encounters[eid] = EncounterInfo(patient_id=patient_id, start=start, end=end)
    return encounters


def _find_sepsis_condition_sets(condition_path: Path) -> tuple[set[str], set[str]]:
    sepsis_patients: set[str] = set()
    sepsis_encounters: set[str] = set()

    for row in _iter_ndjson(condition_path):
        pid = _ref_id((row.get("subject") or {}).get("reference"))
        eid = _ref_id((row.get("encounter") or {}).get("reference"))
        code, display = _extract_code_display(row)

        code_u = code.upper().replace(".", "")
        is_sepsis = any(token in display for token in SEPSIS_PATTERNS) or any(
            code_u.startswith(prefix) for prefix in SEPSIS_CODE_PREFIXES
        )
        if not is_sepsis:
            continue

        if pid:
            sepsis_patients.add(pid)
        if eid:
            sepsis_encounters.add(eid)

    return sepsis_patients, sepsis_encounters


def _build_cohort_encounters(
    encounters: dict[str, EncounterInfo], sepsis_patients: set[str], sepsis_encounters: set[str]
) -> dict[str, EncounterInfo]:
    cohort = {}
    for eid, enc in encounters.items():
        if eid in sepsis_encounters or enc.patient_id in sepsis_patients:
            cohort[eid] = enc
    return cohort


def _build_patient_windows(cohort: dict[str, EncounterInfo]) -> dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]]:
    windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for eid, enc in cohort.items():
        windows[enc.patient_id].append((eid, enc.start, enc.end))

    for pid in windows:
        windows[pid].sort(key=lambda x: x[1])
    return windows


def _map_to_cohort_encounter(
    encounter_id: str | None,
    patient_id: str | None,
    ts: pd.Timestamp | None,
    encounter_ids: set[str],
    patient_windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]],
) -> str | None:
    if encounter_id and encounter_id in encounter_ids:
        return encounter_id

    if not patient_id or ts is None:
        return None

    for eid, start, end in patient_windows.get(patient_id, []):
        if start <= ts <= end:
            return eid
    return None


def _collect_observations(
    obs_path: Path,
    encounter_ids: set[str],
    patient_windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]],
    max_observations: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    code_counts: Counter[str] = Counter()

    for i, row in enumerate(_iter_ndjson(obs_path)):
        if max_observations and i >= max_observations:
            break

        raw_eid = _ref_id((row.get("encounter") or {}).get("reference"))
        pid = _ref_id((row.get("subject") or {}).get("reference"))

        code, display = _extract_code_display(row)
        if not code:
            continue

        value_qty = row.get("valueQuantity") or {}
        value = value_qty.get("value")
        if value is None:
            continue

        ts = _to_timestamp(row.get("effectiveDateTime"))
        if ts is None:
            continue

        eid = _map_to_cohort_encounter(
            encounter_id=raw_eid,
            patient_id=pid,
            ts=ts,
            encounter_ids=encounter_ids,
            patient_windows=patient_windows,
        )
        if not eid:
            continue

        rows.append(
            {
                "encounter_id": eid,
                "time": ts,
                "code": code,
                "display": display,
                "value": float(value),
            }
        )
        code_counts[code] += 1

    return rows, code_counts


def _collect_actions(
    med_path: Path,
    encounter_ids: set[str],
    patient_windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    for row in _iter_ndjson(med_path):
        raw_eid = _ref_id((row.get("context") or {}).get("reference"))
        pid = _ref_id((row.get("subject") or {}).get("reference"))

        med_codeable = row.get("medicationCodeableConcept") or {}
        coding = (med_codeable.get("coding") or [{}])[0]
        display = str(coding.get("display") or "").lower()

        dosage = row.get("dosage") or {}
        dose_value = ((dosage.get("dose") or {}).get("value"))
        dose_unit = str(((dosage.get("dose") or {}).get("unit") or "")).lower()
        rate_value = ((dosage.get("rateQuantity") or {}).get("value"))

        period = row.get("effectivePeriod") or {}
        ts = _to_timestamp(row.get("effectiveDateTime"))
        if ts is None:
            ts = _to_timestamp(period.get("start"))
        if ts is None:
            continue

        eid = _map_to_cohort_encounter(
            encounter_id=raw_eid,
            patient_id=pid,
            ts=ts,
            encounter_ids=encounter_ids,
            patient_windows=patient_windows,
        )
        if not eid:
            continue

        fluid_ml = 0.0
        if dose_value is not None:
            if "ml" in dose_unit or "millil" in dose_unit:
                fluid_ml = float(dose_value)
            elif any(tok in display for tok in ("nacl", "dextrose", "albumin", "ringer", "fluid")):
                fluid_ml = float(dose_value)

        vasopressor_rate = 0.0
        if any(name in display for name in VASPRESSOR_NAMES):
            if rate_value is not None:
                vasopressor_rate = float(rate_value)
            elif dose_value is not None:
                vasopressor_rate = float(dose_value)

        actions.append(
            {
                "encounter_id": eid,
                "time": ts,
                "fluid_ml": max(0.0, fluid_ml),
                "vasopressor": max(0.0, vasopressor_rate),
            }
        )

    return actions


def _discretize_action(fluid: float, vaso: float, fluid_bins: np.ndarray, vaso_bins: np.ndarray) -> int:
    fi = int(np.digitize(fluid, fluid_bins, right=False))
    vi = int(np.digitize(vaso, vaso_bins, right=False))
    fi = min(max(fi, 0), len(fluid_bins))
    vi = min(max(vi, 0), len(vaso_bins))
    return fi * (len(vaso_bins) + 1) + vi


def _build_arrays(
    cohort: dict[str, EncounterInfo],
    deceased_map: dict[str, pd.Timestamp | None],
    obs_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    top_codes: list[str],
    action_bins: int,
    max_hours: int,
    seed: int,
) -> dict[str, np.ndarray]:
    set_seed(seed)

    code_to_idx = {c: i for i, c in enumerate(top_codes)}

    obs_by_enc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obs_rows:
        obs_by_enc[row["encounter_id"]].append(row)

    act_by_enc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_fluids = []
    all_vaso = []
    for row in action_rows:
        act_by_enc[row["encounter_id"]].append(row)
        all_fluids.append(row["fluid_ml"])
        all_vaso.append(row["vasopressor"])

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

    for ep_idx, (eid, enc) in enumerate(cohort.items()):
        total_hours = max(1, int(np.ceil((enc.end - enc.start).total_seconds() / 3600.0)))
        horizon = min(max_hours, total_hours)

        feature_state = np.zeros((len(top_codes),), dtype=np.float32)

        enc_obs = sorted(obs_by_enc.get(eid, []), key=lambda x: x["time"])
        enc_actions = sorted(act_by_enc.get(eid, []), key=lambda x: x["time"])
        obs_ptr = 0

        for t in range(horizon):
            t0 = enc.start + pd.Timedelta(hours=t)
            t1 = t0 + pd.Timedelta(hours=1)

            while obs_ptr < len(enc_obs) and enc_obs[obs_ptr]["time"] < t1:
                row = enc_obs[obs_ptr]
                idx = code_to_idx.get(row["code"])
                if idx is not None:
                    feature_state[idx] = row["value"]
                obs_ptr += 1

            fluid = 0.0
            vaso = 0.0
            for a in enc_actions:
                if t0 <= a["time"] < t1:
                    fluid += a["fluid_ml"]
                    vaso = max(vaso, a["vasopressor"])

            action_idx = _discretize_action(fluid, vaso, fluid_bins, vaso_bins)
            done = 1.0 if t == horizon - 1 else 0.0

            reward = 0.0
            if done > 0:
                deceased = deceased_map.get(enc.patient_id)
                if deceased is not None and deceased <= (enc.end + pd.Timedelta(days=30)):
                    reward = -1.0
                else:
                    reward = 1.0

            observations.append(feature_state.copy())
            actions.append(action_idx)
            rewards.append(reward)
            terminals.append(done)
            episode_terminals.append(done)
            episode_ids.append(ep_idx)

    observations_np = np.asarray(observations, dtype=np.float32)
    actions_np = np.asarray(actions, dtype=np.int64)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    terminals_np = np.asarray(terminals, dtype=np.float32)
    episode_terminals_np = np.asarray(episode_terminals, dtype=np.float32)
    episode_ids_np = np.asarray(episode_ids, dtype=np.int64)

    n_actions = int(actions_np.max() + 1) if actions_np.size else action_bins * action_bins
    counts = np.bincount(actions_np, minlength=n_actions).astype(np.float64)
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
        "behavior_probs": np.asarray(mu, dtype=np.float32),
        "bc_probs": np.asarray(bc_probs, dtype=np.float32),
        "cql_probs": np.asarray(cql_probs, dtype=np.float32),
    }


def main() -> None:
    args = _parse_args()
    fhir_dir = args.fhir_dir
    if not fhir_dir.exists():
        raise FileNotFoundError(f"FHIR folder not found: {fhir_dir}")

    patient_path = fhir_dir / "Patient.ndjson"
    condition_path = fhir_dir / "Condition.ndjson"
    encounter_icu_path = fhir_dir / "EncounterICU.ndjson"
    obs_lab_path = fhir_dir / "ObservationLabevents.ndjson"
    med_icu_path = fhir_dir / "MedicationAdministrationICU.ndjson"

    required = [patient_path, condition_path, encounter_icu_path, obs_lab_path, med_icu_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required FHIR files: {missing}")

    deceased_map = _load_patients(patient_path)
    encounters = _load_icu_encounters(encounter_icu_path)
    sepsis_patients, sepsis_encounters = _find_sepsis_condition_sets(condition_path)
    cohort = _build_cohort_encounters(encounters, sepsis_patients, sepsis_encounters)

    if not cohort:
        raise RuntimeError("No sepsis-matching ICU encounters were found in this mini dataset.")

    cohort_ids = set(cohort.keys())
    patient_windows = _build_patient_windows(cohort)
    obs_rows, code_counts = _collect_observations(
        obs_lab_path,
        encounter_ids=cohort_ids,
        patient_windows=patient_windows,
        max_observations=args.max_observations,
    )
    action_rows = _collect_actions(med_icu_path, encounter_ids=cohort_ids, patient_windows=patient_windows)

    if not obs_rows:
        raise RuntimeError("No numeric observations were found for the selected sepsis ICU cohort.")

    top_codes = [code for code, _ in code_counts.most_common(max(1, args.state_dim))]
    arrays = _build_arrays(
        cohort=cohort,
        deceased_map=deceased_map,
        obs_rows=obs_rows,
        action_rows=action_rows,
        top_codes=top_codes,
        action_bins=args.action_bins,
        max_hours=args.max_hours,
        seed=args.seed,
    )

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **arrays)

    ope_df = pd.DataFrame(
        {
            "episode_id": arrays["episode_ids"],
            "reward": arrays["rewards"],
            "mu_prob": arrays["behavior_probs"],
            "bc_prob": arrays["bc_probs"],
            "cql_prob": arrays["cql_probs"],
        }
    )
    args.out_ope_csv.parent.mkdir(parents=True, exist_ok=True)
    ope_df.to_csv(args.out_ope_csv, index=False)

    summary = {
        "fhir_dir": str(fhir_dir),
        "cohort_icu_encounters": int(len(cohort)),
        "cohort_sepsis_patients": int(len(sepsis_patients)),
        "n_transitions": int(arrays["observations"].shape[0]),
        "state_dim": int(arrays["observations"].shape[1]),
        "observed_actions": int(np.unique(arrays["actions"]).shape[0]),
        "mean_reward": float(np.mean(arrays["rewards"])),
        "output_npz": str(args.out_npz),
        "output_ope_csv": str(args.out_ope_csv),
        "top_feature_codes": top_codes,
    }
    write_json(Path("outputs/data/mimic_fhir_summary.json"), summary)

    print("FHIR sepsis extraction complete")
    print(summary)


if __name__ == "__main__":
    main()
