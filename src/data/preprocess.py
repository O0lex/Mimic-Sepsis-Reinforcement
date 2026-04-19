from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common.io import write_json


SPARSE_CLINICAL_COLS = [
    "lactate",
    "creatinine_lab",
    "wbc",
    "sofa_total",
    "gcs",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
]

MINIMAL_PROFILE = [
    "hr",
    "map",
    "spo2",
    "temp",
    "lactate",
    "sofa_total",
    "age",
    "gender_bin",
]

FULL_PROFILE = [
    "hr",
    "map",
    "spo2",
    "temp",
    "resp_rate",
    "sbp",
    "dbp",
    "lactate",
    "creatinine_lab",
    "wbc",
    "gcs",
    "sofa_total",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
    "age",
    "weight",
    "charlson",
    "gender_bin",
    "has_chf",
    "has_copd",
    "has_ckd",
    "has_diabetes",
    "has_liver_disease",
    "has_malignancy",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BigQuery clinical parquet into minimal/full MDP datasets for offline RL."
    )
    parser.add_argument("--in-parquet", type=Path, default=Path("data/raw/mimic_sepsis_final.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["terminal", "sofa_delta", "composite"],
        default="terminal",
        help="Reward construction mode. 'terminal' preserves legacy +/-1 endpoint rewards.",
    )
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=1.0,
        help="Scale factor for terminal reward (+1/-1) component.",
    )
    parser.add_argument(
        "--sofa-weight",
        type=float,
        default=0.1,
        help="Weight for SOFA delta reward component in composite mode.",
    )
    parser.add_argument(
        "--sofa-clip",
        type=float,
        default=1.0,
        help="Clip absolute SOFA delta reward at this value.",
    )
    return parser.parse_args()


def _pick_first(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    if required:
        raise ValueError(f"Missing required columns. Tried: {candidates}")
    return None


def _as_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _build_gender_binary(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    out = np.zeros((s.shape[0],), dtype=np.float32)
    out[s.isin(["m", "male", "1", "true"])] = 1.0
    return out


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0


def _fit_scaler_from_train(df: pd.DataFrame, feature_cols: list[str], split_col: str) -> StandardScaler:
    scaler = StandardScaler()
    train_mask = df[split_col].astype(str) == "train"
    if not np.any(train_mask):
        raise RuntimeError("No train split rows available for normalization.")
    scaler.fit(df.loc[train_mask, feature_cols].to_numpy(dtype=np.float32))
    return scaler


def _profile_to_npz(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_npz: Path,
    out_ope_csv: Path,
    split_col: str,
    stay_col: str,
    action_col: str,
    reward_col: str,
    terminal_col: str,
) -> dict[str, object]:
    n_actions = int(df[action_col].max()) + 1 if df.shape[0] else 25

    train_mask = df[split_col].astype(str) == "train"
    laplace = 1.0
    train_actions = df.loc[train_mask, action_col].to_numpy(dtype=np.int64)
    counts = np.bincount(train_actions, minlength=n_actions).astype(np.float64) + laplace
    probs = counts / np.clip(np.sum(counts), 1.0, None)

    actions = df[action_col].to_numpy(dtype=np.int64)
    mu = probs[np.clip(actions, 0, n_actions - 1)]
    bc_probs = np.clip(0.9 * mu + 0.1 * (1.0 / max(n_actions, 1)), 1e-6, 1.0)
    cql_bias = np.where(actions < n_actions // 2, 1.1, 0.9)
    cql_probs = np.clip(mu * cql_bias, 1e-6, 1.0)

    episode_ids = pd.factorize(df[stay_col].astype(str), sort=False)[0].astype(np.int64)
    terminals = df[terminal_col].to_numpy(dtype=np.float32)

    arrays = {
        "observations": df[feature_cols].to_numpy(dtype=np.float32),
        "actions": actions.astype(np.int64),
        "rewards": df[reward_col].to_numpy(dtype=np.float32),
        "terminals": terminals,
        "episode_terminals": terminals.copy(),
        "episode_ids": episode_ids,
        "patient_ids": df[stay_col].astype(str).to_numpy(),
        "split": df[split_col].astype(str).to_numpy(),
        "behavior_probs": mu.astype(np.float32),
        "bc_probs": bc_probs.astype(np.float32),
        "cql_probs": cql_probs.astype(np.float32),
    }

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)

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
    out_ope_csv.parent.mkdir(parents=True, exist_ok=True)
    ope_df.to_csv(out_ope_csv, index=False)

    return {
        "out_npz": str(out_npz),
        "out_ope_csv": str(out_ope_csv),
        "state_dim": int(arrays["observations"].shape[1]),
        "n_transitions": int(arrays["observations"].shape[0]),
        "n_stays": int(np.unique(arrays["patient_ids"]).shape[0]),
        "n_actions_observed": int(np.unique(arrays["actions"]).shape[0]),
        "mean_reward": float(np.mean(arrays["rewards"])) if arrays["rewards"].size else 0.0,
        "features": feature_cols,
    }


def build_mdp_profiles(args: argparse.Namespace) -> None:
    if not args.in_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.in_parquet}")

    print("Loading parquet...", args.in_parquet)
    df = pd.read_parquet(args.in_parquet)

    stay_col = _pick_first(df, ["stay_id", "icustay_id"])  # required
    window_col = _pick_first(df, ["window_idx", "step", "timestep"], required=False)
    action_fluid_col = _pick_first(df, ["action_fluid", "iv_fluid", "fluid_ml", "fluids"], required=False)
    action_vaso_col = _pick_first(df, ["action_vaso", "vaso", "vasopressor", "vaso_ne", "vasopressor_ne"], required=False)
    mortality_col = _pick_first(df, ["mortality", "hospital_expire_flag", "died", "death_flag"], required=False)
    gender_col = _pick_first(df, ["gender", "sex"], required=False)

    if action_fluid_col is None:
        df["action_fluid"] = 0.0
        action_fluid_col = "action_fluid"
    if action_vaso_col is None:
        df["action_vaso"] = 0.0
        action_vaso_col = "action_vaso"

    if mortality_col is None:
        print("Warning: mortality column missing. Setting terminal rewards to 0.")
        df["mortality"] = 0
        mortality_col = "mortality"

    if window_col is None:
        df["window_idx"] = df.groupby(stay_col).cumcount()
        window_col = "window_idx"

    # Standardize commonly used column names when aliases are present.
    renames = {}
    for target, candidates in {
        "hr": ["hr", "heart_rate"],
        "map": ["map", "meanbp", "mean_arterial_pressure"],
        "spo2": ["spo2", "o2sat", "oxygen_saturation"],
        "temp": ["temp", "temperature"],
        "resp_rate": ["resp_rate", "rr", "respiratory_rate"],
        "sbp": ["sbp", "sysbp", "systolic_bp"],
        "dbp": ["dbp", "diasbp", "diastolic_bp"],
        "weight": ["weight", "admission_weight"],
    }.items():
        src = _pick_first(df, candidates, required=False)
        if src is not None and src != target:
            renames[src] = target
    if renames:
        df = df.rename(columns=renames)

    if gender_col is not None:
        df["gender_bin"] = _build_gender_binary(df[gender_col])
    else:
        df["gender_bin"] = 0.0

    _as_numeric(
        df,
        [
            action_fluid_col,
            action_vaso_col,
            mortality_col,
            window_col,
            "hr",
            "map",
            "spo2",
            "temp",
            "resp_rate",
            "sbp",
            "dbp",
            "lactate",
            "creatinine_lab",
            "wbc",
            "sofa_total",
            "gcs",
            "respiration",
            "coagulation",
            "liver",
            "cardiovascular",
            "cns",
            "renal",
            "age",
            "weight",
            "charlson",
            "has_chf",
            "has_copd",
            "has_ckd",
            "has_diabetes",
            "has_liver_disease",
            "has_malignancy",
        ],
    )

    _ensure_cols(df, SPARSE_CLINICAL_COLS + ["age", "weight", "charlson"])

    # Grouped forward/back fill for sparse labs and SOFA-related signals.
    df = df.sort_values([stay_col, window_col]).reset_index(drop=True)
    df[SPARSE_CLINICAL_COLS] = (
        df.groupby(stay_col, sort=False)[SPARSE_CLINICAL_COLS].ffill().bfill().fillna(0.0)
    )

    # Ensure numerical defaults for remaining profile columns.
    _ensure_cols(df, FULL_PROFILE)
    for c in FULL_PROFILE:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Action discretization: 5x5 grid.
    df["fluid_idx"] = pd.cut(
        df[action_fluid_col],
        bins=[-np.inf, 0.0, 50.0, 500.0, 1000.0, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["vaso_idx"] = pd.cut(
        df[action_vaso_col],
        bins=[-np.inf, 0.0, 0.05, 0.2, 0.45, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)
    df["action"] = (df["fluid_idx"] * 5 + df["vaso_idx"]).astype(np.int64)

    # Terminal index and reward mapping.
    df["is_terminal"] = (
        df.groupby(stay_col, sort=False)[window_col].transform("max") == df[window_col]
    ).astype(np.float32)

    df["terminal_reward"] = 0.0
    died = pd.to_numeric(df[mortality_col], errors="coerce").fillna(0).astype(int)
    df.loc[(df["is_terminal"] > 0) & (died == 0), "terminal_reward"] = 1.0
    df.loc[(df["is_terminal"] > 0) & (died == 1), "terminal_reward"] = -1.0
    df["terminal_reward"] = df["terminal_reward"] * float(args.terminal_reward_scale)

    sofa_prev = df.groupby(stay_col, sort=False)["sofa_total"].shift(1)
    df["sofa_delta_reward"] = (sofa_prev - df["sofa_total"]).fillna(0.0)
    sofa_clip = abs(float(args.sofa_clip))
    if sofa_clip > 0:
        df["sofa_delta_reward"] = np.clip(df["sofa_delta_reward"], -sofa_clip, sofa_clip)

    if args.reward_mode == "terminal":
        df["reward"] = df["terminal_reward"]
    elif args.reward_mode == "sofa_delta":
        df["reward"] = df["sofa_delta_reward"]
    else:
        df["reward"] = df["terminal_reward"] + float(args.sofa_weight) * df["sofa_delta_reward"]

    # Patient-level split.
    rng = np.random.default_rng(args.seed)
    unique_stays = df[stay_col].astype(str).drop_duplicates().to_numpy()
    shuffled = rng.permutation(unique_stays)
    n_train = int(np.floor(args.train_frac * shuffled.shape[0]))
    n_val = int(np.floor(args.val_frac * shuffled.shape[0]))
    n_train = max(1, min(n_train, shuffled.shape[0]))
    n_val = max(0, min(n_val, shuffled.shape[0] - n_train))

    train_stays = set(shuffled[:n_train].tolist())
    val_stays = set(shuffled[n_train : n_train + n_val].tolist())

    df["split"] = "test"
    stay_str = df[stay_col].astype(str)
    df.loc[stay_str.isin(train_stays), "split"] = "train"
    df.loc[stay_str.isin(val_stays), "split"] = "val"

    # Normalize feature profiles from train-only statistics.
    minimal_cols = [c for c in MINIMAL_PROFILE if c in df.columns]
    full_cols = [c for c in FULL_PROFILE if c in df.columns]

    min_scaler = _fit_scaler_from_train(df=df, feature_cols=minimal_cols, split_col="split")
    full_scaler = _fit_scaler_from_train(df=df, feature_cols=full_cols, split_col="split")

    df_min = df.copy()
    df_full = df.copy()
    df_min[minimal_cols] = min_scaler.transform(df_min[minimal_cols].to_numpy(dtype=np.float32))
    df_full[full_cols] = full_scaler.transform(df_full[full_cols].to_numpy(dtype=np.float32))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    minimal_npz = args.out_dir / "mimic_mdp_minimal_raw.npz"
    minimal_ope = args.out_dir / "mimic_mdp_minimal_ope_table.csv"
    full_npz = args.out_dir / "mimic_mdp_full_raw.npz"
    full_ope = args.out_dir / "mimic_mdp_full_ope_table.csv"

    minimal_summary = _profile_to_npz(
        df=df_min,
        feature_cols=minimal_cols,
        out_npz=minimal_npz,
        out_ope_csv=minimal_ope,
        split_col="split",
        stay_col=stay_col,
        action_col="action",
        reward_col="reward",
        terminal_col="is_terminal",
    )

    full_summary = _profile_to_npz(
        df=df_full,
        feature_cols=full_cols,
        out_npz=full_npz,
        out_ope_csv=full_ope,
        split_col="split",
        stay_col=stay_col,
        action_col="action",
        reward_col="reward",
        terminal_col="is_terminal",
    )

    null_report = df.isnull().sum().sort_values(ascending=False)
    null_report_json = {k: int(v) for k, v in null_report.items() if int(v) > 0}

    summary = {
        "input_parquet": str(args.in_parquet),
        "n_rows": int(df.shape[0]),
        "n_stays": int(df[stay_col].astype(str).nunique()),
        "stay_col": stay_col,
        "window_col": window_col,
        "action_columns": {"fluid": action_fluid_col, "vaso": action_vaso_col},
        "mortality_col": mortality_col,
        "split_counts": {
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
        "minimal_profile": minimal_summary,
        "full_profile": full_summary,
        "reward_config": {
            "reward_mode": args.reward_mode,
            "terminal_reward_scale": float(args.terminal_reward_scale),
            "sofa_weight": float(args.sofa_weight),
            "sofa_clip": float(args.sofa_clip),
            "reward_mean": float(df["reward"].mean()),
            "reward_std": float(df["reward"].std(ddof=0)),
            "reward_min": float(df["reward"].min()),
            "reward_max": float(df["reward"].max()),
        },
        "null_counts_nonzero": null_report_json,
    }

    write_json(args.out_dir / "mimic_mdp_profiles_summary.json", summary)
    print("Preprocessing complete.")
    print(summary)


def main() -> None:
    args = _parse_args()
    build_mdp_profiles(args)


if __name__ == "__main__":
    main()