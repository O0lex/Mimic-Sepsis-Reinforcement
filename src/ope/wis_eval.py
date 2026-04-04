from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed


def build_episode_arrays(df: pd.DataFrame, policy_col: str, clip: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    grouped = df.groupby("episode_id", sort=False)
    episode_weights = []
    episode_returns = []

    for _, ep in grouped:
        pi = ep[policy_col].to_numpy(dtype=np.float64)
        mu = ep["mu_prob"].to_numpy(dtype=np.float64)
        rewards = ep["reward"].to_numpy(dtype=np.float64)

        ratio = pi / np.clip(mu, 1e-8, None)
        if clip is not None:
            ratio = np.clip(ratio, 0.0, clip)

        episode_weights.append(float(np.prod(ratio)))
        episode_returns.append(float(np.sum(rewards)))

    return np.asarray(episode_weights, dtype=np.float64), np.asarray(episode_returns, dtype=np.float64)


def wis_from_episode_arrays(weights: np.ndarray, returns: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(weights * returns) / denom)


def bootstrap_ci(df: pd.DataFrame, policy_col: str, n_boot: int, seed: int, clip: float | None = None) -> tuple[float, float, float]:
    set_seed(seed)
    episode_weights, episode_returns = build_episode_arrays(df, policy_col=policy_col, clip=clip)
    n_episodes = episode_weights.shape[0]
    estimates = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sampled_idx = np.random.randint(0, n_episodes, size=n_episodes)
        w = episode_weights[sampled_idx]
        g = episode_returns[sampled_idx]
        estimates[i] = wis_from_episode_arrays(w, g)

    return float(np.mean(estimates)), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WIS OPE for BC and CQL policies.")
    parser.add_argument("--in-csv", type=Path, default=Path("outputs/data/mock_ope_table.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/ope/wis_summary.json"))
    parser.add_argument("--clip", type=float, default=20.0)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=str, default="test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    required_cols = {"episode_id", "reward", "mu_prob", "bc_prob", "cql_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    split_used = "all"
    if "split" in df.columns and args.eval_split:
        df_split = df[df["split"].astype(str) == args.eval_split].copy()
        if not df_split.empty:
            df = df_split
            split_used = args.eval_split

    behavior_return = float(df.groupby("episode_id", sort=False)["reward"].sum().mean())

    bc_mean, bc_lo, bc_hi = bootstrap_ci(df, policy_col="bc_prob", n_boot=args.n_boot, seed=args.seed, clip=args.clip)
    cql_mean, cql_lo, cql_hi = bootstrap_ci(
        df, policy_col="cql_prob", n_boot=args.n_boot, seed=args.seed + 1, clip=args.clip
    )

    summary = {
        "behavior_episode_return": behavior_return,
        "bc_wis_mean": bc_mean,
        "bc_wis_ci95": [bc_lo, bc_hi],
        "cql_wis_mean": cql_mean,
        "cql_wis_ci95": [cql_lo, cql_hi],
        "clip": args.clip,
        "n_boot": args.n_boot,
        "eval_split": split_used,
        "n_transitions_eval": int(df.shape[0]),
        "n_episodes_eval": int(df["episode_id"].nunique()),
        "input_csv": str(args.in_csv),
    }

    write_json(args.out_json, summary)
    print("WIS evaluation complete")
    print(summary)


if __name__ == "__main__":
    main()
