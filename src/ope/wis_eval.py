from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed


def _build_d3_dataset_from_npz(npz_path: Path):
    try:
        from d3rlpy.dataset import MDPDataset
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required for model-based probability estimation.") from exc

    arrays = np.load(npz_path)
    observations = arrays["observations"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    terminals = arrays["terminals"]
    timeouts = np.zeros_like(terminals, dtype=np.float32)

    try:
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
    except TypeError:
        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=arrays["episode_terminals"],
        )

    return dataset, arrays


def _predict_logged_probs_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    dataset,
    bc_model_path: Path,
    batch_size: int,
) -> np.ndarray:
    try:
        import d3rlpy
        import torch
    except ImportError as exc:
        raise RuntimeError("d3rlpy and torch are required for BC-based mu estimation.") from exc

    algo = d3rlpy.algos.DiscreteBCConfig().create(device="cpu")
    algo.build_with_dataset(dataset)
    algo.load_model(str(bc_model_path))

    imitator = None
    modules = getattr(algo.impl, "modules", None)
    if modules is not None:
        imitator = getattr(modules, "imitator", None)
    if imitator is None:
        imitator = getattr(algo.impl, "_imitator", None)
    if imitator is None:
        raise RuntimeError("Unable to locate BC imitator network for probability extraction.")

    n = observations.shape[0]
    logged_probs = np.empty((n,), dtype=np.float64)
    action_idx = np.clip(actions.astype(np.int64), 0, None)

    for start in range(0, n, max(1, batch_size)):
        end = min(n, start + max(1, batch_size))
        obs_t = torch.tensor(observations[start:end], dtype=torch.float32)
        with torch.no_grad():
            dist_or_logits = imitator(obs_t)

        if hasattr(dist_or_logits, "probs"):
            probs = dist_or_logits.probs.detach().cpu().numpy()
        else:
            logits = dist_or_logits.detach().cpu().numpy()
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)

        max_action = probs.shape[1] - 1
        sel = np.clip(action_idx[start:end], 0, max_action)
        logged_probs[start:end] = probs[np.arange(probs.shape[0]), sel]

    # CHANGED: Increased floor from 1e-8 to 1e-4 to reduce variance
    return np.clip(logged_probs, 1e-4, 1.0)


def _predict_logged_probs_cql(
    observations: np.ndarray,
    actions: np.ndarray,
    dataset,
    cql_model_path: Path,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    try:
        import d3rlpy
        import torch
    except ImportError as exc:
        raise RuntimeError("d3rlpy and torch are required for CQL probability estimation.") from exc

    algo = d3rlpy.algos.DiscreteCQLConfig(alpha=1.0).create(device="cpu")
    algo.build_with_dataset(dataset)
    algo.load_model(str(cql_model_path))

    q_forwarder = getattr(algo.impl, "_q_func_forwarder", None)
    if q_forwarder is None or not hasattr(q_forwarder, "compute_expected_q"):
        raise RuntimeError("Unable to locate CQL q-function forwarder for probability extraction.")

    n = observations.shape[0]
    logged_probs = np.empty((n,), dtype=np.float64)
    action_idx = np.clip(actions.astype(np.int64), 0, None)
    temp = max(1e-6, float(temperature))

    for start in range(0, n, max(1, batch_size)):
        end = min(n, start + max(1, batch_size))
        obs_t = torch.tensor(observations[start:end], dtype=torch.float32)
        with torch.no_grad():
            q = q_forwarder.compute_expected_q(obs_t)
            probs_t = torch.softmax(q / temp, dim=1)
        probs = probs_t.detach().cpu().numpy()

        max_action = probs.shape[1] - 1
        sel = np.clip(action_idx[start:end], 0, max_action)
        logged_probs[start:end] = probs[np.arange(probs.shape[0]), sel]

    return np.clip(logged_probs, 1e-8, 1.0)


def build_episode_arrays(df: pd.DataFrame, policy_col: str, clip: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    grouped = df.groupby("episode_id", sort=False)
    episode_weights = []
    episode_returns = []

    for _, ep in grouped:
        pi = ep[policy_col].to_numpy(dtype=np.float64)
        mu = ep["mu_prob"].to_numpy(dtype=np.float64)
        rewards = ep["reward"].to_numpy(dtype=np.float64)

        # Calculate importance ratios
        # Using a floor of 1e-8 for mu here is fine because of cumulative clipping
        ratios = pi / np.clip(mu, 1e-8, None)
        
        # Cumulative product of ratios for the whole episode
        log_ratios = np.log(pi) - np.log(np.clip(mu, 1e-8, None))
        cum_weight = np.exp(np.sum(log_ratios))
        
        # CHANGED: Clip the total episode weight instead of individual steps
        if clip is not None:
            cum_weight = np.clip(cum_weight, 0.0, clip)

        episode_weights.append(float(cum_weight))
        episode_returns.append(float(np.sum(rewards)))

    return np.asarray(episode_weights, dtype=np.float64), np.asarray(episode_returns, dtype=np.float64)


def wis_from_episode_arrays(weights: np.ndarray, returns: np.ndarray) -> tuple[float, float]:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return 0.0, 0.0
    
    # Calculate WIS Mean
    estimate = float(np.sum(weights * returns) / denom)
    
    # Calculate ESS: (sum(w)^2) / sum(w^2)
    ess = float((denom**2) / np.sum(weights**2))
    
    return estimate, ess


def bootstrap_ci(df: pd.DataFrame, policy_col: str, n_boot: int, seed: int, clip: float | None = None) -> tuple[float, float, float, float]:
    set_seed(seed)
    episode_weights, episode_returns = build_episode_arrays(df, policy_col=policy_col, clip=clip)
    n_episodes = episode_weights.shape[0]
    estimates = np.empty(n_boot, dtype=np.float64)
    
    # Calculate global ESS for the whole test set
    _, global_ess = wis_from_episode_arrays(episode_weights, episode_returns)

    for i in range(n_boot):
        sampled_idx = np.random.randint(0, n_episodes, size=n_episodes)
        w = episode_weights[sampled_idx]
        g = episode_returns[sampled_idx]
        # Only take the first element (the WIS estimate) for the CI distribution
        val, _ = wis_from_episode_arrays(w, g)
        estimates[i] = val

    return float(np.mean(estimates)), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)), global_ess


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WIS OPE for BC and CQL policies.")
    parser.add_argument("--in-csv", type=Path, default=Path("outputs/data/mock_ope_table.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/ope/wis_summary.json"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--clip", type=float, default=20.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--mu-source", type=str, choices=["csv", "bc_model"], default="csv")
    parser.add_argument("--dataset-npz", type=Path, default=None)
    parser.add_argument("--bc-model", type=Path, default=None)
    parser.add_argument("--cql-model", type=Path, default=None)
    parser.add_argument("--cql-temperature", type=float, default=1.0)
    parser.add_argument("--prob-batch-size", type=int, default=2048)
    parser.add_argument("--sync-bc-prob-with-mu", action="store_true")
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

    if args.mu_source == "bc_model":
        if args.dataset_npz is None or args.bc_model is None:
            raise ValueError("--mu-source bc_model requires --dataset-npz and --bc-model.")
        if not args.dataset_npz.exists():
            raise FileNotFoundError(f"Dataset NPZ not found: {args.dataset_npz}")
        if not args.bc_model.exists():
            raise FileNotFoundError(f"BC model file not found: {args.bc_model}")

        dataset, arrays = _build_d3_dataset_from_npz(args.dataset_npz)
        observations = arrays["observations"]
        actions = arrays["actions"].astype(np.int64)

        if df.shape[0] != observations.shape[0]:
            raise ValueError(
                f"CSV rows ({df.shape[0]}) do not match dataset transitions ({observations.shape[0]})."
            )

        bc_mu = _predict_logged_probs_bc(
            observations=observations,
            actions=actions,
            dataset=dataset,
            bc_model_path=args.bc_model,
            batch_size=args.prob_batch_size,
        )
        df["mu_prob"] = bc_mu

        if args.sync_bc_prob_with_mu:
            df["bc_prob"] = bc_mu

        if args.cql_model is not None:
            if not args.cql_model.exists():
                raise FileNotFoundError(f"CQL model file not found: {args.cql_model}")
            cql_probs = _predict_logged_probs_cql(
                observations=observations,
                actions=actions,
                dataset=dataset,
                cql_model_path=args.cql_model,
                temperature=args.cql_temperature,
                batch_size=args.prob_batch_size,
            )
            df["cql_prob"] = cql_probs

        if "action" not in df.columns:
            df["action"] = actions

    split_used = "all"
    if "split" in df.columns and args.eval_split:
        df_split = df[df["split"].astype(str) == args.eval_split].copy()
        if not df_split.empty:
            df = df_split
            split_used = args.eval_split

    behavior_return = float(df.groupby("episode_id", sort=False)["reward"].sum().mean())

    bc_mean, bc_lo, bc_hi, bc_ess = bootstrap_ci(df, policy_col="bc_prob", n_boot=args.n_boot, seed=args.seed, clip=args.clip)
    cql_mean, cql_lo, cql_hi, cql_ess = bootstrap_ci(df, policy_col="cql_prob", n_boot=args.n_boot, seed=args.seed + 1, clip=args.clip)

    # 2. Build the final summary (Include ESS here so it's saved!)
    summary = {
        "behavior_episode_return": behavior_return,
        "bc_wis_mean": bc_mean,
        "bc_wis_ci95": [bc_lo, bc_hi],
        "bc_ess": bc_ess,  # Added for correctness
        "cql_wis_mean": cql_mean,
        "cql_wis_ci95": [cql_lo, cql_hi],
        "cql_ess": cql_ess,  # Added for correctness
        "clip": args.clip,
        "n_boot": args.n_boot,
        "eval_split": split_used,
        "n_transitions_eval": int(df.shape[0]),
        "n_episodes_eval": int(df["episode_id"].nunique()),
        "input_csv": str(args.in_csv),
        "mu_source": args.mu_source,
        "dataset_npz": str(args.dataset_npz) if args.dataset_npz is not None else None,
        "bc_model": str(args.bc_model) if args.bc_model is not None else None,
        "cql_model": str(args.cql_model) if args.cql_model is not None else None,
    }

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    write_json(args.out_json, summary)
    print("WIS evaluation complete")
    print(summary)


if __name__ == "__main__":
    main()
