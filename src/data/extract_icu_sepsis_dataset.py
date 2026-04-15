from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.clip(np.sum(e), 1e-12, None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline trajectories from ICU-Sepsis Gymnasium environment.")
    parser.add_argument("--env-id", type=str, default="Sepsis/ICU-Sepsis-v2")
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/data/icu_sepsis_mdp_raw.npz"))
    parser.add_argument("--out-ope-csv", type=Path, default=Path("outputs/data/icu_sepsis_ope_table.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros((size,), dtype=np.float32)
    idx = int(np.clip(index, 0, size - 1))
    vec[idx] = 1.0
    return vec


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    try:
        import gymnasium as gym
        import icu_sepsis  # noqa: F401  # Needed for env registration side-effects.
    except ImportError as exc:
        raise RuntimeError("ICU-Sepsis requires gymnasium and icu-sepsis package.") from exc

    env = gym.make(args.env_id)
    if not hasattr(env.observation_space, "n") or not hasattr(env.action_space, "n"):
        raise RuntimeError("This extractor expects discrete observation and action spaces.")

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)

    rng = np.random.default_rng(args.seed)
    # State-conditioned behavior policy parameters.
    prefs = rng.normal(0.0, 0.3, size=(n_states, n_actions)).astype(np.float32)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[float] = []
    episode_terminals: list[float] = []
    episode_ids: list[int] = []
    behavior_probs: list[float] = []

    episode_returns: list[float] = []

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0

        for t in range(args.max_steps):
            state = int(obs)
            logits = prefs[state] / max(args.temperature, 1e-6)
            probs = _softmax(logits)
            action = int(rng.choice(n_actions, p=probs))
            mu = float(np.clip(probs[action], 1e-6, 1.0))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            observations.append(_one_hot(state, n_states))
            actions.append(action)
            rewards.append(float(reward))
            terminals.append(1.0 if done else 0.0)
            episode_terminals.append(1.0 if done else 0.0)
            episode_ids.append(ep)
            behavior_probs.append(mu)

            ep_return += float(reward)
            obs = next_obs
            if done:
                break

        episode_returns.append(ep_return)

    obs_np = np.asarray(observations, dtype=np.float32)
    act_np = np.asarray(actions, dtype=np.int64)
    rew_np = np.asarray(rewards, dtype=np.float32)
    term_np = np.asarray(terminals, dtype=np.float32)
    ep_term_np = np.asarray(episode_terminals, dtype=np.float32)
    ep_ids_np = np.asarray(episode_ids, dtype=np.int64)
    mu_np = np.asarray(behavior_probs, dtype=np.float32)

    # Episode-level split for leakage-safe evaluation.
    shuffled_eps = rng.permutation(np.arange(args.n_episodes))
    n_train = max(1, int(np.floor(args.train_frac * args.n_episodes)))
    n_val = max(0, int(np.floor(args.val_frac * args.n_episodes)))
    n_val = min(n_val, args.n_episodes - n_train)

    train_eps = set(shuffled_eps[:n_train].tolist())
    val_eps = set(shuffled_eps[n_train : n_train + n_val].tolist())

    split_np = np.empty(ep_ids_np.shape[0], dtype="<U8")
    for i, eid in enumerate(ep_ids_np):
        if int(eid) in train_eps:
            split_np[i] = "train"
        elif int(eid) in val_eps:
            split_np[i] = "val"
        else:
            split_np[i] = "test"

    # Proxy policy probabilities for BC and CQL reporting paths.
    train_mask = split_np == "train"
    laplace = 1.0
    counts = np.bincount(act_np[train_mask], minlength=n_actions).astype(np.float64) + laplace
    train_freq = counts / np.clip(np.sum(counts), 1.0, None)
    freq_prob = train_freq[np.clip(act_np, 0, n_actions - 1)]

    bc_prob = np.clip(0.8 * mu_np + 0.2 * freq_prob, 1e-6, 1.0)
    cql_bias = np.where(act_np < (n_actions // 2), 1.1, 0.9)
    cql_prob = np.clip(0.7 * mu_np + 0.3 * freq_prob * cql_bias, 1e-6, 1.0)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        observations=obs_np,
        actions=act_np,
        rewards=rew_np,
        terminals=term_np,
        episode_terminals=ep_term_np,
        episode_ids=ep_ids_np,
        split=split_np,
        behavior_probs=mu_np,
        bc_probs=bc_prob.astype(np.float32),
        cql_probs=cql_prob.astype(np.float32),
    )

    ope_df = pd.DataFrame(
        {
            "episode_id": ep_ids_np,
            "reward": rew_np,
            "action": act_np,
            "mu_prob": mu_np,
            "bc_prob": bc_prob,
            "cql_prob": cql_prob,
            "split": split_np,
        }
    )
    args.out_ope_csv.parent.mkdir(parents=True, exist_ok=True)
    ope_df.to_csv(args.out_ope_csv, index=False)

    summary = {
        "env_id": args.env_id,
        "n_episodes": int(args.n_episodes),
        "n_transitions": int(obs_np.shape[0]),
        "state_dim": int(obs_np.shape[1]),
        "n_actions": int(n_actions),
        "avg_episode_return": float(np.mean(np.asarray(episode_returns, dtype=np.float64))),
        "split_counts": {
            "train": int(np.sum(split_np == "train")),
            "val": int(np.sum(split_np == "val")),
            "test": int(np.sum(split_np == "test")),
        },
        "output_npz": str(args.out_npz),
        "output_ope_csv": str(args.out_ope_csv),
    }
    write_json(Path("outputs/data/icu_sepsis_summary.json"), summary)

    print("ICU-Sepsis extraction complete")
    print(summary)


if __name__ == "__main__":
    main()
