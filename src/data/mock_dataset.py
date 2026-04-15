from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import write_json
from src.common.seed import set_seed


@dataclass
class MockConfig:
    n_episodes: int
    min_horizon: int
    max_horizon: int
    state_dim: int
    n_actions: int
    reward_scale: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_mock_trajectories(cfg: MockConfig, seed: int) -> dict[str, np.ndarray]:
    set_seed(seed)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[float] = []
    episode_terminals: list[float] = []
    episode_ids: list[int] = []
    behavior_probs: list[float] = []
    bc_probs: list[float] = []
    cql_probs: list[float] = []

    for ep in range(cfg.n_episodes):
        horizon = np.random.randint(cfg.min_horizon, cfg.max_horizon + 1)

        # Latent severity evolves as random walk.
        severity = np.random.normal(0.0, 0.8)
        for t in range(horizon):
            severity += np.random.normal(0.0, 0.2)
            state = np.random.normal(0.0, 1.0, size=(cfg.state_dim,)).astype(np.float32)
            state[0] = severity

            # Logged behavior tends to higher actions for high severity.
            logits = np.linspace(-1.2, 1.2, cfg.n_actions) + 0.35 * severity
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(np.random.choice(cfg.n_actions, p=probs))

            # Reward favors action close to a smooth target policy with noise.
            target_action = int(np.clip((severity + 2.0) / 4.0 * (cfg.n_actions - 1), 0, cfg.n_actions - 1))
            reward = cfg.reward_scale * (1.0 - abs(action - target_action) / max(cfg.n_actions - 1, 1))
            reward += np.random.normal(0.0, 0.05)

            done = 1.0 if (t == horizon - 1) else 0.0

            observations.append(state)
            actions.append(action)
            rewards.append(float(reward))
            terminals.append(done)
            episode_terminals.append(done)
            episode_ids.append(ep)

            mu = float(np.clip(probs[action], 1e-5, 1.0))
            # Mock BC and CQL probabilities to exercise WIS pipeline.
            bc_prob = float(np.clip(0.8 * mu + 0.2 / cfg.n_actions, 1e-5, 1.0))
            cql_shift = _sigmoid(np.array([severity], dtype=np.float32))[0] - 0.5
            cql_prob = float(np.clip(mu + 0.2 * cql_shift, 1e-5, 1.0))

            behavior_probs.append(mu)
            bc_probs.append(bc_prob)
            cql_probs.append(cql_prob)

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
        "episode_terminals": np.asarray(episode_terminals, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "behavior_probs": np.asarray(behavior_probs, dtype=np.float32),
        "bc_probs": np.asarray(bc_probs, dtype=np.float32),
        "cql_probs": np.asarray(cql_probs, dtype=np.float32),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock sepsis-like offline trajectories.")
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/data/mock_mdp_raw.npz"))
    parser.add_argument("--out-ope-csv", type=Path, default=Path("outputs/data/mock_ope_table.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--min-horizon", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--n-actions", type=int, default=25)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = MockConfig(
        n_episodes=args.n_episodes,
        min_horizon=args.min_horizon,
        max_horizon=args.max_horizon,
        state_dim=args.state_dim,
        n_actions=args.n_actions,
        reward_scale=args.reward_scale,
    )

    data = build_mock_trajectories(cfg, args.seed)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **data)

    df = pd.DataFrame(
        {
            "episode_id": data["episode_ids"],
            "reward": data["rewards"],
            "action": data["actions"],
            "mu_prob": data["behavior_probs"],
            "bc_prob": data["bc_probs"],
            "cql_prob": data["cql_probs"],
        }
    )
    args.out_ope_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_ope_csv, index=False)

    summary = {
        "n_transitions": int(data["observations"].shape[0]),
        "n_episodes": int(cfg.n_episodes),
        "state_dim": int(cfg.state_dim),
        "n_actions": int(cfg.n_actions),
        "mean_reward": float(np.mean(data["rewards"])),
        "terminal_rate": float(np.mean(data["terminals"])),
        "out_npz": str(args.out_npz),
        "out_ope_csv": str(args.out_ope_csv),
    }
    write_json(Path("outputs/data/mock_summary.json"), summary)
    print("Mock dataset generated:")
    print(summary)


if __name__ == "__main__":
    main()
