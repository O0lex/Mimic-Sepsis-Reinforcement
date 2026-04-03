from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.io import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a d3rlpy-compatible MDPDataset from raw arrays.")
    parser.add_argument("--in-npz", type=Path, default=Path("outputs/data/mock_mdp_raw.npz"))
    parser.add_argument("--out-h5", type=Path, default=Path("outputs/data/mock_mdp_dataset.h5"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.in_npz.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.in_npz}")

    if args.out_h5.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.out_h5}. Use --force to overwrite.")

    arrays = np.load(args.in_npz)
    observations = arrays["observations"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    terminals = arrays["terminals"]
    episode_terminals = arrays["episode_terminals"]
    timeouts = np.zeros_like(terminals, dtype=np.float32)

    try:
        from d3rlpy.dataset import MDPDataset
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required to export MDPDataset. Install with pip install d3rlpy") from exc

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)

    # d3rlpy changed dataset constructor fields across versions.
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
            episode_terminals=episode_terminals,
        )
    dataset.dump(str(args.out_h5))

    summary = {
        "in_npz": str(args.in_npz),
        "out_h5": str(args.out_h5),
        "n_transitions": int(observations.shape[0]),
        "state_dim": int(observations.shape[1]),
        "n_actions_observed": int(np.unique(actions).shape[0]),
        "mean_reward": float(np.mean(rewards)),
    }
    write_json(Path("outputs/data/mdp_dataset_summary.json"), summary)
    print("MDPDataset exported:")
    print(summary)


if __name__ == "__main__":
    main()
