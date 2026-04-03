from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.io import write_json
from src.common.seed import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Discrete CQL with d3rlpy.")
    parser.add_argument("--dataset-h5", type=Path, default=Path("outputs/data/mock_mdp_dataset.h5"))
    parser.add_argument("--dataset-npz", type=Path, default=Path("outputs/data/mock_mdp_raw.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/models/cql"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def _build_cql_config(d3rlpy_module, batch_size: int, alpha: float):
    config_class = d3rlpy_module.algos.DiscreteCQLConfig
    candidate_kwargs = [
        {"alpha": alpha, "batch_size": batch_size},
        {"conservative_weight": alpha, "batch_size": batch_size},
        {"batch_size": batch_size},
    ]
    for kwargs in candidate_kwargs:
        try:
            return config_class(**kwargs), kwargs
        except TypeError:
            continue
    raise RuntimeError("Could not create DiscreteCQLConfig with known parameter names.")


def _fit_algo(algo, dataset, n_steps: int, eval_interval: int) -> None:
    fit_variants = [
        {"dataset": dataset, "n_steps": n_steps, "n_steps_per_epoch": max(1, eval_interval), "show_progress": True},
        {"dataset": dataset, "n_epochs": max(1, n_steps // max(1, eval_interval)), "show_progress": True},
    ]
    for kwargs in fit_variants:
        try:
            algo.fit(**kwargs)
            return
        except TypeError:
            continue
    raise RuntimeError("Unable to call algo.fit with known d3rlpy signatures.")


def _load_dataset(h5_path: Path, npz_path: Path):
    from d3rlpy.dataset import InfiniteBuffer, MDPDataset, ReplayBuffer

    if h5_path.exists():
        try:
            with h5_path.open("rb") as f:
                return ReplayBuffer.load(f, buffer=InfiniteBuffer())
        except Exception as exc:
            print(f"Warning: failed to load {h5_path} via ReplayBuffer.load: {exc}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Neither dataset file exists: {h5_path} or {npz_path}")

    arrays = np.load(npz_path)
    observations = arrays["observations"]
    actions = arrays["actions"]
    rewards = arrays["rewards"]
    terminals = arrays["terminals"]
    timeouts = np.zeros_like(terminals, dtype=np.float32)
    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required. Install with pip install d3rlpy") from exc

    dataset = _load_dataset(args.dataset_h5, args.dataset_npz)

    config, used_kwargs = _build_cql_config(d3rlpy, batch_size=args.batch_size, alpha=args.alpha)
    algo = config.create(device=args.device)

    _fit_algo(algo, dataset, n_steps=args.n_steps, eval_interval=args.eval_interval)

    run_dir = args.out_dir / f"alpha_{args.alpha:g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "cql_model.d3"
    algo.save_model(str(model_path))

    metrics = {
        "algorithm": "DiscreteCQL",
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "alpha_requested": args.alpha,
        "config_kwargs_used": used_kwargs,
        "dataset_h5": str(args.dataset_h5),
        "dataset_npz": str(args.dataset_npz),
        "model_path": str(model_path),
    }
    write_json(run_dir / "train_metrics.json", metrics)
    print("CQL training complete")
    print(metrics)


if __name__ == "__main__":
    main()
