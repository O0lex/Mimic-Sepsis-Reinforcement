from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from src.common.io import write_json
from src.common.seed import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Behavior Cloning baseline with d3rlpy.")
    parser.add_argument("--dataset-h5", type=Path, default=Path("outputs/data/mock_mdp_dataset.h5"))
    parser.add_argument("--dataset-npz", type=Path, default=Path("outputs/data/mock_mdp_raw.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/models/bc"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=os.getenv("D3RLPY_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    return parser.parse_args()


def _fit_algo(algo, dataset, n_steps: int, eval_interval: int) -> None:
    # Handle API differences across d3rlpy versions.
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


def _find_new_log_dir(before: set[Path], pattern: str) -> Path | None:
    log_root = Path("d3rlpy_logs")
    if not log_root.exists():
        return None
    after = set(log_root.glob(pattern))
    created = sorted((after - before), key=lambda p: p.stat().st_mtime)
    if created:
        return created[-1]
    candidates = sorted(after, key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


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


def _resolve_device(requested_device: str) -> str:
    raw = str(requested_device or "cpu").strip().lower()
    if raw.startswith("cuda"):
        try:
            import torch
        except ImportError:
            print("Warning: requested CUDA device but torch is unavailable; falling back to CPU.")
            return "cpu"

        if not torch.cuda.is_available():
            print(f"Warning: requested device '{requested_device}' but CUDA is unavailable; falling back to CPU.")
            return "cpu"

    return raw


def main() -> None:
    args = _parse_args()

    set_seed(args.seed)

    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("d3rlpy is required. Install with pip install d3rlpy") from exc

    dataset = _load_dataset(args.dataset_h5, args.dataset_npz)

    resolved_device = _resolve_device(args.device)

    config = d3rlpy.algos.DiscreteBCConfig(batch_size=args.batch_size)
    algo = config.create(device=resolved_device)

    log_root = Path("d3rlpy_logs")
    before_logs = set(log_root.glob("DiscreteBC_*")) if log_root.exists() else set()
    _fit_algo(algo, dataset, n_steps=args.n_steps, eval_interval=args.eval_interval)
    log_dir = _find_new_log_dir(before=before_logs, pattern="DiscreteBC_*")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "bc_model.d3"
    algo.save_model(str(model_path))

    metrics = {
        "algorithm": "DiscreteBC",
        "seed": args.seed,
        "device_requested": args.device,
        "device": resolved_device,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "dataset_h5": str(args.dataset_h5),
        "dataset_npz": str(args.dataset_npz),
        "model_path": str(model_path),
        "log_dir": str(log_dir) if log_dir is not None else None,
    }
    write_json(args.out_dir / "train_metrics.json", metrics)
    print("BC training complete")
    print(metrics)


if __name__ == "__main__":
    main()
