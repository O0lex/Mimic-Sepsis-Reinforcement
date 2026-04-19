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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--model-arch", type=str, choices=["mlp", "dueling"], default="mlp")
    return parser.parse_args()


def _build_cql_config(d3rlpy_module, batch_size: int, alpha: float, gamma: float, model_arch: str):
    config_class = d3rlpy_module.algos.DiscreteCQLConfig
    arch_kwargs = {}
    if model_arch == "dueling":
        from src.train.dueling_q import DuelingQFunctionFactory

        arch_kwargs = {"q_func_factory": DuelingQFunctionFactory()}

    candidate_kwargs = [
        {"alpha": alpha, "batch_size": batch_size, "gamma": gamma, **arch_kwargs},
        {"conservative_weight": alpha, "batch_size": batch_size, "gamma": gamma, **arch_kwargs},
        {"alpha": alpha, "batch_size": batch_size, **arch_kwargs},
        {"conservative_weight": alpha, "batch_size": batch_size, **arch_kwargs},
        {"batch_size": batch_size, **arch_kwargs},
    ]
    for kwargs in candidate_kwargs:
        try:
            return config_class(**kwargs), kwargs
        except TypeError:
            continue
    raise RuntimeError("Could not create DiscreteCQLConfig with known parameter names.")


def _fit_algo(
    algo,
    dataset,
    n_steps: int,
    eval_interval: int,
    experiment_name: str,
    log_root: Path,
) -> dict:
    fit_variants = [
        {
            "dataset": dataset,
            "n_steps": n_steps,
            "n_steps_per_epoch": max(1, eval_interval),
            "show_progress": True,
            "experiment_name": experiment_name,
            "with_timestamp": False,
            "logdir": str(log_root),
        },
        {
            "dataset": dataset,
            "n_epochs": max(1, n_steps // max(1, eval_interval)),
            "show_progress": True,
            "experiment_name": experiment_name,
            "with_timestamp": False,
            "logdir": str(log_root),
        },
        {"dataset": dataset, "n_steps": n_steps, "n_steps_per_epoch": max(1, eval_interval), "show_progress": True},
        {"dataset": dataset, "n_epochs": max(1, n_steps // max(1, eval_interval)), "show_progress": True},
    ]
    for kwargs in fit_variants:
        try:
            algo.fit(**kwargs)
            return kwargs
        except TypeError:
            continue
    raise RuntimeError("Unable to call algo.fit with known d3rlpy signatures.")


def _make_jsonable_kwargs(kwargs: dict) -> dict:
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _find_new_log_dir(before: set[Path], pattern: str, log_root: Path) -> Path | None:
    if not log_root.exists():
        return None
    after = {p for p in log_root.glob(pattern) if p.is_dir()}
    created = sorted((after - before), key=lambda p: p.stat().st_mtime)
    if created:
        return created[-1]
    candidates = sorted(after, key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _find_log_dir_by_experiment_name(log_roots: list[Path], experiment_name: str) -> Path | None:
    candidates: list[Path] = []
    for root in log_roots:
        if not root.exists():
            continue
        matches = [p for p in root.glob(f"*{experiment_name}*") if p.is_dir()]
        candidates.extend(matches)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


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

    config, used_kwargs = _build_cql_config(
        d3rlpy,
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        model_arch=args.model_arch,
    )
    algo = config.create(device=args.device)

    run_dir = args.out_dir / f"alpha_{args.alpha:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic naming avoids race conditions across fast alpha/step sweeps.
    experiment_name = (
        f"CQL_{args.model_arch}_alpha_{args.alpha:g}_gamma_{args.gamma:g}_steps_{args.n_steps}_seed_{args.seed}"
    )
    run_log_root = run_dir / "d3rlpy_logs"
    run_log_root.mkdir(parents=True, exist_ok=True)

    global_log_root = Path("d3rlpy_logs")
    before_run_logs = {p for p in run_log_root.glob("*CQL*") if p.is_dir()} if run_log_root.exists() else set()
    before_global_logs = {p for p in global_log_root.glob("*CQL*") if p.is_dir()} if global_log_root.exists() else set()

    fit_kwargs_used = _fit_algo(
        algo,
        dataset,
        n_steps=args.n_steps,
        eval_interval=args.eval_interval,
        experiment_name=experiment_name,
        log_root=run_log_root,
    )
    fit_kwargs_summary = {k: v for k, v in fit_kwargs_used.items() if k != "dataset"}
    config_kwargs_summary = _make_jsonable_kwargs(used_kwargs)

    log_dir = _find_new_log_dir(before=before_run_logs, pattern="*CQL*", log_root=run_log_root)
    if log_dir is None:
        log_dir = _find_new_log_dir(before=before_global_logs, pattern="*CQL*", log_root=global_log_root)
    if log_dir is None:
        log_dir = _find_log_dir_by_experiment_name(
            log_roots=[run_log_root, global_log_root],
            experiment_name=experiment_name,
        )

    model_path = run_dir / "cql_model.d3"
    # Save architecture + weights for robust reload across custom Q-function factories.
    algo.save(str(model_path))
    # Also keep a raw state_dict for debugging/legacy inspection.
    algo.save_model(str(run_dir / "cql_model_state.pt"))

    metrics = {
        "algorithm": "DiscreteCQL",
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "alpha_requested": args.alpha,
        "gamma_requested": args.gamma,
        "model_arch": args.model_arch,
        "config_kwargs_used": config_kwargs_summary,
        "fit_kwargs_used": fit_kwargs_summary,
        "experiment_name": experiment_name,
        "requested_log_root": str(run_log_root),
        "dataset_h5": str(args.dataset_h5),
        "dataset_npz": str(args.dataset_npz),
        "model_path": str(model_path),
        "log_dir": str(log_dir) if log_dir is not None else None,
    }
    write_json(run_dir / "train_metrics.json", metrics)
    print("CQL training complete")
    print(metrics)


if __name__ == "__main__":
    main()
