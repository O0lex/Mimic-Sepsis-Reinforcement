from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.common.io import ensure_parent, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready metrics artifacts from pipeline outputs.")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/data/mimic_fhir_summary.json"))
    parser.add_argument("--wis-json", type=Path, default=Path("outputs/ope/mimic_fhir_wis_summary.json"))
    parser.add_argument("--ope-csv", type=Path, default=Path("outputs/data/mimic_fhir_ope_table.csv"))
    parser.add_argument("--dataset-npz", type=Path, default=Path("outputs/data/mimic_fhir_mdp_raw.npz"))
    parser.add_argument("--bc-metrics-json", type=Path, default=Path("outputs/models/bc/train_metrics.json"))
    parser.add_argument("--cql-dir", type=Path, default=Path("outputs/models/cql"))
    parser.add_argument("--bc-model", type=Path, default=Path("outputs/models/bc/bc_model.d3"))
    parser.add_argument("--cql-model", type=Path, default=Path("outputs/models/cql/alpha_1/cql_model.d3"))
    parser.add_argument("--fig-dir", type=Path, default=Path("report/generated/figures"))
    parser.add_argument("--cql-temperature", type=float, default=1.0)
    parser.add_argument("--prob-batch-size", type=int, default=2048)
    parser.add_argument("--out-json", type=Path, default=Path("report/generated/report_metrics.json"))
    parser.add_argument("--out-table", type=Path, default=Path("report/generated/results_table.tex"))
    parser.add_argument("--icu-summary-json", type=Path, default=Path("outputs/data/icu_sepsis_summary.json"))
    parser.add_argument("--icu-wis-json", type=Path, default=Path("outputs/ope/icu_sepsis_wis_summary.json"))
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_alpha_from_path(path: Path) -> float:
    token = path.parent.name
    if token.startswith("alpha_"):
        try:
            return float(token.replace("alpha_", ""))
        except ValueError:
            return 0.0
    return 0.0


def _find_latest_log_dir(pattern: str) -> Path | None:
    root = Path("d3rlpy_logs")
    if not root.exists():
        return None
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _resolve_log_dir(metrics: dict | None, pattern: str) -> Path | None:
    if metrics is not None:
        log_dir = metrics.get("log_dir")
        if log_dir:
            p = Path(log_dir)
            if p.exists():
                return p
    return _find_latest_log_dir(pattern)


def _read_metric_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"Unexpected metric curve format in {path}")
    df = df.iloc[:, :3].copy()
    df.columns = ["epoch", "step", "value"]
    return df


def _build_d3_dataset(npz_path: Path):
    from d3rlpy.dataset import MDPDataset

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


def _predict_bc_probs_all_actions(observations: np.ndarray, dataset, bc_model: Path, batch_size: int) -> np.ndarray:
    import d3rlpy
    import torch

    algo = d3rlpy.algos.DiscreteBCConfig().create(device="cpu")
    algo.build_with_dataset(dataset)
    algo.load_model(str(bc_model))

    modules = getattr(algo.impl, "modules", None)
    imitator = getattr(modules, "imitator", None) if modules is not None else None
    if imitator is None:
        imitator = getattr(algo.impl, "_imitator", None)
    if imitator is None:
        raise RuntimeError("Unable to find BC imitator module for action distribution plotting.")

    out = []
    for start in range(0, observations.shape[0], max(1, batch_size)):
        end = min(observations.shape[0], start + max(1, batch_size))
        x = torch.tensor(observations[start:end], dtype=torch.float32)
        with torch.no_grad():
            y = imitator(x)
        if hasattr(y, "probs"):
            probs = y.probs.detach().cpu().numpy()
        else:
            logits = y.detach().cpu().numpy()
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)
        out.append(probs)

    return np.concatenate(out, axis=0)


def _predict_cql_probs_all_actions(
    observations: np.ndarray,
    dataset,
    cql_model: Path,
    temperature: float,
    batch_size: int,
) -> np.ndarray:
    import d3rlpy
    import torch

    algo = d3rlpy.algos.DiscreteCQLConfig(alpha=1.0).create(device="cpu")
    algo.build_with_dataset(dataset)
    algo.load_model(str(cql_model))

    q_forwarder = getattr(algo.impl, "_q_func_forwarder", None)
    if q_forwarder is None or not hasattr(q_forwarder, "compute_expected_q"):
        raise RuntimeError("Unable to find CQL q forwarder for action distribution plotting.")

    temp = max(1e-6, float(temperature))
    out = []
    for start in range(0, observations.shape[0], max(1, batch_size)):
        end = min(observations.shape[0], start + max(1, batch_size))
        x = torch.tensor(observations[start:end], dtype=torch.float32)
        with torch.no_grad():
            q = q_forwarder.compute_expected_q(x)
            probs = torch.softmax(q / temp, dim=1).detach().cpu().numpy()
        out.append(probs)
    return np.concatenate(out, axis=0)


def _policy_survival_proxy(v: float) -> float:
    return float(np.clip((float(v) + 1.0) / 2.0, 0.0, 1.0))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(np.sum(p), 1e-12, None)
    q = q / np.clip(np.sum(q), 1e-12, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log(np.clip(p / np.clip(m, 1e-12, None), 1e-12, None)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log(np.clip(q / np.clip(m, 1e-12, None), 1e-12, None)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def _wis_from_logged_probs(df: pd.DataFrame, policy_prob_col: str, clip: float) -> float:
    grouped = df.groupby("episode_id", sort=False)
    episode_weights = []
    episode_returns = []

    for _, ep in grouped:
        pi = ep[policy_prob_col].to_numpy(dtype=np.float64)
        mu = ep["mu_prob"].to_numpy(dtype=np.float64)
        rew = ep["reward"].to_numpy(dtype=np.float64)

        # Log-space product for numerical stability
        log_ratios = np.log(np.clip(pi, 1e-12, 1.0)) - np.log(np.clip(mu, 1e-8, 1.0))
        cum_weight = np.exp(np.sum(log_ratios))
        
        # Cumulative clipping as per latest correctness standards
        if clip is not None:
            cum_weight = np.clip(cum_weight, 0.0, float(clip))

        episode_weights.append(float(cum_weight))
        episode_returns.append(float(np.sum(rew)))

    weights = np.asarray(episode_weights, dtype=np.float64)
    returns = np.asarray(episode_returns, dtype=np.float64)
    denom = float(np.sum(weights))
    return float(np.sum(weights * returns) / denom) if denom > 0.0 else 0.0


def _plot_clinical_heatmaps(
    clinician_dist: np.ndarray,
    bc_dist: np.ndarray,
    cql_dist: np.ndarray,
    fig_dir: Path,
) -> dict[str, str]:
    # For this project the action mapping is fixed to 5x5 fluid/vasopressor bins.
    if clinician_dist.shape[0] != 25 or bc_dist.shape[0] != 25 or cql_dist.shape[0] != 25:
        return {}

    labels = ["None", "Low", "Med", "High", "Max"]
    policies = [
        (clinician_dist.reshape(5, 5), "Clinician Policy", "fig10a_heatmap_clinician.png"),
        (bc_dist.reshape(5, 5), "BC Policy", "fig10b_heatmap_bc.png"),
        (cql_dist.reshape(5, 5), "CQL Policy", "fig10c_heatmap_cql.png"),
    ]
    outputs: dict[str, str] = {}

    for grid, title, fname in policies:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(grid, cmap="YlGnBu", origin="lower", aspect="auto")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Vasopressor Dose", fontsize=10)
        ax.set_ylabel("IV Fluid Dose", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        cbar = plt.colorbar(im)
        cbar.set_label("Frequency")
        fig.tight_layout()
        out_path = fig_dir / fname
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        outputs[fname.replace(".png", "")] = str(out_path)

    return outputs


def _plot_mortality_vs_deviation(
    logged_actions: np.ndarray,
    cql_actions: np.ndarray,
    rewards: np.ndarray,
    episode_ids: np.ndarray,
    fig_dir: Path,
) -> tuple[Path | None, dict[str, float]]:
    if not (
        logged_actions.size
        and cql_actions.size
        and rewards.size
        and episode_ids.size
        and logged_actions.shape[0] == cql_actions.shape[0] == rewards.shape[0] == episode_ids.shape[0]
    ):
        return None, {}

    deviation = np.abs(logged_actions.astype(np.int64) - cql_actions.astype(np.int64)).astype(np.float64)
    frame = pd.DataFrame(
        {
            "episode_id": episode_ids.astype(np.int64),
            "deviation": deviation,
            "reward": rewards.astype(np.float64),
        }
    )

    ep = frame.groupby("episode_id", sort=False).agg(
        mean_deviation=("deviation", "mean"),
        episode_return=("reward", "sum"),
    )
    ep["mortality"] = (ep["episode_return"] < 0.0).astype(np.float64)
    if ep.shape[0] < 10 or np.unique(ep["mean_deviation"].to_numpy()).shape[0] < 2:
        return None, {}

    q = min(5, int(np.unique(ep["mean_deviation"].to_numpy()).shape[0]))
    bins = pd.qcut(ep["mean_deviation"], q=q, duplicates="drop")
    binned = (
        ep.assign(bin=bins)
        .groupby("bin", observed=False)
        .agg(
            mean_deviation=("mean_deviation", "mean"),
            mortality_rate=("mortality", "mean"),
            n=("mortality", "size"),
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(
        binned["mean_deviation"],
        binned["mortality_rate"],
        marker="o",
        color="black",
        linewidth=1.8,
        markersize=5,
    )
    ax.fill_between(
        binned["mean_deviation"],
        0.0,
        binned["mortality_rate"],
        color="0.2",
        alpha=0.12,
    )
    ax.set_title("Mortality vs Deviation from CQL Policy")
    ax.set_xlabel("Mean |Clinician Action - CQL Action| per Episode")
    ax.set_ylabel("Mortality Rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = fig_dir / "fig8_mortality_deviation.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    corr = float(np.corrcoef(binned["mean_deviation"], binned["mortality_rate"])[0, 1])
    return out_path, {
        "mortality_deviation_corr": corr,
        "n_bins": float(binned.shape[0]),
    }


def _build_alpha_sensitivity(
    args: argparse.Namespace,
    dataset,
    observations: np.ndarray,
    df_eval: pd.DataFrame,
    actions: np.ndarray,
    behavior_v: float,
    wis_clip: float,
) -> tuple[Path | None, list[dict[str, float]]]:
    alpha_points: list[dict[str, float]] = []
    model_paths = sorted(args.cql_dir.glob("alpha_*/cql_model.d3"), key=_parse_alpha_from_path)
    if not model_paths:
        return None, alpha_points

    for mpath in model_paths:
        alpha = _parse_alpha_from_path(mpath)
        try:
            probs_all = _predict_cql_probs_all_actions(
                observations=observations,
                dataset=dataset,
                cql_model=mpath,
                temperature=args.cql_temperature,
                batch_size=args.prob_batch_size,
            )
        except Exception as exc:
            print(f"Warning: skipping alpha={alpha:g} sensitivity point due to model eval error: {exc}")
            continue
        sel = np.clip(actions.astype(np.int64), 0, probs_all.shape[1] - 1)
        logged_probs = probs_all[np.arange(probs_all.shape[0]), sel]

        df_tmp = df_eval.copy()
        n = min(df_tmp.shape[0], logged_probs.shape[0])
        if n <= 0:
            continue
        if n != df_tmp.shape[0]:
            print(
                "Warning: alpha sensitivity row mismatch; "
                f"using first {n} rows (df={df_tmp.shape[0]}, probs={logged_probs.shape[0]})."
            )
        df_tmp = df_tmp.iloc[:n].copy()
        df_tmp["alpha_eval_prob"] = logged_probs[:n]
        alpha_wis = _wis_from_logged_probs(df_tmp, policy_prob_col="alpha_eval_prob", clip=wis_clip)
        alpha_points.append({"alpha": float(alpha), "wis_return": float(alpha_wis)})

    if not alpha_points:
        return None, alpha_points

    alpha_points = sorted(alpha_points, key=lambda x: x["alpha"])
    x = [p["alpha"] for p in alpha_points]
    y = [p["wis_return"] for p in alpha_points]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(x, y, marker="o", linestyle="-", color="black", linewidth=1.8, label="CQL")
    ax.axhline(y=behavior_v, color="0.35", linestyle="--", linewidth=1.4, label="Clinician baseline")
    ax.set_title("Hyperparameter Sensitivity (CQL Alpha)")
    ax.set_xlabel("Conservative Penalty (Alpha)")
    ax.set_ylabel("WIS Return")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = args.fig_dir / "fig9_alpha_sensitivity.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return out_path, alpha_points


def main() -> None:
    args = _parse_args()
    if not args.summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {args.summary_json}")
    if not args.wis_json.exists():
        raise FileNotFoundError(f"Missing WIS JSON: {args.wis_json}")

    summary = _read_json(args.summary_json)
    wis = _read_json(args.wis_json)

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # Baseline values are needed by multiple figures.
    behavior_v = float(wis.get("behavior_episode_return", 0.0))
    bc_v = float(wis.get("bc_wis_mean", 0.0))
    cql_v = float(wis.get("cql_wis_mean", 0.0))
    behavior_s = _policy_survival_proxy(behavior_v)
    bc_s = _policy_survival_proxy(bc_v)
    cql_s = _policy_survival_proxy(cql_v)

    # Reward verification metadata for reporting fidelity.
    reward_unique_values: list[float] = []
    reward_is_terminal_binary = False

    dataset = None
    arrays = None
    if args.dataset_npz.exists():
        dataset, arrays = _build_d3_dataset(args.dataset_npz)
        reward_unique_values = [float(x) for x in np.unique(arrays["rewards"]).tolist()]
        reward_is_terminal_binary = set(np.round(np.asarray(reward_unique_values), 6).tolist()).issubset({-1.0, 0.0, 1.0})

    # Plot 1: Training objective curves (paper Figure 2 analogue).
    bc_metrics = _read_json(args.bc_metrics_json) if args.bc_metrics_json.exists() else None
    bc_log_dir = _resolve_log_dir(metrics=bc_metrics, pattern="DiscreteBC_*")

    cql_run_metrics: list[tuple[float, dict]] = []
    if args.cql_dir.exists():
        metrics_paths = sorted(args.cql_dir.glob("alpha_*/train_metrics.json"))
        if not metrics_paths:
            metrics_paths = sorted(args.cql_dir.glob("alpha_**/**/train_metrics.json"))
        for mpath in metrics_paths:
            m = _read_json(mpath)
            cql_run_metrics.append((float(m.get("alpha_requested", 0.0)), m))
    cql_run_metrics.sort(key=lambda x: x[0])

    training_curves_fig = None
    if bc_log_dir is not None or cql_run_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        # BC objective curve.
        if bc_log_dir is not None:
            bc_im_path = bc_log_dir / "imitation_loss.csv"
            bc_loss_path = bc_log_dir / "loss.csv"
            bc_curve_path = bc_im_path if bc_im_path.exists() else bc_loss_path
            if bc_curve_path.exists():
                bc_df = _read_metric_curve(bc_curve_path)
                axes[0].plot(bc_df["step"], bc_df["value"], color="#1f77b4", linewidth=2.0)
                axes[0].set_title("BC Training Objective")
                axes[0].set_xlabel("Step")
                axes[0].set_ylabel("Imitation Loss")
                axes[0].grid(alpha=0.3)
            else:
                axes[0].set_title("BC Training Objective (Unavailable)")
                axes[0].axis("off")
        else:
            axes[0].set_title("BC Training Objective (Unavailable)")
            axes[0].axis("off")

        # CQL objective curves by alpha.
        plotted_cql = False
        for alpha, metrics in cql_run_metrics:
            cql_log_dir = _resolve_log_dir(metrics=metrics, pattern="DiscreteCQL_*")
            if cql_log_dir is None:
                continue
            td_path = cql_log_dir / "td_loss.csv"
            total_path = cql_log_dir / "loss.csv"
            curve_path = td_path if td_path.exists() else total_path
            if not curve_path.exists():
                continue
            cql_df = _read_metric_curve(curve_path)
            axes[1].plot(cql_df["step"], cql_df["value"], linewidth=1.8, label=f"alpha={alpha:g}")
            plotted_cql = True

        if plotted_cql:
            axes[1].set_title("CQL Training Objective by Alpha")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("TD/Loss")
            axes[1].grid(alpha=0.3)
            axes[1].legend(frameon=False, fontsize=8)
        else:
            axes[1].set_title("CQL Training Objective (Unavailable)")
            axes[1].axis("off")

        fig.tight_layout()
        training_curves_fig = args.fig_dir / "fig2_training_objective_curves.png"
        fig.savefig(training_curves_fig, dpi=160)
        plt.close(fig)

    # Plot 2: Action distribution comparison (paper Figure 5 analogue).
    action_dist_fig = None
    mortality_deviation_fig = None
    alpha_sensitivity_fig = None
    heatmap_figs: dict[str, str] = {}
    action_metrics: dict[str, float] = {}
    alpha_sensitivity_points: list[dict[str, float]] = []
    model_eval_warning: str | None = None
    bc_action_dist = None
    cql_action_dist = None
    clinician_action_dist = None
    cql_greedy_actions = None

    if arrays is not None and args.bc_model.exists() and args.cql_model.exists():
        try:
            actions = arrays["actions"].astype(np.int64)
            n_actions = int(actions.max() + 1) if actions.size else 1

            clinician_action_dist = np.bincount(actions, minlength=n_actions).astype(np.float64)
            clinician_action_dist /= np.clip(np.sum(clinician_action_dist), 1e-12, None)

            observations = arrays["observations"]
            bc_probs = _predict_bc_probs_all_actions(
                observations=observations,
                dataset=dataset,
                bc_model=args.bc_model,
                batch_size=args.prob_batch_size,
            )
            cql_probs = _predict_cql_probs_all_actions(
                observations=observations,
                dataset=dataset,
                cql_model=args.cql_model,
                temperature=args.cql_temperature,
                batch_size=args.prob_batch_size,
            )

            bc_action_dist = np.mean(bc_probs, axis=0)
            cql_action_dist = np.mean(cql_probs, axis=0)
            cql_greedy_actions = np.argmax(cql_probs, axis=1).astype(np.int64)

            # Align action-space sizes if model/action artifacts differ.
            max_a = max(clinician_action_dist.shape[0], bc_action_dist.shape[0], cql_action_dist.shape[0])
            clinician_action_dist = np.pad(clinician_action_dist, (0, max_a - clinician_action_dist.shape[0]))
            bc_action_dist = np.pad(bc_action_dist, (0, max_a - bc_action_dist.shape[0]))
            cql_action_dist = np.pad(cql_action_dist, (0, max_a - cql_action_dist.shape[0]))

            x = np.arange(max_a)
            width = 0.28
            fig, ax = plt.subplots(figsize=(max(10, max_a * 0.35), 4.8))
            ax.bar(x - width, clinician_action_dist, width=width, label="Clinician (logged)", alpha=0.9)
            ax.bar(x, bc_action_dist, width=width, label="BC policy", alpha=0.85)
            ax.bar(x + width, cql_action_dist, width=width, label="CQL policy", alpha=0.85)
            ax.set_title("Action Distribution on Evaluation States")
            ax.set_xlabel("Discrete Action ID")
            ax.set_ylabel("Probability")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(frameon=False)
            fig.tight_layout()
            action_dist_fig = args.fig_dir / "fig5_action_distribution_comparison.png"
            fig.savefig(action_dist_fig, dpi=160)
            plt.close(fig)

            action_metrics = {
                "js_clinician_vs_bc": _js_divergence(clinician_action_dist, bc_action_dist),
                "js_clinician_vs_cql": _js_divergence(clinician_action_dist, cql_action_dist),
                "js_bc_vs_cql": _js_divergence(bc_action_dist, cql_action_dist),
            }

            heatmap_figs = _plot_clinical_heatmaps(
                clinician_dist=clinician_action_dist,
                bc_dist=bc_action_dist,
                cql_dist=cql_action_dist,
                fig_dir=args.fig_dir,
            )

            if args.ope_csv.exists():
                ope_df_full = pd.read_csv(args.ope_csv)
                if ope_df_full.shape[0] == arrays["actions"].shape[0]:
                    eval_split = wis.get("eval_split", "all")
                    if "split" in ope_df_full.columns and eval_split != "all":
                        mask = ope_df_full["split"].astype(str) == str(eval_split)
                    else:
                        mask = np.ones((ope_df_full.shape[0],), dtype=bool)

                    actions_logged = (
                        ope_df_full["action"].to_numpy(dtype=np.int64)
                        if "action" in ope_df_full.columns
                        else arrays["actions"].astype(np.int64)
                    )
                    rewards_eval = (
                        ope_df_full["reward"].to_numpy(dtype=np.float64)
                        if "reward" in ope_df_full.columns
                        else arrays["rewards"].astype(np.float64)
                    )
                    episode_ids_eval = (
                        ope_df_full["episode_id"].to_numpy(dtype=np.int64)
                        if "episode_id" in ope_df_full.columns
                        else arrays["episode_ids"].astype(np.int64)
                    )

                    mortality_deviation_fig, mortality_metrics = _plot_mortality_vs_deviation(
                        logged_actions=actions_logged[mask],
                        cql_actions=cql_greedy_actions[mask],
                        rewards=rewards_eval[mask],
                        episode_ids=episode_ids_eval[mask],
                        fig_dir=args.fig_dir,
                    )
                    action_metrics.update(mortality_metrics)

                    if set(["episode_id", "reward", "mu_prob"]).issubset(set(ope_df_full.columns)):
                        df_alpha = ope_df_full.copy()
                        if "split" in df_alpha.columns and eval_split != "all":
                            df_alpha = df_alpha[df_alpha["split"].astype(str) == str(eval_split)].copy()
                            alpha_mask = mask
                        else:
                            alpha_mask = np.ones((ope_df_full.shape[0],), dtype=bool)

                        wis_clip = float(wis.get("clip", 20.0))
                        alpha_sensitivity_fig, alpha_sensitivity_points = _build_alpha_sensitivity(
                            args=args,
                            dataset=dataset,
                            observations=arrays["observations"][alpha_mask],
                            df_eval=df_alpha,
                            actions=actions_logged[alpha_mask],
                            behavior_v=behavior_v,
                            wis_clip=wis_clip,
                        )
        except Exception as exc:
            model_eval_warning = str(exc)
            print(
                "Warning: skipping model-dependent figures (fig5/fig8/fig9/fig10*) due to "
                f"model-dataset mismatch or load error: {exc}"
            )

    # Plot 3: Feature importance (paper Figure 4 analogue) using RF on terminal outcome.
    feat_importance_fig = None
    top_features: list[dict[str, float]] = []
    if arrays is not None:
        observations = arrays["observations"]
        episode_ids = arrays["episode_ids"].astype(np.int64)
        rewards = arrays["rewards"].astype(np.float64)

        end_mask = np.r_[episode_ids[1:] != episode_ids[:-1], True]
        last_idx = np.where(end_mask)[0]
        ep_returns = pd.DataFrame({"episode_id": episode_ids, "reward": rewards}).groupby("episode_id")[
            "reward"
        ].sum()
        y = (ep_returns.to_numpy() > 0).astype(int)

        if last_idx.shape[0] == y.shape[0] and np.unique(y).shape[0] >= 2:
            X = observations[last_idx]
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
            rf.fit(X, y)

            names = summary.get("top_feature_codes") or [f"f{i}" for i in range(X.shape[1])]
            if len(names) < X.shape[1]:
                names = names + [f"f{i}" for i in range(len(names), X.shape[1])]
            names = names[: X.shape[1]]

            importances = rf.feature_importances_
            order = np.argsort(importances)[::-1][: min(20, importances.shape[0])]

            top_features = [{"feature": str(names[i]), "importance": float(importances[i])} for i in order.tolist()]

            fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.33 * len(order))))
            y_pos = np.arange(len(order))
            vals = importances[order][::-1]
            labels = [str(names[i]) for i in order[::-1]]
            ax.barh(y_pos, vals, color="#2ca02c", alpha=0.85)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Random Forest Importance")
            ax.set_title("Top Clinical Feature Importance (Proxy)")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            feat_importance_fig = args.fig_dir / "fig4_feature_importance_rf_proxy.png"
            fig.savefig(feat_importance_fig, dpi=160)
            plt.close(fig)

    # Plot 4/5: Return-survival relationship and final survival comparison (paper Figure 6/7 analogues).
    return_survival_fig = None
    survival_comparison_fig = None

    xs = np.asarray([behavior_v, bc_v, cql_v], dtype=np.float64)
    ys = np.asarray([behavior_s, bc_s, cql_s], dtype=np.float64)
    corr = float(np.corrcoef(xs, ys)[0, 1]) if np.unique(xs).shape[0] > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(xs, ys, s=80, color=["#7f7f7f", "#1f77b4", "#d62728"])
    for name, x, y in zip(["Behavior", "BC", "CQL"], xs, ys):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 6), fontsize=9)
    if np.unique(xs).shape[0] > 1:
        coeff = np.polyfit(xs, ys, deg=1)
        xx = np.linspace(float(xs.min()), float(xs.max()), 40)
        yy = coeff[0] * xx + coeff[1]
        ax.plot(xx, yy, linestyle="--", linewidth=1.6, alpha=0.8)
    ax.set_title("Policy Value vs Survival Proxy")
    ax.set_xlabel("Expected Return")
    ax.set_ylabel("Survival Proxy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return_survival_fig = args.fig_dir / "fig6_return_survival_relationship.png"
    fig.savefig(return_survival_fig, dpi=160)
    plt.close(fig)

    bc_ci = wis.get("bc_wis_ci95", [bc_v, bc_v])
    cql_ci = wis.get("cql_wis_ci95", [cql_v, cql_v])
    surv_vals = np.asarray([behavior_s, bc_s, cql_s], dtype=np.float64)
    err_low = np.asarray(
        [
            0.0,
            max(0.0, behavior_s - _policy_survival_proxy(float(bc_ci[0]))),
            max(0.0, cql_s - _policy_survival_proxy(float(cql_ci[0]))),
        ],
        dtype=np.float64,
    )
    err_hi = np.asarray(
        [
            0.0,
            max(0.0, _policy_survival_proxy(float(bc_ci[1])) - bc_s),
            max(0.0, _policy_survival_proxy(float(cql_ci[1])) - cql_s),
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.bar(["Behavior", "BC", "CQL"], surv_vals, yerr=np.vstack([err_low, err_hi]), capsize=4)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Survival Proxy")
    ax.set_title("Final Survival Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    survival_comparison_fig = args.fig_dir / "fig7_final_survival_comparison.png"
    fig.savefig(survival_comparison_fig, dpi=160)
    plt.close(fig)

    merged = {
        "cohort_icu_encounters": summary.get("cohort_icu_encounters"),
        "cohort_sepsis_patients": summary.get("cohort_sepsis_patients"),
        "n_transitions": summary.get("n_transitions"),
        "state_dim": summary.get("state_dim"),
        "split_counts": summary.get("split_counts"),
        "behavior_episode_return": wis.get("behavior_episode_return"),
        "bc_wis_mean": wis.get("bc_wis_mean"),
        "bc_wis_ci95": wis.get("bc_wis_ci95"),
        "cql_wis_mean": wis.get("cql_wis_mean"),
        "cql_wis_ci95": wis.get("cql_wis_ci95"),
        "eval_split": wis.get("eval_split", "all"),
        "n_episodes_eval": wis.get("n_episodes_eval"),
        "reward_unique_values": reward_unique_values,
        "reward_is_terminal_binary": reward_is_terminal_binary,
        "survival_label_is_rigorous": reward_is_terminal_binary,
        "action_mapping": {
            "formula": "action = fluid_bin * 5 + vaso_bin",
            "decode": {
                "fluid_bin": "action // 5",
                "vaso_bin": "action % 5",
            },
            "fluid_bins_ml": ["0", "1-50", "51-500", "501-1000", ">1000"],
            "vaso_bins_mcgkgmin": ["0", "0.001-0.05", "0.051-0.20", "0.201-0.45", ">0.45"],
        },
        "behavior_survival_proxy": behavior_s,
        "bc_survival_proxy": bc_s,
        "cql_survival_proxy": cql_s,
        "value_survival_corr": corr,
        "action_distribution_divergence": action_metrics,
        "model_eval_warning": model_eval_warning,
        "alpha_sensitivity_points": alpha_sensitivity_points,
        "top_features_rf_proxy": top_features,
        "figures": {
            "fig2_training_objective_curves": str(training_curves_fig) if training_curves_fig is not None else None,
            "fig4_feature_importance_rf_proxy": str(feat_importance_fig) if feat_importance_fig is not None else None,
            "fig5_action_distribution_comparison": str(action_dist_fig) if action_dist_fig is not None else None,
            "fig6_return_survival_relationship": str(return_survival_fig) if return_survival_fig is not None else None,
            "fig7_final_survival_comparison": str(survival_comparison_fig) if survival_comparison_fig is not None else None,
            "fig8_mortality_deviation": str(mortality_deviation_fig) if mortality_deviation_fig is not None else None,
            "fig9_alpha_sensitivity": str(alpha_sensitivity_fig) if alpha_sensitivity_fig is not None else None,
            **heatmap_figs,
        },
        "recommended_additional_metrics": [
            "ESS (effective sample size) per policy",
            "Off-support action rate",
            "Binning-interval ablation (1/2/4/6/8h)",
            "Subgroup policy value by lactate/shock severity",
        ],
    }

    if (
        args.icu_summary_json.exists()
        and args.icu_wis_json.exists()
        and args.icu_summary_json.resolve() != args.summary_json.resolve()
    ):
        icu_summary = _read_json(args.icu_summary_json)
        icu_wis = _read_json(args.icu_wis_json)
        merged["icu_sepsis"] = {
            "n_episodes": icu_summary.get("n_episodes"),
            "n_transitions": icu_summary.get("n_transitions"),
            "state_dim": icu_summary.get("state_dim"),
            "avg_episode_return": icu_summary.get("avg_episode_return"),
            "behavior_episode_return": icu_wis.get("behavior_episode_return"),
            "bc_wis_mean": icu_wis.get("bc_wis_mean"),
            "bc_wis_ci95": icu_wis.get("bc_wis_ci95"),
            "cql_wis_mean": icu_wis.get("cql_wis_mean"),
            "cql_wis_ci95": icu_wis.get("cql_wis_ci95"),
            "eval_split": icu_wis.get("eval_split", "all"),
            "n_episodes_eval": icu_wis.get("n_episodes_eval"),
        }
    write_json(args.out_json, merged)

    table = "\n".join(
        [
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Policy & Mean Return & 95\\% CI Low & 95\\% CI High \\\\",
            "\\midrule",
            f"Behavior ({wis.get('eval_split', 'all')} split) & {wis['behavior_episode_return']:.3f} & -- & -- \\\\",
            f"BC & {wis['bc_wis_mean']:.3f} & {wis['bc_wis_ci95'][0]:.3f} & {wis['bc_wis_ci95'][1]:.3f} \\\\",
            f"CQL & {wis['cql_wis_mean']:.3f} & {wis['cql_wis_ci95'][0]:.3f} & {wis['cql_wis_ci95'][1]:.3f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    ensure_parent(args.out_table)
    args.out_table.write_text(table, encoding="utf-8")

    print("Report assets generated")
    print({"out_json": str(args.out_json), "out_table": str(args.out_table)})


if __name__ == "__main__":
    main()
