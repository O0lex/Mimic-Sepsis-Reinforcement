from __future__ import annotations

import argparse
import math
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib

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


def main() -> None:
    args = _parse_args()
    if not args.summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {args.summary_json}")
    if not args.wis_json.exists():
        raise FileNotFoundError(f"Missing WIS JSON: {args.wis_json}")

    summary = _read_json(args.summary_json)
    wis = _read_json(args.wis_json)

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Training objective curves (paper Figure 2 analogue).
    bc_metrics = _read_json(args.bc_metrics_json) if args.bc_metrics_json.exists() else None
    bc_log_dir = _resolve_log_dir(metrics=bc_metrics, pattern="DiscreteBC_*")

    cql_run_metrics: list[tuple[float, dict]] = []
    if args.cql_dir.exists():
        for mpath in sorted(args.cql_dir.glob("alpha_*/train_metrics.json")):
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
    action_metrics: dict[str, float] = {}
    bc_action_dist = None
    cql_action_dist = None
    clinician_action_dist = None

    if args.dataset_npz.exists() and args.bc_model.exists() and args.cql_model.exists():
        dataset, arrays = _build_d3_dataset(args.dataset_npz)
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

    # Plot 3: Feature importance (paper Figure 4 analogue) using RF on terminal outcome.
    feat_importance_fig = None
    top_features: list[dict[str, float]] = []
    if args.dataset_npz.exists():
        arrays = np.load(args.dataset_npz)
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
    behavior_v = float(wis.get("behavior_episode_return", 0.0))
    bc_v = float(wis.get("bc_wis_mean", 0.0))
    cql_v = float(wis.get("cql_wis_mean", 0.0))

    behavior_s = _policy_survival_proxy(behavior_v)
    bc_s = _policy_survival_proxy(bc_v)
    cql_s = _policy_survival_proxy(cql_v)

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
        "behavior_survival_proxy": behavior_s,
        "bc_survival_proxy": bc_s,
        "cql_survival_proxy": cql_s,
        "value_survival_corr": corr,
        "action_distribution_divergence": action_metrics,
        "top_features_rf_proxy": top_features,
        "figures": {
            "fig2_training_objective_curves": str(training_curves_fig) if training_curves_fig is not None else None,
            "fig4_feature_importance_rf_proxy": str(feat_importance_fig) if feat_importance_fig is not None else None,
            "fig5_action_distribution_comparison": str(action_dist_fig) if action_dist_fig is not None else None,
            "fig6_return_survival_relationship": str(return_survival_fig) if return_survival_fig is not None else None,
            "fig7_final_survival_comparison": str(survival_comparison_fig) if survival_comparison_fig is not None else None,
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
