from __future__ import annotations

import argparse
from pathlib import Path

import json

from src.common.io import ensure_parent, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report-ready metrics artifacts from pipeline outputs.")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/data/mimic_fhir_summary.json"))
    parser.add_argument("--wis-json", type=Path, default=Path("outputs/ope/mimic_fhir_wis_summary.json"))
    parser.add_argument("--out-json", type=Path, default=Path("report/generated/report_metrics.json"))
    parser.add_argument("--out-table", type=Path, default=Path("report/generated/results_table.tex"))
    parser.add_argument("--icu-summary-json", type=Path, default=Path("outputs/data/icu_sepsis_summary.json"))
    parser.add_argument("--icu-wis-json", type=Path, default=Path("outputs/ope/icu_sepsis_wis_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.summary_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {args.summary_json}")
    if not args.wis_json.exists():
        raise FileNotFoundError(f"Missing WIS JSON: {args.wis_json}")

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    wis = json.loads(args.wis_json.read_text(encoding="utf-8"))

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
    }

    if args.icu_summary_json.exists() and args.icu_wis_json.exists():
        icu_summary = json.loads(args.icu_summary_json.read_text(encoding="utf-8"))
        icu_wis = json.loads(args.icu_wis_json.read_text(encoding="utf-8"))
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
