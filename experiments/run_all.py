from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for token in raw.split(","):
        s = token.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals


def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in raw.split(","):
        s = token.strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def _seeded_dir(base_dir: str, seed: int, multi_seed: bool) -> str:
    if not multi_seed:
        return base_dir
    return f"{base_dir}/seed_{seed}"


def _aggregate_seed_wis_jsons(seed_json_paths: list[Path], out_json: Path, seeds: list[int]) -> None:
    valid_paths = [p for p in seed_json_paths if p.exists()]
    if not valid_paths:
        raise FileNotFoundError(f"No seed WIS JSON files found for aggregation target: {out_json}")

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in valid_paths]

    def _collect(name: str) -> list[float]:
        vals: list[float] = []
        for r in rows:
            v = r.get(name)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    def _std(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        mu = _mean(xs)
        var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
        return float(math.sqrt(var))

    def _ci95_for_mean(xs: list[float]) -> list[float]:
        if not xs:
            return [0.0, 0.0]
        mu = _mean(xs)
        if len(xs) < 2:
            return [mu, mu]
        half = 1.96 * _std(xs) / math.sqrt(len(xs))
        return [float(mu - half), float(mu + half)]

    bc_means = _collect("bc_wis_mean")
    cql_means = _collect("cql_wis_mean")
    behavior_vals = _collect("behavior_episode_return")
    bc_ess_vals = _collect("bc_ess")
    cql_ess_vals = _collect("cql_ess")

    base = rows[0].copy()
    base["behavior_episode_return"] = _mean(behavior_vals) if behavior_vals else base.get("behavior_episode_return", 0.0)
    base["bc_wis_mean"] = _mean(bc_means) if bc_means else base.get("bc_wis_mean", 0.0)
    base["cql_wis_mean"] = _mean(cql_means) if cql_means else base.get("cql_wis_mean", 0.0)
    base["bc_wis_ci95"] = _ci95_for_mean(bc_means)
    base["cql_wis_ci95"] = _ci95_for_mean(cql_means)
    if bc_ess_vals:
        base["bc_ess"] = _mean(bc_ess_vals)
    if cql_ess_vals:
        base["cql_ess"] = _mean(cql_ess_vals)

    base["seed_aggregation"] = {
        "seeds_requested": [int(s) for s in seeds],
        "seeds_found": [int(s) for s in seeds if any(f"seed{s}" in p.name for p in valid_paths)],
        "n_seed_files": int(len(valid_paths)),
        "bc_wis_seed_std": _std(bc_means),
        "cql_wis_seed_std": _std(cql_means),
        "seed_json_paths": [str(p) for p in valid_paths],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(base, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end mock offline RL pipeline.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--use-fhir", action="store_true")
    parser.add_argument("--use-icu-sepsis", action="store_true")
    parser.add_argument("--use-bigquery", action="store_true")
    parser.add_argument("--use-mimic-profiles", action="store_true")
    parser.add_argument("--in-parquet", type=str, default="data/raw/mimic_sepsis_final.parquet")
    parser.add_argument("--reward-mode", type=str, choices=["terminal", "sofa_delta", "composite"], default="terminal")
    parser.add_argument("--terminal-reward-scale", type=float, default=1.0)
    parser.add_argument("--sofa-weight", type=float, default=0.1)
    parser.add_argument("--sofa-clip", type=float, default=1.0)
    parser.add_argument("--profile", type=str, choices=["minimal", "full", "both"], default="both")
    parser.add_argument(
        "--fhir-dir",
        type=str,
        default="data/mimic-iv-clinical-database-demo-on-fhir-2.0/mimic-fhir",
    )
    parser.add_argument("--icu-env-id", type=str, default="Sepsis/ICU-Sepsis-v2")
    parser.add_argument("--gcp-project-id", type=str, default="")
    parser.add_argument("--bq-location", type=str, default="US")
    parser.add_argument("--mimic-project", type=str, default="physionet-data")
    parser.add_argument("--derived-dataset", type=str, default="mimiciv_derived")
    parser.add_argument("--icu-dataset", type=str, default="mimiciv_icu")
    parser.add_argument("--hosp-dataset", type=str, default="mimiciv_hosp")
    parser.add_argument("--max-stays", type=int, default=500)
    parser.add_argument("--mu-source", type=str, choices=["auto", "csv", "bc_model"], default="auto")
    parser.add_argument(
        "--cql-alphas",
        type=str,
        default="0.25,0.5,0.75,1.0,1.5,2.0,2.5,5.0,7.5,10.0",
        help="Comma-separated alpha values for CQL sweep in mimic-profile mode.",
    )
    parser.add_argument(
        "--cql-step-sweep",
        type=str,
        default="3000,5000,10000,15000,50000,100000",
        help="Comma-separated n-steps values for CQL sweep in mimic-profile mode.",
    )
    parser.add_argument(
        "--cql-gammas",
        type=str,
        default="0.90,0.95,0.99",
        help="Comma-separated discount factors for CQL runs.",
    )
    parser.add_argument(
        "--cql-architectures",
        type=str,
        default="mlp",
        help="Comma-separated CQL model architectures to train (e.g. mlp,dueling).",
    )
    parser.add_argument(
        "--cql-long-alphas",
        type=str,
        default="1.0,2.5,5.0",
        help="Comma-separated alpha values that should additionally run at long horizons.",
    )
    parser.add_argument(
        "--cql-long-steps",
        type=str,
        default="50000,100000",
        help="Comma-separated long-run n-steps values for selected --cql-long-alphas.",
    )
    parser.add_argument(
        "--cql-eval-interval",
        type=int,
        default=10000,
        help="Training metric logging/report interval for CQL runs.",
    )
    parser.add_argument("--bc-steps", type=int, default=1000)
    parser.add_argument("--report-alpha", type=float, default=1.5)
    parser.add_argument("--report-steps", type=int, default=1000)
    parser.add_argument("--report-gamma", type=float, default=0.95)
    parser.add_argument("--report-arch", type=str, default="mlp")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds for BC/CQL/WIS runs. Results are aggregated across seeds.",
    )
    parser.add_argument(
        "--report-seed",
        type=int,
        default=None,
        help="Seed to use for report model-path selection. Defaults to first seed from --seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]

    mode_flags = [args.use_fhir, args.use_icu_sepsis, args.use_bigquery, args.use_mimic_profiles]
    if sum(1 for x in mode_flags if x) > 1:
        raise ValueError("Use only one of --use-fhir, --use-icu-sepsis, --use-bigquery, or --use-mimic-profiles.")

    if args.use_bigquery and not args.gcp_project_id:
        raise ValueError("--use-bigquery requires --gcp-project-id.")

    dataset_npz = "outputs/data/mock_mdp_raw.npz"
    dataset_h5 = "outputs/data/mock_mdp_dataset.h5"
    ope_csv = "outputs/data/mock_ope_table.csv"
    wis_json = "outputs/ope/wis_summary.json"
    summary_json = "outputs/data/mock_summary.json"
    aggregate_jobs: list[tuple[Path, list[Path], list[int]]] = []

    if args.use_mimic_profiles:
        selected_profiles = [args.profile] if args.profile in {"minimal", "full"} else ["minimal", "full"]
        seeds = sorted(set(_parse_int_list(args.seeds)))
        if not seeds:
            raise ValueError("--seeds resolved to an empty list.")
        multi_seed = len(seeds) > 1
        report_seed = args.report_seed if args.report_seed is not None else seeds[0]
        if report_seed not in seeds:
            raise ValueError(f"--report-seed ({report_seed}) must be one of --seeds ({seeds}).")
        steps = [
            [
                args.python,
                "-m",
                "src.data.preprocess",
                "--in-parquet",
                args.in_parquet,
                "--out-dir",
                "outputs/data",
                "--reward-mode",
                args.reward_mode,
                "--terminal-reward-scale",
                str(args.terminal_reward_scale),
                "--sofa-weight",
                str(args.sofa_weight),
                "--sofa-clip",
                str(args.sofa_clip),
            ]
        ]

        mu_source = args.mu_source
        if mu_source == "auto":
            mu_source = "bc_model" if not args.skip_train else "csv"

        for profile in selected_profiles:
            dataset_npz = f"outputs/data/mimic_mdp_{profile}_raw.npz"
            dataset_h5 = f"outputs/data/mimic_mdp_{profile}_dataset.h5"
            ope_csv = f"outputs/data/mimic_mdp_{profile}_ope_table.csv"
            wis_json = f"outputs/ope/mimic_mdp_{profile}_wis_summary.json"
            summary_json = "outputs/data/mimic_mdp_profiles_summary.json"

            bc_dir = f"outputs/models/{profile}/bc"
            cql_dir = f"outputs/models/{profile}/cql"
            refinement_alphas = _parse_float_list(args.cql_alphas)
            step_sweep = _parse_int_list(args.cql_step_sweep)
            long_alphas = _parse_float_list(args.cql_long_alphas)
            long_steps = _parse_int_list(args.cql_long_steps)
            gammas = _parse_float_list(args.cql_gammas)
            architectures = [s.strip() for s in args.cql_architectures.split(",") if s.strip()]
            if not refinement_alphas:
                raise ValueError("--cql-alphas resolved to an empty list.")
            if not step_sweep:
                raise ValueError("--cql-step-sweep resolved to an empty list.")
            if not long_alphas:
                raise ValueError("--cql-long-alphas resolved to an empty list.")
            if not long_steps:
                raise ValueError("--cql-long-steps resolved to an empty list.")
            if not gammas:
                raise ValueError("--cql-gammas resolved to an empty list.")
            if not architectures:
                raise ValueError("--cql-architectures resolved to an empty list.")
            for arch in architectures:
                if arch not in {"mlp", "dueling"}:
                    raise ValueError(f"Unsupported architecture '{arch}'. Expected one of: mlp, dueling.")
            if args.report_arch not in architectures:
                architectures.append(args.report_arch)

            if args.report_alpha not in refinement_alphas:
                refinement_alphas.append(args.report_alpha)
            if args.report_steps not in step_sweep:
                step_sweep.append(args.report_steps)
            if args.report_gamma not in gammas:
                gammas.append(args.report_gamma)
            if args.report_alpha not in long_alphas:
                long_alphas.append(args.report_alpha)
            if args.report_steps not in long_steps:
                long_steps.append(args.report_steps)

            refinement_alphas = sorted(set(refinement_alphas))
            step_sweep = sorted(set(step_sweep))
            long_alphas = sorted(set(long_alphas))
            long_steps = sorted(set(long_steps))
            gammas = sorted(set(gammas))

            base_pairs = {(int(s), float(a), float(g)) for s in step_sweep for a in refinement_alphas for g in gammas}
            training_pairs = set(base_pairs)
            training_pairs.update({(int(s), float(a), float(g)) for s in long_steps for a in long_alphas for g in gammas})

            report_alpha = args.report_alpha
            report_steps = args.report_steps
            report_gamma = args.report_gamma
            report_arch = args.report_arch

            def _float_tag(a: float) -> str:
                s = f"{a:g}"
                return s.replace(".", "_") if "." in s else s

            def _arch_root(base_dir: str, arch: str) -> str:
                return base_dir if arch == "mlp" else f"{base_dir}/arch_{arch}"

            report_bc_dir = _seeded_dir(bc_dir, report_seed, multi_seed)
            bc_model = f"{report_bc_dir}/bc_model.d3"

            report_cql_dir = _seeded_dir(_arch_root(cql_dir, report_arch), report_seed, multi_seed)
            cql_model = f"{report_cql_dir}/steps_{report_steps}/gamma_{report_gamma:g}/alpha_{report_alpha:g}/cql_model.d3"

            steps.append(
                [
                    args.python,
                    "-m",
                    "src.data.build_mdp_dataset",
                    "--in-npz",
                    dataset_npz,
                    "--out-h5",
                    dataset_h5,
                    "--force",
                ]
            )

            if not args.skip_train:
                for seed in seeds:
                    seed_bc_dir = _seeded_dir(bc_dir, seed, multi_seed)
                    steps.append(
                        [
                            args.python,
                            "-m",
                            "src.train.train_bc",
                            "--dataset-h5",
                            dataset_h5,
                            "--dataset-npz",
                            dataset_npz,
                            "--out-dir",
                            seed_bc_dir,
                            "--seed",
                            str(seed),
                            "--n-steps",
                            str(args.bc_steps),
                        ]
                    )
                for arch in architectures:
                    cql_arch_dir = _arch_root(cql_dir, arch)
                    for step_count in step_sweep:
                        for seed in seeds:
                            seed_cql_root = _seeded_dir(cql_arch_dir, seed, multi_seed)
                            cql_step_dir = f"{seed_cql_root}/steps_{step_count}"
                            for gamma in gammas:
                                cql_gamma_dir = f"{cql_step_dir}/gamma_{gamma:g}"
                                for alpha in refinement_alphas:
                                    steps.append(
                                        [
                                            args.python,
                                            "-m",
                                            "src.train.train_cql",
                                            "--dataset-h5",
                                            dataset_h5,
                                            "--dataset-npz",
                                            dataset_npz,
                                            "--out-dir",
                                            cql_gamma_dir,
                                            "--alpha",
                                            str(alpha),
                                            "--gamma",
                                            str(gamma),
                                            "--model-arch",
                                            arch,
                                            "--seed",
                                            str(seed),
                                            "--n-steps",
                                            str(step_count),
                                            "--eval-interval",
                                            str(args.cql_eval_interval),
                                        ]
                                    )
                # Add targeted long-horizon runs for selected alphas.
                for arch in architectures:
                    cql_arch_dir = _arch_root(cql_dir, arch)
                    for step_count in long_steps:
                        for seed in seeds:
                            seed_cql_root = _seeded_dir(cql_arch_dir, seed, multi_seed)
                            cql_step_dir = f"{seed_cql_root}/steps_{step_count}"
                            for gamma in gammas:
                                cql_gamma_dir = f"{cql_step_dir}/gamma_{gamma:g}"
                                for alpha in long_alphas:
                                    if (int(step_count), float(alpha), float(gamma)) in base_pairs:
                                        continue
                                    steps.append(
                                        [
                                            args.python,
                                            "-m",
                                            "src.train.train_cql",
                                            "--dataset-h5",
                                            dataset_h5,
                                            "--dataset-npz",
                                            dataset_npz,
                                            "--out-dir",
                                            cql_gamma_dir,
                                            "--alpha",
                                            str(alpha),
                                            "--gamma",
                                            str(gamma),
                                            "--model-arch",
                                            arch,
                                            "--seed",
                                            str(seed),
                                            "--n-steps",
                                            str(step_count),
                                            "--eval-interval",
                                            str(args.cql_eval_interval),
                                        ]
                                    )

            if mu_source == "bc_model":
                for step_count, alpha, gamma in sorted(training_pairs, key=lambda x: (x[0], x[2], x[1])):
                    alpha_tag = _float_tag(alpha)
                    gamma_tag = _float_tag(gamma)
                    canonical_wis_json = Path(
                        f"outputs/ope/mimic_mdp_{profile}_wis_s{step_count}_a{alpha_tag}_g{gamma_tag}.json"
                    )
                    seed_json_paths: list[Path] = []

                    for seed in seeds:
                        seed_bc_model = f"{_seeded_dir(bc_dir, seed, multi_seed)}/bc_model.d3"
                        seed_report_cql_dir = _seeded_dir(_arch_root(cql_dir, report_arch), seed, multi_seed)
                        model_path = f"{seed_report_cql_dir}/steps_{step_count}/gamma_{gamma:g}/alpha_{alpha:g}/cql_model.d3"
                        seed_wis_json = Path(
                            f"outputs/ope/mimic_mdp_{profile}_wis_s{step_count}_a{alpha_tag}_g{gamma_tag}_seed{seed}.json"
                        )
                        seed_json_paths.append(seed_wis_json)

                        steps.append(
                            [
                                args.python,
                                "-m",
                                "src.ope.wis_eval",
                                "--in-csv",
                                ope_csv,
                                "--out-json",
                                str(seed_wis_json),
                                "--mu-source",
                                "bc_model",
                                "--dataset-npz",
                                dataset_npz,
                                "--bc-model",
                                seed_bc_model,
                                "--cql-model",
                                model_path,
                                "--seed",
                                str(seed),
                                "--sync-bc-prob-with-mu",
                            ]
                        )

                    aggregate_jobs.append((canonical_wis_json, seed_json_paths, seeds))
                wis_json = (
                    f"outputs/ope/mimic_mdp_{profile}_wis_s{report_steps}_a{_float_tag(report_alpha)}_g{_float_tag(report_gamma)}.json"
                )
                cql_model = (
                    f"{report_cql_dir}/steps_{report_steps}/gamma_{report_gamma:g}/alpha_{report_alpha:g}/cql_model.d3"
                )
            else:
                wis_step = [
                    args.python,
                    "-m",
                    "src.ope.wis_eval",
                    "--in-csv",
                    ope_csv,
                    "--out-json",
                    wis_json,
                    "--out-csv",
                    ope_csv,
                    "--mu-source",
                    mu_source,
                ]
                steps.append(wis_step)

            steps.append(
                [
                    args.python,
                    "-m",
                    "src.ope.build_report_assets",
                    "--summary-json",
                    summary_json,
                    "--wis-json",
                    wis_json,
                    "--ope-csv",
                    ope_csv,
                    "--dataset-npz",
                    dataset_npz,
                    "--bc-metrics-json",
                    f"{report_bc_dir}/train_metrics.json",
                    "--cql-dir",
                    report_cql_dir,
                    "--bc-model",
                    bc_model,
                    "--cql-model",
                    cql_model,
                    "--out-json",
                    f"report/generated/{profile}_report_metrics.json",
                    "--out-table",
                    f"report/generated/{profile}_results_table.tex",
                    "--fig-dir",
                    f"report/generated/figures/{profile}",
                ]
            )
    elif args.use_icu_sepsis:
        dataset_npz = "outputs/data/icu_sepsis_mdp_raw.npz"
        dataset_h5 = "outputs/data/icu_sepsis_mdp_dataset.h5"
        ope_csv = "outputs/data/icu_sepsis_ope_table.csv"
        wis_json = "outputs/ope/icu_sepsis_wis_summary.json"
        summary_json = "outputs/data/icu_sepsis_summary.json"
        steps = [
            [
                args.python,
                "-m",
                "src.data.extract_icu_sepsis_dataset",
                "--env-id",
                args.icu_env_id,
                "--out-npz",
                dataset_npz,
                "--out-ope-csv",
                ope_csv,
            ],
            [
                args.python,
                "-m",
                "src.data.build_mdp_dataset",
                "--in-npz",
                dataset_npz,
                "--out-h5",
                dataset_h5,
                "--force",
            ],
        ]
    elif args.use_fhir:
        dataset_npz = "outputs/data/mimic_fhir_mdp_raw.npz"
        dataset_h5 = "outputs/data/mimic_fhir_mdp_dataset.h5"
        ope_csv = "outputs/data/mimic_fhir_ope_table.csv"
        wis_json = "outputs/ope/mimic_fhir_wis_summary.json"
        summary_json = "outputs/data/mimic_fhir_summary.json"
        steps = [
            [
                args.python,
                "-m",
                "src.data.extract_sepsis_cohort",
                "--fhir-dir",
                args.fhir_dir,
                "--out-npz",
                dataset_npz,
                "--out-ope-csv",
                ope_csv,
            ],
            [
                args.python,
                "-m",
                "src.data.build_mdp_dataset",
                "--in-npz",
                dataset_npz,
                "--out-h5",
                dataset_h5,
                "--force",
            ],
        ]
    elif args.use_bigquery:
        dataset_npz = "outputs/data/mimic_bq_mdp_raw.npz"
        dataset_h5 = "outputs/data/mimic_bq_mdp_dataset.h5"
        ope_csv = "outputs/data/mimic_bq_ope_table.csv"
        wis_json = "outputs/ope/mimic_bq_wis_summary.json"
        summary_json = "outputs/data/mimic_bq_summary.json"
        steps = [
            [
                args.python,
                "-m",
                "src.data.extract_sepsis_cohort_bigquery",
                "--gcp-project-id",
                args.gcp_project_id,
                "--bq-location",
                args.bq_location,
                "--mimic-project",
                args.mimic_project,
                "--derived-dataset",
                args.derived_dataset,
                "--icu-dataset",
                args.icu_dataset,
                "--hosp-dataset",
                args.hosp_dataset,
                "--max-stays",
                str(args.max_stays),
                "--out-npz",
                dataset_npz,
                "--out-ope-csv",
                ope_csv,
            ],
            [
                args.python,
                "-m",
                "src.data.build_mdp_dataset",
                "--in-npz",
                dataset_npz,
                "--out-h5",
                dataset_h5,
                "--force",
            ],
        ]
    else:
        steps = [
            [args.python, "-m", "src.data.mock_dataset"],
            [args.python, "-m", "src.data.build_mdp_dataset", "--force"],
        ]

    if not args.skip_train:
        if args.use_icu_sepsis:
            steps.extend(
                [
                    [
                        args.python,
                        "-m",
                        "src.train.train_bc",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "0.1",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "1.0",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "5.0",
                        "--n-steps",
                        "500",
                    ],
                ]
            )
        elif args.use_fhir:
            steps.extend(
                [
                    [
                        args.python,
                        "-m",
                        "src.train.train_bc",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "0.1",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "1.0",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "5.0",
                        "--n-steps",
                        "500",
                    ],
                ]
            )
        elif args.use_bigquery:
            steps.extend(
                [
                    [
                        args.python,
                        "-m",
                        "src.train.train_bc",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "0.1",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "1.0",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        dataset_h5,
                        "--dataset-npz",
                        dataset_npz,
                        "--alpha",
                        "5.0",
                        "--n-steps",
                        "500",
                    ],
                ]
            )
        else:
            steps.extend(
                [
                    [args.python, "-m", "src.train.train_bc", "--n-steps", "500"],
                    [args.python, "-m", "src.train.train_cql", "--alpha", "0.1", "--n-steps", "500"],
                    [args.python, "-m", "src.train.train_cql", "--alpha", "1.0", "--n-steps", "500"],
                    [args.python, "-m", "src.train.train_cql", "--alpha", "5.0", "--n-steps", "500"],
                ]
            )

    if not args.use_mimic_profiles:
        mu_source = args.mu_source
        if mu_source == "auto":
            mu_source = "bc_model" if not args.skip_train else "csv"

        wis_step = [
            args.python,
            "-m",
            "src.ope.wis_eval",
            "--in-csv",
            ope_csv,
            "--out-json",
            wis_json,
            "--out-csv",
            ope_csv,
            "--mu-source",
            mu_source,
        ]

        if mu_source == "bc_model":
            wis_step.extend(
                [
                    "--dataset-npz",
                    dataset_npz,
                    "--bc-model",
                    "outputs/models/bc/bc_model.d3",
                    "--cql-model",
                    "outputs/models/cql/alpha_1/cql_model.d3",
                    "--sync-bc-prob-with-mu",
                ]
            )

        steps.append(wis_step)

        steps.append(
            [
                args.python,
                "-m",
                "src.ope.build_report_assets",
                "--summary-json",
                summary_json,
                "--wis-json",
                wis_json,
                "--ope-csv",
                ope_csv,
                "--dataset-npz",
                dataset_npz,
                "--bc-metrics-json",
                "outputs/models/bc/train_metrics.json",
                "--cql-dir",
                "outputs/models/cql",
                "--bc-model",
                "outputs/models/bc/bc_model.d3",
                "--cql-model",
                "outputs/models/cql/alpha_1/cql_model.d3",
                "--out-json",
                "report/generated/report_metrics.json",
                "--out-table",
                "report/generated/results_table.tex",
                "--fig-dir",
                "report/generated/figures",
            ]
        )

    for cmd in steps:
        _run(cmd, cwd=root)

    for out_json, seed_jsons, seeds in aggregate_jobs:
        print("Aggregating seed WIS:", str(out_json))
        _aggregate_seed_wis_jsons(seed_json_paths=seed_jsons, out_json=out_json, seeds=seeds)

    print("Pipeline complete. Outputs are under finalProj/outputs")


if __name__ == "__main__":
    main()
