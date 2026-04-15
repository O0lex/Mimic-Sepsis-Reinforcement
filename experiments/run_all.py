from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end mock offline RL pipeline.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--use-fhir", action="store_true")
    parser.add_argument("--use-icu-sepsis", action="store_true")
    parser.add_argument("--use-bigquery", action="store_true")
    parser.add_argument("--use-mimic-profiles", action="store_true")
    parser.add_argument("--in-parquet", type=str, default="data/raw/mimic_sepsis_final.parquet")
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

    if args.use_mimic_profiles:
        selected_profiles = [args.profile] if args.profile in {"minimal", "full"} else ["minimal", "full"]
        steps = [
            [
                args.python,
                "-m",
                "src.data.preprocess",
                "--in-parquet",
                args.in_parquet,
                "--out-dir",
                "outputs/data",
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
            bc_model = f"{bc_dir}/bc_model.d3"
            cql_model = f"{cql_dir}/alpha_1/cql_model.d3"

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
                            "--out-dir",
                            bc_dir,
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
                            "--out-dir",
                            cql_dir,
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
                            "--out-dir",
                            cql_dir,
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
                            "--out-dir",
                            cql_dir,
                            "--alpha",
                            "5.0",
                            "--n-steps",
                            "500",
                        ],
                    ]
                )

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
                        bc_model,
                        "--cql-model",
                        cql_model,
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
                    f"{bc_dir}/train_metrics.json",
                    "--cql-dir",
                    cql_dir,
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

    print("Pipeline complete. Outputs are under finalProj/outputs")


if __name__ == "__main__":
    main()
