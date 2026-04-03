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
    parser.add_argument(
        "--fhir-dir",
        type=str,
        default="data/mimic-iv-clinical-database-demo-on-fhir-2.0/mimic-fhir",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]

    if args.use_fhir:
        steps = [
            [
                args.python,
                "-m",
                "src.data.extract_sepsis_cohort",
                "--fhir-dir",
                args.fhir_dir,
                "--out-npz",
                "outputs/data/mimic_fhir_mdp_raw.npz",
                "--out-ope-csv",
                "outputs/data/mimic_fhir_ope_table.csv",
            ],
            [
                args.python,
                "-m",
                "src.data.build_mdp_dataset",
                "--in-npz",
                "outputs/data/mimic_fhir_mdp_raw.npz",
                "--out-h5",
                "outputs/data/mimic_fhir_mdp_dataset.h5",
                "--force",
            ],
        ]
    else:
        steps = [
            [args.python, "-m", "src.data.mock_dataset"],
            [args.python, "-m", "src.data.build_mdp_dataset", "--force"],
        ]

    if not args.skip_train:
        if args.use_fhir:
            steps.extend(
                [
                    [
                        args.python,
                        "-m",
                        "src.train.train_bc",
                        "--dataset-h5",
                        "outputs/data/mimic_fhir_mdp_dataset.h5",
                        "--dataset-npz",
                        "outputs/data/mimic_fhir_mdp_raw.npz",
                        "--n-steps",
                        "500",
                    ],
                    [
                        args.python,
                        "-m",
                        "src.train.train_cql",
                        "--dataset-h5",
                        "outputs/data/mimic_fhir_mdp_dataset.h5",
                        "--dataset-npz",
                        "outputs/data/mimic_fhir_mdp_raw.npz",
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
                        "outputs/data/mimic_fhir_mdp_dataset.h5",
                        "--dataset-npz",
                        "outputs/data/mimic_fhir_mdp_raw.npz",
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
                        "outputs/data/mimic_fhir_mdp_dataset.h5",
                        "--dataset-npz",
                        "outputs/data/mimic_fhir_mdp_raw.npz",
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

    if args.use_fhir:
        steps.append(
            [
                args.python,
                "-m",
                "src.ope.wis_eval",
                "--in-csv",
                "outputs/data/mimic_fhir_ope_table.csv",
                "--out-json",
                "outputs/ope/mimic_fhir_wis_summary.json",
            ]
        )
    else:
        steps.append([args.python, "-m", "src.ope.wis_eval"])

    for cmd in steps:
        _run(cmd, cwd=root)

    print("Pipeline complete. Outputs are under finalProj/outputs")


if __name__ == "__main__":
    main()
