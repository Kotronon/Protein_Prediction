#!/usr/bin/env python3
"""Run the prediction-improvement analysis pipeline.

The pipeline intentionally keeps expensive prediction generation resumable:
completed UdonPred CAID directories are reused unless ``--force-predictions`` is
passed. It combines the useful pieces from the current branches: UdonPred
matrix, simple baselines, annotation ceiling, normalized headroom, validation
ensembles, and optional predictor-behavior diagnostics.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteinprediction-mpl")


HUMAN_PROTEOME_PREDICTORS = {
    "trizod": Path("results/human_proteome/UdonPred/trizod"),
    "chezod": Path("results/human_proteome/UdonPred/chezod"),
    "softdis": Path("results/human_proteome/UdonPred/softdis"),
    "atlas": Path("results/human_proteome/UdonPred/atlas"),
    "plddt": Path("results/human_proteome/UdonPred/plddt"),
    "disprot": Path("results/human_proteome/UdonPred/disprot"),
    "SETH": Path("results/human_proteome/SETH/seth_human_proteome.caid"),
    "IUPred3": Path("results/human_proteome/IUPred3"),
    "ADOPT": Path("results/human_proteome/ADOPT"),
    "metapredict": Path("results/human_proteome/metapredict/metapredict_human_proteome.caid"),
    "PUNCH2_light": Path("results/human_proteome/PUNCH2_light/disorder"),
    "DisoFLAG": Path("results/human_proteome/DisoFLAG/caid"),
    "DisorderUnetLM": Path("results/human_proteome/DisorderUnetLM/disorder"),
    "DisPredict3": Path("results/human_proteome/Dispredict3_native/caid"),
}
DATASETS = [
    "trizod",
    "trizod_updated",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot",
]


def has_caid(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".caid"
    if path.is_dir():
        return any(path.glob("*.caid"))
    return False


def available_predictor_args(include_pdbflex: bool) -> list[str]:
    predictors = dict(HUMAN_PROTEOME_PREDICTORS)
    if include_pdbflex:
        predictors["pdbflex"] = Path("results/human_proteome/UdonPred/pdbflex")
    return [f"{name}={path}" for name, path in predictors.items() if has_caid(path)]


def exclusion_suffix(excluded: list[str]) -> str:
    return "" if not excluded else "_without_" + "_".join(excluded)


def focus_suffix(focus_datasets: list[str]) -> str:
    return "" if not focus_datasets else "_focus_" + "_".join(focus_datasets)


def pipeline_suffix(
    excluded: list[str],
    replace_trizod_with_updated: bool,
    focus_datasets: list[str],
) -> str:
    prefix = "_trizod_updated" if replace_trizod_with_updated else ""
    return prefix + exclusion_suffix(excluded) + focus_suffix(focus_datasets)


def ensure_trizod_updated_assets(
    python: str,
    root: Path,
    source_dir: Path,
    weights_source: Path,
    weights_dir: Path,
    dry_run: bool,
) -> None:
    data_dir = root / "UdonPred" / "data" / "trizod_updated"
    required_data = [
        data_dir / "train.fasta",
        data_dir / "train.jsonl",
        data_dir / "valid.fasta",
        data_dir / "valid.jsonl",
        data_dir / "test.fasta",
        data_dir / "test.jsonl",
    ]
    if not all(path.exists() for path in required_data):
        run_step(
            "Prepare updated TriZOD UdonPred data",
            [
                python,
                "udonpred_analysis_extensions/scripts/prepare_trizod_updated_udonpred.py",
                "--source-dir",
                str(source_dir),
                "--output-dir",
                str(data_dir),
            ],
            dry_run,
            continue_on_error=False,
            cwd=root,
        )

    required_heads = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
    if dry_run:
        print("\n== Prepare updated TriZOD weights ==")
        print(f"Ensure {weights_dir} contains base heads and trizod_updated.onnx")
        return
    weights_dir.mkdir(parents=True, exist_ok=True)
    for head in required_heads:
        source = root / "UdonPred" / "weights" / f"{head}.onnx"
        destination = weights_dir / f"{head}.onnx"
        if not destination.exists():
            shutil.copy2(source, destination)
    updated_destination = weights_dir / "trizod_updated.onnx"
    if not updated_destination.exists():
        if not weights_source.exists():
            raise FileNotFoundError(
                f"Missing trizod_updated ONNX head: {weights_source}. "
                "Export or bootstrap it before using --replace-trizod-with-updated."
            )
        shutil.copy2(weights_source, updated_destination)


def run_step(
    name: str,
    cmd: list[str],
    dry_run: bool,
    continue_on_error: bool,
    cwd: Path,
) -> None:
    print(f"\n== {name} ==")
    print(" ".join(cmd))
    if dry_run:
        return
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError:
        if continue_on_error:
            print(f"Step failed but continuing: {name}", file=sys.stderr)
            return
        raise


def outputs_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def maybe_run_step(
    name: str,
    cmd: list[str],
    outputs: list[Path],
    reuse_existing: bool,
    dry_run: bool,
    continue_on_error: bool,
    cwd: Path,
) -> None:
    if reuse_existing and outputs_exist(outputs):
        print(f"\n== {name} ==")
        print("Reusing existing outputs: " + ", ".join(str(path) for path in outputs))
        return
    run_step(name, cmd, dry_run, continue_on_error, cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--force-predictions", action="store_true")
    parser.add_argument("--force-validation", action="store_true")
    parser.add_argument("--skip-validation-predictions", action="store_true")
    parser.add_argument(
        "--max-fit-residues",
        type=int,
        default=0,
        help="Subsample validation residues for ensemble fitting; 0 uses all validation residues.",
    )
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=0,
        help="Largest predictor subset tried by subset-ensemble search; 0 uses all active heads.",
    )
    parser.add_argument(
        "--ensemble-focus-datasets",
        nargs="*",
        default=[],
        choices=DATASETS,
        help="Fit/select global ensembles only on these validation datasets; default uses all active datasets.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip pipeline steps whose primary output files already exist.",
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        choices=DATASETS,
        help="Exclude datasets from ensemble inputs, ensemble targets, and improvement figures.",
    )
    parser.add_argument(
        "--replace-trizod-with-updated",
        action="store_true",
        help="Use trizod_updated instead of trizod in ensemble inputs, targets, and figures.",
    )
    parser.add_argument(
        "--trizod-updated-source-dir",
        type=Path,
        default=Path("trizod_updated"),
        help="Updated TriZOD release directory used to prepare UdonPred/data/trizod_updated.",
    )
    parser.add_argument(
        "--trizod-updated-head",
        type=Path,
        default=Path("udonpred_analysis_extensions/UdonPred/weights_caid4_8heads/trizod_updated.onnx"),
        help="ONNX head used for trizod_updated predictions.",
    )
    parser.add_argument(
        "--updated-weights-dir",
        type=Path,
        default=Path("UdonPred/weights_caid4_8heads"),
        help="Weights directory containing all base heads plus trizod_updated.onnx.",
    )
    parser.add_argument("--include-mmseqs", action="store_true")
    parser.add_argument("--include-pdbflex-global", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ceiling", action="store_true")
    parser.add_argument("--skip-headroom", action="store_true")
    parser.add_argument("--skip-ensembles", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-global-behavior", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def validate_device(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA was requested, but torch is not installed.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False in this Python "
            "environment. Use --device cpu here, or run on an NVIDIA/CUDA machine."
        )


def main() -> None:
    args = parse_args()
    validate_device(args.device)
    root = Path.cwd()
    python = sys.executable
    suffix = pipeline_suffix(
        args.exclude_datasets,
        args.replace_trizod_with_updated,
        args.ensemble_focus_datasets,
    )
    ensemble_output_dir = Path(f"results/ensembles{suffix}")
    figure_output_dir = Path(f"results/figures/prediction_improvement{suffix}")
    weights_dir = args.updated_weights_dir if args.replace_trizod_with_updated else Path("UdonPred/weights")

    if args.replace_trizod_with_updated:
        ensure_trizod_updated_assets(
            python,
            root,
            args.trizod_updated_source_dir,
            args.trizod_updated_head,
            args.updated_weights_dir,
            args.dry_run,
        )

    if args.force_predictions and args.skip_predictions:
        raise ValueError("--force-predictions and --skip-predictions are mutually exclusive")

    if not args.skip_matrix:
        cmd = [
            python,
            "scripts/run_udonpred_matrix.py",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--smooth",
            str(args.smooth),
        ]
        if args.skip_predictions:
            cmd.append("--skip-predictions")
        if args.force_predictions:
            cmd.append("--force")
        if args.bootstrap_samples:
            cmd.extend(["--bootstrap-samples", str(args.bootstrap_samples)])
        maybe_run_step(
            "UdonPred 7x7 test matrix",
            cmd,
            [Path("results/udonpred_matrix/matrix.csv")],
            args.reuse_existing,
            args.dry_run,
            args.continue_on_error,
            root,
        )

    if not args.skip_baselines:
        maybe_run_step(
            "Simple sequence baselines",
            [python, "scripts/run_simple_baselines.py"],
            [Path("results/simple_baselines/matrix.csv")],
            args.reuse_existing,
            args.dry_run,
            args.continue_on_error,
            root,
        )

    if not args.skip_ceiling:
        maybe_run_step(
            "Exact annotation ceiling",
            [
                python,
                "scripts/estimate_annotation_ceiling.py",
                "--output-dir",
                "results/annotation_ceiling",
            ],
            [
                Path("results/annotation_ceiling/annotation_ceiling_summary.csv"),
                Path("results/annotation_ceiling/overlap_details.csv"),
            ],
            args.reuse_existing,
            args.dry_run,
            args.continue_on_error,
            root,
        )
        if args.include_mmseqs:
            maybe_run_step(
                "MMseqs annotation ceiling",
                [
                    python,
                    "scripts/estimate_annotation_ceiling.py",
                    "--use-mmseqs",
                    "--output-dir",
                    "results/annotation_ceiling_mmseqs",
                ],
                [
                    Path("results/annotation_ceiling_mmseqs/annotation_ceiling_summary.csv"),
                    Path("results/annotation_ceiling_mmseqs/overlap_details.csv"),
                ],
                args.reuse_existing,
                args.dry_run,
                args.continue_on_error,
                root,
            )

    if not args.skip_headroom:
        maybe_run_step(
            "Normalized headroom",
            [python, "scripts/compute_normalized_headroom.py"],
            [
                Path("results/normalized_headroom/ceiling_matrix.csv"),
                Path("results/normalized_headroom/normalized_headroom_summary.csv"),
            ],
            args.reuse_existing,
            args.dry_run,
            args.continue_on_error,
            root,
        )

    if not args.skip_ensembles:
        cmd = [
            python,
            "scripts/run_udonpred_ensembles.py",
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--smooth",
            str(args.smooth),
            "--weights-dir",
            str(weights_dir),
        ]
        if args.force_validation:
            cmd.append("--force-validation")
        if args.skip_validation_predictions:
            cmd.append("--skip-validation-predictions")
        if args.max_fit_residues:
            cmd.extend(["--max-fit-residues", str(args.max_fit_residues)])
        if args.max_subset_size:
            cmd.extend(["--max-subset-size", str(args.max_subset_size)])
        if args.ensemble_focus_datasets:
            cmd.append("--ensemble-focus-datasets")
            cmd.extend(args.ensemble_focus_datasets)
        cmd.extend(["--output-dir", str(ensemble_output_dir)])
        if args.exclude_datasets:
            cmd.append("--exclude-datasets")
            cmd.extend(args.exclude_datasets)
        if args.replace_trizod_with_updated:
            cmd.append("--replace-trizod-with-updated")
        maybe_run_step(
            "Validation-trained UdonPred ensembles",
            cmd,
            [
                ensemble_output_dir / "ensemble_matrix.csv",
                ensemble_output_dir / "ensemble_summary.csv",
            ],
            args.reuse_existing,
            args.dry_run,
            args.continue_on_error,
            root,
        )

    if not args.skip_figures:
        required = [
            Path("results/simple_baselines/matrix.csv"),
            Path("results/udonpred_matrix/matrix.csv"),
            ensemble_output_dir / "ensemble_matrix.csv",
            Path("results/normalized_headroom/ceiling_matrix.csv"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing and not args.dry_run:
            print("\n== Prediction improvement figures ==")
            print("Skipped: missing " + ", ".join(missing))
        else:
            maybe_run_step(
                "Prediction improvement figures",
                [
                    python,
                    "scripts/plot_prediction_improvement.py",
                    "--ensemble-matrix",
                    str(ensemble_output_dir / "ensemble_matrix.csv"),
                    "--ensemble-summary",
                    str(ensemble_output_dir / "ensemble_summary.csv"),
                    "--best-individual-heads",
                    str(ensemble_output_dir / "best_individual_heads.csv"),
                    "--ensemble-weights",
                    str(ensemble_output_dir / "ensemble_weights.csv"),
                    "--ensemble-subset-choices",
                    str(ensemble_output_dir / "ensemble_subset_choices.csv"),
                    "--validation-selected-heads",
                    str(ensemble_output_dir / "validation_selected_heads.csv"),
                    "--output-dir",
                    str(figure_output_dir),
                    *(
                        ["--exclude-datasets", *args.exclude_datasets]
                        if args.exclude_datasets
                        else []
                    ),
                    *(
                        ["--replace-trizod-with-updated"]
                        if args.replace_trizod_with_updated
                        else []
                    ),
                ],
                [
                    figure_output_dir / "ensemble_delta_heatmap.png",
                    figure_output_dir / "best_individual_vs_ensemble.png",
                    figure_output_dir / "best_ensemble_combination_by_metric.png",
                    figure_output_dir / "baseline_udon_ensemble_ceiling.png",
                    figure_output_dir / "improvement_summary.csv",
                ],
                args.reuse_existing,
                args.dry_run,
                args.continue_on_error,
                root,
            )

    if not args.skip_global_behavior:
        predictor_args = available_predictor_args(args.include_pdbflex_global)
        pairwise_csv = Path("results/compare_predictors_with_all_predictors_wo_pdbflex/pairwise_agreement.csv")
        if len(predictor_args) >= 2 and pairwise_csv.exists():
            cmd = [
                python,
                "scripts/analyze_global_predictor_behavior.py",
                "--pairwise-csv",
                str(pairwise_csv),
                "--annotation-csv",
                "results/annotation_ceiling/annotation_ceiling_summary.csv",
                "--output-dir",
                "results/compare_predictors_with_all_predictors_wo_pdbflex/global_behavior",
            ]
            for predictor in predictor_args:
                cmd.extend(["--predictor", predictor])
            run_step("Global predictor behavior", cmd, args.dry_run, args.continue_on_error, root)
        else:
            print("\n== Global predictor behavior ==")
            print("Skipped: need at least two CAID predictor inputs and pairwise_agreement.csv.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
