#!/usr/bin/env python3
"""Run UdonPred's 7x7 cross-dataset evaluation matrix.

This script mirrors the Figure 2 workflow in UdonPred/eval/eval.ipynb, but uses
the bundled ONNX heads in UdonPred/weights instead of the notebook's ../exported
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
NEGATED_DATASETS = {"plddt", "chezod"}


def to_numpy(values: torch.Tensor) -> list[float]:
    return values.detach().cpu().numpy()


def auroc(preds: torch.Tensor, labels: torch.Tensor, task: str = "binary") -> float:
    del task
    return float(roc_auc_score(to_numpy(labels).astype(int), to_numpy(preds)))


def average_precision(preds: torch.Tensor, labels: torch.Tensor, task: str = "binary") -> float:
    del task
    return float(average_precision_score(to_numpy(labels).astype(int), to_numpy(preds)))


def spearman_corrcoef(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if len(preds) < 2:
        return float("nan")
    value = spearmanr(to_numpy(preds), to_numpy(labels)).statistic
    return float(value)


def read_labels(path: Path) -> dict[str, list[float]]:
    labels = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            labels[str(record["id"])] = record["y"]
    return labels


def read_results(input_dir: Path) -> dict[str, list[float]]:
    results = {}
    for path in sorted(input_dir.glob("*.caid")):
        with path.open() as handle:
            lines = handle.readlines()

        protein_id = lines[0].strip().lstrip(">")
        
        # TriZOD FASTA IDs need format conversion: "11011111" -> "11011_1_1_1"
        # Only apply if ID is 7-8 digits (TriZOD format)
        if len(protein_id) >= 7 and protein_id.isdigit():
            base_len = len(protein_id) - 3
            protein_id = protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])
        
        scores = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            scores.append(float(line.split("\t")[2]))

        results[protein_id] = scores
    return results


def flatten_masked(values: list[list[float]], mask_value: float = 999) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor([item for row in values for item in row], dtype=torch.float32)
    mask = tensor != mask_value
    return tensor[mask], mask


def bootstrap_std(preds: torch.Tensor, labels: torch.Tensor, metric_fn, n: int, **kwargs) -> float:
    metrics = []
    for _ in range(n):
        indices = torch.randint(0, len(labels), (len(labels),))
        metrics.append(metric_fn(preds[indices], labels[indices], **kwargs))
    return torch.tensor(metrics).std().item()


def run_predictions(
    udonpred_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    smooth: float,
    force: bool,
) -> None:
    weights_dir = udonpred_dir / "weights"
    for train_set in DATASETS:
        for test_set in DATASETS:
            result_dir = output_dir / "predictions" / f"{train_set}_{test_set}"
            if result_dir.exists() and any(result_dir.glob("*.caid")) and not force:
                print(f"Skipping existing predictions: {result_dir}")
                continue

            result_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "uv",
                "run",
                "predict.py",
                str(udonpred_dir / "data" / test_set / "test.fasta"),
                str(weights_dir),
                "--target",
                train_set,
                "--output",
                str(result_dir),
                "--device",
                device,
                "--batch-size",
                str(batch_size),
                "--smooth",
                str(smooth),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, cwd=udonpred_dir, check=True)


def load_prediction_frames(udonpred_dir: Path, output_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for test_set in DATASETS:
        labels = read_labels(udonpred_dir / "data" / test_set / "test.jsonl")
        for train_set in DATASETS:
            preds = read_results(output_dir / "predictions" / f"{train_set}_{test_set}")
            missing = sorted(set(labels) - set(preds))
            if missing:
                raise ValueError(
                    f"{train_set}_{test_set} is missing {len(missing)} predictions; "
                    f"first missing id: {missing[0]}"
                )
            results[train_set][test_set] = pd.DataFrame(
                {"label": pd.Series(labels), "pred": pd.Series(preds)}
            )
    return results


def compute_metrics(
    frames: dict[str, dict[str, pd.DataFrame]],
    bootstrap_samples: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    stds: dict[str, dict[str, float]] = defaultdict(dict)

    for train_set in DATASETS:
        for test_set in DATASETS:
            frame = frames[train_set][test_set]
            labels, mask = flatten_masked(frame["label"].tolist())
            preds = torch.tensor(
                [item for row in frame["pred"].tolist() for item in row],
                dtype=torch.float32,
            )[mask]

            if train_set in NEGATED_DATASETS:
                preds = -preds
            if test_set in NEGATED_DATASETS:
                labels = -labels

            if test_set == "disprot":
                binary_labels = labels.to(torch.int)
                auroc_value = auroc(preds, binary_labels, task="binary")
                ap_value = average_precision(preds, binary_labels, task="binary")
                values[train_set]["disprot\n(AUROC)"] = float(auroc_value)
                values[train_set]["disprot\n(AP)"] = float(ap_value)
                if bootstrap_samples:
                    stds[train_set]["disprot\n(AUROC)"] = bootstrap_std(
                        preds, binary_labels, auroc, bootstrap_samples, task="binary"
                    )
                    stds[train_set]["disprot\n(AP)"] = bootstrap_std(
                        preds, binary_labels, average_precision, bootstrap_samples, task="binary"
                    )
            else:
                value = spearman_corrcoef(preds, labels)
                values[train_set][test_set] = float(value)
                if bootstrap_samples:
                    stds[train_set][test_set] = bootstrap_std(
                        preds, labels, spearman_corrcoef, bootstrap_samples
                    )

    return values, stds


def write_matrix_csv(matrix: dict[str, dict[str, float]], path: Path) -> None:
    columns = [
        "train_dataset",
        "trizod",
        "chezod",
        "softdis",
        "pdbflex",
        "atlas",
        "plddt",
        "disprot\n(AP)",
        "disprot\n(AUROC)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for train_set in DATASETS:
            row = {"train_dataset": train_set}
            for column in columns[1:]:
                value = matrix[train_set].get(column)
                row[column] = "" if value is None else f"{value:.6f}"
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/udonpred_matrix"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument("--force", action="store_true", help="Regenerate existing prediction files.")
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Only recompute metrics from existing prediction files.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Bootstrap samples for standard deviations. Use 100 to match the notebook.",
    )
    return parser.parse_args()


def validate_device(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False in this Python "
            "environment. Use --device cpu here, or run on an NVIDIA/CUDA machine."
        )


def main() -> None:
    args = parse_args()
    udonpred_dir = args.udonpred_dir.resolve()
    output_dir = args.output_dir.resolve()
    validate_device(args.device)

    if not args.skip_predictions:
        run_predictions(
            udonpred_dir,
            output_dir,
            args.device,
            args.batch_size,
            args.smooth,
            args.force,
        )

    frames = load_prediction_frames(udonpred_dir, output_dir)
    matrix, stds = compute_metrics(frames, args.bootstrap_samples)

    write_matrix_csv(matrix, output_dir / "matrix.csv")
    if args.bootstrap_samples:
        write_matrix_csv(stds, output_dir / "matrix_std.csv")

    print(f"Wrote {output_dir / 'matrix.csv'}")
    if args.bootstrap_samples:
        print(f"Wrote {output_dir / 'matrix_std.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
