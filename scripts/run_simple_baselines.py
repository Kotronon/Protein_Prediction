#!/usr/bin/env python3
"""Run simple disorder baselines on the bundled UdonPred datasets.

The output mirrors ``scripts/run_udonpred_matrix.py`` closely enough to compare
against the UdonPred 7x7 matrix:

* continuous datasets are scored with residue-level Spearman correlation
* DisProt is scored with residue-level average precision and AUROC
* CheZOD and pLDDT labels are negated so higher values consistently mean
  "more disorder" during evaluation

Baselines:

* ``aa_composition_logreg`` trains a balanced logistic regression on global
  amino-acid composition and broadcasts the protein-level probability to every
  residue. Continuous targets are binarized at the training-set median in the
  common disorder direction.
* ``coil_propensity`` is a train-independent per-residue secondary-structure
  heuristic: helix/sheet/turn Chou-Fasman propensities are smoothed locally and
  converted into a coil-like probability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
# These datasets have the opposite sign convention in the UdonPred evaluation:
# after negation, larger labels consistently mean "more disorder".
NEGATED_DATASETS = {"plddt", "chezod"}
# UdonPred uses 999 as a sentinel for residues without an evaluable label.
MASK_VALUE = 999.0
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}

# Chou-Fasman-style propensities. Values are normalized locally below; they are
# intentionally used only as a weak, transparent secondary-structure baseline.
# values for Helox and sheet: https://bmrb.io/referenc/choufas.shtml
# values for turn: Chou and Fasman beta turn prediction from https://tools.immuneepitope.org/bcell/help/
HELIX_PROPENSITY = {
    "A": 1.45,
    "C": 0.77,
    "D": 0.98,
    "E": 1.53,
    "F": 1.12,
    "G": 0.53,
    "H": 1.24,
    "I": 1.00,
    "K": 1.07,
    "L": 1.34,
    "M": 1.20,
    "N": 0.73,
    "P": 0.59,
    "Q": 1.17,
    "R": 0.79,
    "S": 0.79,
    "T": 0.82,
    "V": 1.14,
    "W": 1.14,
    "Y": 0.61,
}
SHEET_PROPENSITY = {
    "A": 0.97,
    "C": 1.30,
    "D": 0.80,
    "E": 0.26,
    "F": 1.28,
    "G": 0.81,
    "H": 0.71,
    "I": 1.60,
    "K": 0.74,
    "L": 1.22,
    "M": 1.67,
    "N": 0.65,
    "P": 0.62,
    "Q": 1.23,
    "R": 0.90,
    "S": 0.72,
    "T": 1.20,
    "V": 1.65,
    "W": 1.19,
    "Y": 1.29,
}
TURN_PROPENSITY = {
    "A": 0.66,
    "C": 1.19,
    "D": 1.46,
    "E": 0.74,
    "F": 0.60,
    "G": 1.56,
    "H": 0.95,
    "I": 0.47,
    "K": 1.01,
    "L": 0.59,
    "M": 0.60,
    "N": 1.56,
    "P": 1.52,
    "Q": 0.98,
    "R": 0.95,
    "S": 1.43,
    "T": 0.96,
    "V": 0.50,
    "W": 0.96,
    "Y": 1.14,
}


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str
    labels: np.ndarray

"""Read Dataset and store their records."""
def read_jsonl(path: Path) -> list[ProteinRecord]:
    records = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            sequence = str(record["x_0"])
            labels = np.asarray(record["y"], dtype=np.float64)
            if len(sequence) != len(labels):
                raise ValueError(
                    f"{path}: {record['id']} has {len(sequence)} residues but {len(labels)} labels"
                )
            records.append(ProteinRecord(str(record["id"]), sequence, labels))
    return records

"""Negate labels for datasets cheZOD and pLDDT so that higher values consistently indicate more disorder during evaluation."""
def labels_in_disorder_direction(labels: np.ndarray, dataset: str) -> np.ndarray:
    return -labels if dataset in NEGATED_DATASETS else labels

"""Create a boolean mask for valid (non-masked, non-NaN) labels."""
def valid_mask(labels: np.ndarray) -> np.ndarray:
    return np.isfinite(labels) & (labels != MASK_VALUE)

"""Flatten lists of per-residue values while applying a mask to filter out invalid entries."""
def amino_acid_composition(sequence: str) -> np.ndarray:
    """Return normalized 20-dimensional amino-acid composition features."""
    features = np.zeros(len(AMINO_ACIDS), dtype=np.float64)
    total = 0
    for residue in sequence.upper():
        index = AA_TO_INDEX.get(residue)
        if index is not None:
            features[index] += 1.0
            total += 1
    if total:
        features /= total
    return features

"""Build training data for the amino-acid composition logistic regression baseline."""
def build_composition_training_data(
    records: Iterable[ProteinRecord],
    dataset: str,
    max_train_residues: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for record in records:
        directed_labels = labels_in_disorder_direction(record.labels, dataset)
        mask = valid_mask(record.labels)
        if not np.any(mask):
            continue
        residue_labels = directed_labels[mask]
        protein_features = amino_acid_composition(record.sequence)
        # The feature is protein-level, but the evaluation is residue-level.
        # Repeating the same composition vector gives each labeled residue a
        # training example while keeping this baseline intentionally simple.
        features.append(np.repeat(protein_features[None, :], len(residue_labels), axis=0))
        labels.append(residue_labels)

    if not features:
        raise ValueError(f"No valid training labels for {dataset}")

    x = np.vstack(features)
    y_continuous = np.concatenate(labels)
    if dataset == "disprot":
        y = y_continuous.astype(int)
    else:
        # Continuous disorder targets are converted to a binary training target.
        # This keeps the classifier simple while preserving the evaluation on the
        # original continuous labels via Spearman correlation.
        threshold = float(np.median(y_continuous))
        y = (y_continuous >= threshold).astype(int)
        if len(np.unique(y)) != 2:
            # Some datasets have many values exactly at the median; a strict
            # split avoids collapsing every residue into the positive class.
            y = (y_continuous > threshold).astype(int)

    if len(np.unique(y)) != 2:
        raise ValueError(f"{dataset} training labels collapse to one class")

    if max_train_residues and len(y) > max_train_residues:
        indices = rng.choice(len(y), size=max_train_residues, replace=False)
        x = x[indices]
        y = y[indices]
    return x, y

"""Parse prediction files in UdonPred format, returning a mapping from protein ID to per-residue scores."""
def predict_composition_model(model, records: Iterable[ProteinRecord]) -> list[np.ndarray]:
    predictions = []
    for record in records:
        features = amino_acid_composition(record.sequence).reshape(1, -1)
        protein_probability = float(model.predict_proba(features)[0, 1])
        # Broadcast the protein-level probability to all residues. Any residue
        # resolution achieved by this baseline therefore comes only from the
        # evaluation masks and not from local sequence context.
        predictions.append(np.full(len(record.sequence), protein_probability, dtype=np.float64))
    return predictions

"""Smooth values with a uniform kernel, using edge padding to maintain the same length."""
def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

"""Take coil propensity features based on smoothed Chou-Fasman propensities and convert them into a disorder-like signal."""
def coil_propensity(sequence: str, window: int) -> np.ndarray:
    """Compute a weak coil-like score from local Chou-Fasman propensities."""
    helix = np.asarray([HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence.upper()])
    sheet = np.asarray([SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence.upper()])
    turn = np.asarray([TURN_PROPENSITY.get(aa, 1.0) for aa in sequence.upper()])

    # Smooth local propensities so the score behaves like a small-window
    # secondary-structure predictor rather than independent residue lookup.
    helix = smooth(helix, window)
    sheet = smooth(sheet, window)
    turn = smooth(turn, window)

    # Treat helix/sheet propensity as structured signal and turn propensity as a
    # crude coil proxy. The ratio maps residues to a bounded disorder-like score.
    structured = np.maximum(helix, sheet)
    coil_signal = turn / (turn + structured + 1e-12)
    return np.clip(coil_signal, 0.0, 1.0)

"""Evaluate predictions against labels for a given dataset, applying appropriate metrics and handling edge cases."""
def predict_coil_propensity(records: Iterable[ProteinRecord], window: int) -> list[np.ndarray]:
    return [coil_propensity(record.sequence, window) for record in records]

"""flatten_valid applies the valid_mask to filter out invalid labels and corresponding predictions, returning 1D arrays of valid labels and predictions for evaluation."""
def flatten_valid(
    records: Iterable[ProteinRecord], predictions: Iterable[np.ndarray], dataset: str
) -> tuple[np.ndarray, np.ndarray]:
    flat_labels = []
    flat_predictions = []
    for record, pred in zip(records, predictions):
        if len(pred) != len(record.labels):
            raise ValueError(
                f"{record.protein_id}: prediction length {len(pred)} != label length {len(record.labels)}"
            )
        # Metrics are computed only on residues with real labels.
        mask = valid_mask(record.labels)
        if not np.any(mask):
            continue
        flat_labels.append(labels_in_disorder_direction(record.labels, dataset)[mask])
        flat_predictions.append(pred[mask])

    if not flat_labels:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.concatenate(flat_labels), np.concatenate(flat_predictions)

"""Generate column names for metrics based on the dataset, handling special cases like DisProt which has two metrics."""
def metric_columns_for_dataset(dataset: str) -> list[str]:
    if dataset == "disprot":
        return ["disprot\n(AP)", "disprot\n(AUROC)"]
    return [dataset]

"""Evaluate predictions against labels for a given dataset, applying appropriate metrics and handling edge cases."""
def evaluate(records: list[ProteinRecord], predictions: list[np.ndarray], dataset: str) -> dict[str, float]:
    labels, preds = flatten_valid(records, predictions, dataset)
    if len(labels) == 0:
        return {column: math.nan for column in metric_columns_for_dataset(dataset)}

    if dataset == "disprot":
        binary_labels = labels.astype(int)
        return {
            "disprot\n(AP)": float(average_precision_score(binary_labels, preds)),
            "disprot\n(AUROC)": float(roc_auc_score(binary_labels, preds)),
        }

    if np.unique(preds).size < 2 or np.unique(labels).size < 2:
        return {dataset: math.nan}
    return {dataset: float(spearmanr(preds, labels).statistic)}

"""Write the results matrix to a CSV file, ensuring consistent formatting and handling NaN values appropriately."""
def write_matrix_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    columns = [
        "baseline",
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
        for row in rows:
            output_row = {}
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    output_row[column] = "" if math.isnan(value) else f"{value:.6f}"
                else:
                    output_row[column] = value
            writer.writerow(output_row)
            
"""Plot result matrix as a heatmap, saving the figure to disk."""
def plot_matrix_heatmap(matrix: dict[str, dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_datasets = list(matrix.keys())
    test_datasets = [
        "trizod",
        "chezod",
        "softdis",
        "pdbflex",
        "atlas",
        "plddt",
        "disprot\n(AP)",
        "disprot\n(AUROC)",
    ]
    values = np.full((len(train_datasets), len(test_datasets)), np.nan, dtype=np.float64)
    for i, train_set in enumerate(train_datasets):
        for j, test_set in enumerate(test_datasets):
            value = matrix[train_set].get(test_set)
            if value is not None:
                values[i, j] = value

        
    plt.figure(figsize=(10, 6))
    sns.heatmap(values, xticklabels=test_datasets, yticklabels=train_datasets, annot=True, fmt=".3f")
    plt.xlabel("Test dataset")
    plt.ylabel("Training dataset")
    plt.title("Baseline cross-dataset evaluation matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

"""Parse command-line arguments for the script, providing defaults and help messages for each option."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/simple_baselines"))
    parser.add_argument(
        "--max-train-residues",
        type=int,
        default=300_000,
        help="Subsample this many labeled residues per logistic-regression training set; 0 disables.",
    )
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--coil-window", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    udonpred_dir = args.udonpred_dir.resolve()
    output_dir = args.output_dir.resolve()
    rng = np.random.default_rng(args.random_seed)

    train_records = {
        dataset: read_jsonl(udonpred_dir / "data" / dataset / "train.jsonl")
        for dataset in DATASETS
    }
    test_records = {
        dataset: read_jsonl(udonpred_dir / "data" / dataset / "test.jsonl")
        for dataset in DATASETS
    }

    rows: list[dict[str, str | float]] = []
    for train_dataset in DATASETS:
        x_train, y_train = build_composition_training_data(
            train_records[train_dataset], train_dataset, args.max_train_residues, rng
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=args.random_seed,
                solver="liblinear",
            ),
        )
        model.fit(x_train, y_train)

        row: dict[str, str | float] = {
            "baseline": "aa_composition_logreg",
            "train_dataset": train_dataset,
        }
        for test_dataset in DATASETS:
            predictions = predict_composition_model(model, test_records[test_dataset])
            row.update(evaluate(test_records[test_dataset], predictions, test_dataset))
        rows.append(row)

    coil_row: dict[str, str | float] = {
        "baseline": "coil_propensity",
        "train_dataset": "none",
    }
    for test_dataset in DATASETS:
        predictions = predict_coil_propensity(test_records[test_dataset], args.coil_window)
        coil_row.update(evaluate(test_records[test_dataset], predictions, test_dataset))
    rows.append(coil_row)

    matrix_path = output_dir / "matrix.csv"
    write_matrix_csv(rows, matrix_path)
    plot_matrix_heatmap(
        {row["train_dataset"]: {k: v for k, v in row.items() if k not in {"baseline", "train_dataset"}} for row in rows},
        output_dir / "baseline_matrix_heatmap.png",
    )
    print(f"Wrote {matrix_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
