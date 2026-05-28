#!/usr/bin/env python3
"""Estimate inter-annotation agreement between UdonPred evaluation datasets.

This script compares real residue annotations for proteins that occur in more
than one evaluation dataset. It is an approximate experimental ceiling for
disorder-prediction performance, not a shuffled-label or model benchmark.

Run from the repository root:

    python scripts/estimate_annotation_ceiling.py --output-dir results/annotation_ceiling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from run_simple_baselines import (
    DATASETS,
    NEGATED_DATASETS,
    ProteinRecord,
    read_jsonl,
    valid_mask,
)


BINARY_DATASETS = {"disprot"}
SUMMARY_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "match_mode",
    "n_proteins_overlap",
    "n_residues_compared",
    "annotation_type_a",
    "annotation_type_b",
    "metric",
    "value",
    "threshold_used",
    "notes",
]
DETAIL_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "protein_id_a",
    "protein_id_b",
    "sequence_length",
    "n_residues_compared",
    "match_mode",
]


@dataclass(frozen=True)
class MatchedProtein:
    record_a: ProteinRecord
    record_b: ProteinRecord
    match_mode: str
    n_residues_compared: int


def annotation_type(dataset: str) -> str:
    return "binary" if dataset in BINARY_DATASETS else "continuous"


def labels_in_common_direction(labels: np.ndarray, dataset: str) -> np.ndarray:
    """Return labels where larger values always mean more disorder.

    TriZOD, SoftDis, PDBFlex, Atlas, and DisProt are already used in this
    direction by the project code. CheZOD is negated because the existing
    UdonPred matrix and baseline scripts treat lower raw CheZOD values as more
    disordered. pLDDT is stored as a confidence score, so it is converted to a
    disorder-like score in [0, 1] with 1 - pLDDT / 100.
    """
    labels = labels.astype(np.float64, copy=True)
    if dataset == "plddt":
        converted = labels.copy()
        mask = valid_mask(labels)
        converted[mask] = 1.0 - (converted[mask] / 100.0)
        return converted
    if dataset in NEGATED_DATASETS:
        return -labels
    return labels


def load_dataset_records(udonpred_dir: Path, dataset: str) -> list[ProteinRecord]:
    path = udonpred_dir / "data" / dataset / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation labels for {dataset}: {path}")
    return read_jsonl(path)


def index_by_id(records: list[ProteinRecord]) -> dict[str, list[ProteinRecord]]:
    index: dict[str, list[ProteinRecord]] = defaultdict(list)
    for record in records:
        if record.protein_id:
            index[record.protein_id].append(record)
    return index


def index_by_sequence(records: list[ProteinRecord]) -> dict[str, list[ProteinRecord]]:
    index: dict[str, list[ProteinRecord]] = defaultdict(list)
    for record in records:
        if record.sequence:
            index[record.sequence].append(record)
    return index


def comparable_mask(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    limit = min(len(labels_a), len(labels_b))
    if limit == 0:
        return np.asarray([], dtype=bool)
    mask_a = valid_mask(labels_a[:limit])
    mask_b = valid_mask(labels_b[:limit])
    return mask_a & mask_b


def build_match(
    record_a: ProteinRecord,
    record_b: ProteinRecord,
    match_mode: str,
    min_residues: int,
) -> MatchedProtein | None:
    # Without fuzzy alignment, residue positions are comparable only when the
    # sequences are identical or a stable ID match has the same sequence length.
    if match_mode == "sequence" and record_a.sequence != record_b.sequence:
        return None
    if len(record_a.sequence) != len(record_b.sequence):
        return None
    mask = comparable_mask(record_a.labels, record_b.labels)
    n_residues = int(mask.sum())
    if n_residues < min_residues:
        return None
    return MatchedProtein(record_a, record_b, match_mode, n_residues)


def find_overlaps(
    records_a: list[ProteinRecord],
    records_b: list[ProteinRecord],
    min_residues: int,
) -> list[MatchedProtein]:
    matches: list[MatchedProtein] = []
    used: set[tuple[str, str]] = set()

    ids_a = index_by_id(records_a)
    ids_b = index_by_id(records_b)
    for protein_id in sorted(set(ids_a) & set(ids_b)):
        for record_a, record_b in product(ids_a[protein_id], ids_b[protein_id]):
            match = build_match(record_a, record_b, "id", min_residues)
            if match is None:
                continue
            key = (record_a.protein_id, record_b.protein_id)
            used.add(key)
            matches.append(match)

    seqs_a = index_by_sequence(records_a)
    seqs_b = index_by_sequence(records_b)
    for sequence in sorted(set(seqs_a) & set(seqs_b)):
        for record_a, record_b in product(seqs_a[sequence], seqs_b[sequence]):
            key = (record_a.protein_id, record_b.protein_id)
            if key in used:
                continue
            match = build_match(record_a, record_b, "sequence", min_residues)
            if match is None:
                continue
            used.add(key)
            matches.append(match)

    return matches


def pair_match_mode(matches: list[MatchedProtein]) -> str:
    modes = sorted({match.match_mode for match in matches})
    if not modes:
        return "none"
    return "+".join(modes)


def flatten_matched_labels(
    matches: list[MatchedProtein],
    dataset_a: str,
    dataset_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels_a = []
    labels_b = []
    for match in matches:
        raw_a = match.record_a.labels
        raw_b = match.record_b.labels
        mask = comparable_mask(raw_a, raw_b)
        directed_a = labels_in_common_direction(raw_a, dataset_a)
        directed_b = labels_in_common_direction(raw_b, dataset_b)
        labels_a.append(directed_a[: len(mask)][mask])
        labels_b.append(directed_b[: len(mask)][mask])
    if not labels_a:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return np.concatenate(labels_a), np.concatenate(labels_b)


def flatten_matched_protein_labels(
    matches: list[MatchedProtein],
    dataset_a: str,
    dataset_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    protein_labels_a = []
    protein_labels_b = []
    for match in matches:
        raw_a = match.record_a.labels
        raw_b = match.record_b.labels
        mask = comparable_mask(raw_a, raw_b)
        if not np.any(mask):
            continue
        directed_a = labels_in_common_direction(raw_a, dataset_a)
        directed_b = labels_in_common_direction(raw_b, dataset_b)
        protein_labels_a.append(float(np.mean(directed_a[: len(mask)][mask])))
        protein_labels_b.append(float(np.mean(directed_b[: len(mask)][mask])))
    return (
        np.asarray(protein_labels_a, dtype=np.float64),
        np.asarray(protein_labels_b, dtype=np.float64),
    )


def safe_metric(fn, *values) -> tuple[float, str]:
    try:
        return float(fn(*values)), ""
    except ValueError as exc:
        return math.nan, str(exc)
    except FloatingPointError as exc:
        return math.nan, str(exc)


def has_two_classes(values: np.ndarray) -> bool:
    return np.unique(values).size >= 2


def threshold_continuous(values: np.ndarray, threshold: float) -> np.ndarray:
    return (values >= threshold).astype(int)


def compute_pair_metrics(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    dataset_a: str,
    dataset_b: str,
    threshold: float,
) -> list[tuple[str, float, str]]:
    type_a = annotation_type(dataset_a)
    type_b = annotation_type(dataset_b)
    metrics: list[tuple[str, float, str]] = []

    if len(labels_a) == 0:
        return [("no_metric", math.nan, "no comparable residues")]

    if type_a == "continuous" and type_b == "continuous":
        if np.unique(labels_a).size < 2 or np.unique(labels_b).size < 2:
            note = "correlation undefined because one annotation has fewer than two values"
            metrics.extend([("spearman", math.nan, note), ("pearson", math.nan, note)])
        else:
            value, note = safe_metric(lambda x, y: spearmanr(x, y).statistic, labels_a, labels_b)
            metrics.append(("spearman", value, note))
            value, note = safe_metric(lambda x, y: pearsonr(x, y).statistic, labels_a, labels_b)
            metrics.append(("pearson", value, note))
        diff = labels_a - labels_b
        metrics.append(("mae", float(np.mean(np.abs(diff))), ""))
        metrics.append(("rmse", float(np.sqrt(np.mean(diff * diff))), ""))
        return metrics

    if type_a == "binary" and type_b == "binary":
        binary_a = labels_a.astype(int)
        binary_b = labels_b.astype(int)
        metrics.append(("accuracy", float(accuracy_score(binary_a, binary_b)), ""))
        if not has_two_classes(binary_a) or not has_two_classes(binary_b):
            note = "metric may be undefined because at least one annotation has one class"
            metrics.append(("f1", math.nan, note))
            metrics.append(("mcc", math.nan, note))
            metrics.append(("cohen_kappa", math.nan, note))
        else:
            metrics.append(("f1", float(f1_score(binary_a, binary_b, zero_division=0)), ""))
            metrics.append(("mcc", float(matthews_corrcoef(binary_a, binary_b)), ""))
            metrics.append(("cohen_kappa", float(cohen_kappa_score(binary_a, binary_b)), ""))
        return metrics

    if type_a == "continuous":
        continuous = labels_a
        binary = labels_b.astype(int)
        continuous_dataset = dataset_a
    else:
        continuous = labels_b
        binary = labels_a.astype(int)
        continuous_dataset = dataset_b

    if not has_two_classes(binary):
        note = "AUROC/AP/F1 undefined because binary annotation has one class"
        metrics.append(("auroc", math.nan, note))
        metrics.append(("average_precision", math.nan, note))
        metrics.append(("f1_thresholded", math.nan, note))
    else:
        value, note = safe_metric(roc_auc_score, binary, continuous)
        metrics.append(("auroc", value, note))
        value, note = safe_metric(average_precision_score, binary, continuous)
        metrics.append(("average_precision", value, note))
        pred_binary = threshold_continuous(continuous, threshold)
        if not has_two_classes(pred_binary):
            metrics.append(
                (
                    "f1_thresholded",
                    math.nan,
                    f"{continuous_dataset} threshold produced one predicted class",
                )
            )
        else:
            metrics.append(("f1_thresholded", float(f1_score(binary, pred_binary, zero_division=0)), ""))

    if np.unique(continuous).size < 2 or not has_two_classes(binary):
        metrics.append(("spearman", math.nan, "Spearman undefined because one input has fewer than two values"))
    else:
        value, note = safe_metric(lambda x, y: spearmanr(x, y).statistic, continuous, binary)
        metrics.append(("spearman", value, note))
    return metrics


def compute_protein_pair_metrics(
    protein_labels_a: np.ndarray,
    protein_labels_b: np.ndarray,
) -> list[tuple[str, float, str]]:
    if len(protein_labels_a) == 0:
        return [("protein_no_metric", math.nan, "no comparable proteins")]
    if len(protein_labels_a) < 2:
        note = "protein-level correlation undefined because fewer than two proteins overlap"
        return [("protein_spearman", math.nan, note), ("protein_pearson", math.nan, note)]
    if np.unique(protein_labels_a).size < 2 or np.unique(protein_labels_b).size < 2:
        note = "protein-level correlation undefined because one annotation has fewer than two values"
        return [("protein_spearman", math.nan, note), ("protein_pearson", math.nan, note)]

    metrics = []
    value, note = safe_metric(
        lambda x, y: spearmanr(x, y).statistic,
        protein_labels_a,
        protein_labels_b,
    )
    metrics.append(("protein_spearman", value, note))
    value, note = safe_metric(
        lambda x, y: pearsonr(x, y).statistic,
        protein_labels_a,
        protein_labels_b,
    )
    metrics.append(("protein_pearson", value, note))
    diff = protein_labels_a - protein_labels_b
    metrics.append(("protein_mae", float(np.mean(np.abs(diff))), ""))
    metrics.append(("protein_rmse", float(np.sqrt(np.mean(diff * diff))), ""))
    return metrics


def format_value(value: float) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def make_summary_row(
    dataset_a: str,
    dataset_b: str,
    match_mode: str,
    n_proteins_overlap: int,
    n_residues_compared: int,
    metric: str,
    value: float,
    threshold: float,
    notes: str,
) -> dict[str, str | int | float]:
    return {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "match_mode": match_mode,
        "n_proteins_overlap": n_proteins_overlap,
        "n_residues_compared": n_residues_compared,
        "annotation_type_a": annotation_type(dataset_a),
        "annotation_type_b": annotation_type(dataset_b),
        "metric": metric,
        "value": value,
        "threshold_used": threshold,
        "notes": notes,
    }


def write_summary_csv(rows: list[dict[str, str | int | float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["value"] = format_value(float(row["value"]))
            writer.writerow(output)


def write_details_csv(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/annotation_ceiling"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--min-residues", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    udonpred_dir = args.udonpred_dir.resolve()
    output_dir = args.output_dir.resolve()

    records_by_dataset: dict[str, list[ProteinRecord]] = {}
    missing: list[str] = []
    for dataset in args.datasets:
        try:
            records_by_dataset[dataset] = load_dataset_records(udonpred_dir, dataset)
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing and not records_by_dataset:
        raise SystemExit("No datasets could be loaded:\n" + "\n".join(missing))
    if missing:
        print("Skipping missing datasets:", file=sys.stderr)
        for message in missing:
            print(f"  {message}", file=sys.stderr)

    summary_rows: list[dict[str, str | int | float]] = []
    detail_rows: list[dict[str, str | int]] = []

    for dataset_a, dataset_b in combinations(records_by_dataset, 2):
        matches = find_overlaps(
            records_by_dataset[dataset_a],
            records_by_dataset[dataset_b],
            args.min_residues,
        )
        match_mode = pair_match_mode(matches)
        n_residues = sum(match.n_residues_compared for match in matches)

        if args.verbose:
            print(
                f"{dataset_a} vs {dataset_b}: {len(matches)} proteins, "
                f"{n_residues} residues, match={match_mode}"
            )

        if not matches:
            summary_rows.append(
                make_summary_row(
                    dataset_a,
                    dataset_b,
                    match_mode,
                    0,
                    0,
                    "no_overlap",
                    math.nan,
                    args.threshold,
                    "no exact ID or exact sequence overlap with enough comparable residues",
                )
            )
            continue

        for match in matches:
            detail_rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "protein_id_a": match.record_a.protein_id,
                    "protein_id_b": match.record_b.protein_id,
                    "sequence_length": len(match.record_a.sequence),
                    "n_residues_compared": match.n_residues_compared,
                    "match_mode": match.match_mode,
                }
            )

        labels_a, labels_b = flatten_matched_labels(matches, dataset_a, dataset_b)
        protein_labels_a, protein_labels_b = flatten_matched_protein_labels(
            matches,
            dataset_a,
            dataset_b,
        )
        metrics = [
            *compute_pair_metrics(labels_a, labels_b, dataset_a, dataset_b, args.threshold),
            *compute_protein_pair_metrics(protein_labels_a, protein_labels_b),
        ]
        base_notes = []
        if dataset_a == "plddt" or dataset_b == "plddt":
            base_notes.append("pLDDT converted to disorder score as 1 - pLDDT / 100")
        if dataset_a in NEGATED_DATASETS - {"plddt"} or dataset_b in NEGATED_DATASETS - {"plddt"}:
            base_notes.append("CheZOD labels negated following existing project convention")
        for metric, value, metric_note in metrics:
            notes = "; ".join(note for note in [*base_notes, metric_note] if note)
            summary_rows.append(
                make_summary_row(
                    dataset_a,
                    dataset_b,
                    match_mode,
                    len(matches),
                    int(len(labels_a)),
                    metric,
                    value,
                    args.threshold,
                    notes,
                )
            )

    write_summary_csv(summary_rows, output_dir / "annotation_ceiling_summary.csv")
    write_details_csv(detail_rows, output_dir / "overlap_details.csv")
    with (output_dir / "annotation_ceiling_summary.json").open("w") as handle:
        json.dump(json_ready(summary_rows), handle, indent=2)

    print(f"Wrote {output_dir / 'annotation_ceiling_summary.csv'}")
    print(f"Wrote {output_dir / 'annotation_ceiling_summary.json'}")
    print(f"Wrote {output_dir / 'overlap_details.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
