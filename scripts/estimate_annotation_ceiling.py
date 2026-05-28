#!/usr/bin/env python3
"""Estimate inter-annotation agreement between UdonPred evaluation datasets.

This script compares real residue annotations for proteins that occur in more
than one evaluation dataset. Exact ID/sequence matches are always evaluated;
optional MMseqs2 local-alignment matches can add identity-thresholded ceiling
levels. It is an approximate experimental ceiling for disorder-prediction
performance, not a shuffled-label or model benchmark.

Run from the repository root:

    python scripts/estimate_annotation_ceiling.py --output-dir results/annotation_ceiling
    python scripts/estimate_annotation_ceiling.py --use-mmseqs --output-dir results/annotation_ceiling_mmseqs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory

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
    "comparison_level",
    "match_mode",
    "min_identity",
    "min_coverage",
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
    "comparison_level",
    "protein_id_a",
    "protein_id_b",
    "sequence_length",
    "n_residues_compared",
    "match_mode",
    "percent_identity",
    "query_coverage",
    "target_coverage",
    "alignment_length",
]


@dataclass(frozen=True)
class MatchedProtein:
    record_a: ProteinRecord
    record_b: ProteinRecord
    match_mode: str
    n_residues_compared: int
    aligned_indices: tuple[tuple[int, int], ...] | None = None
    percent_identity: float | None = None
    query_coverage: float | None = None
    target_coverage: float | None = None
    alignment_length: int | None = None


@dataclass(frozen=True)
class MmseqsHit:
    record_a_index: int
    record_b_index: int
    percent_identity: float
    query_coverage: float
    target_coverage: float
    alignment_length: int
    bit_score: float
    aligned_indices: tuple[tuple[int, int], ...]


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


def comparable_aligned_indices(
    record_a: ProteinRecord,
    record_b: ProteinRecord,
    aligned_indices: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    valid_a = valid_mask(record_a.labels)
    valid_b = valid_mask(record_b.labels)
    comparable = []
    for index_a, index_b in aligned_indices:
        if index_a >= len(valid_a) or index_b >= len(valid_b):
            continue
        if valid_a[index_a] and valid_b[index_b]:
            comparable.append((index_a, index_b))
    return tuple(comparable)


def build_mmseqs_match(
    record_a: ProteinRecord,
    record_b: ProteinRecord,
    hit: MmseqsHit,
    min_residues: int,
) -> MatchedProtein | None:
    aligned_indices = comparable_aligned_indices(record_a, record_b, hit.aligned_indices)
    n_residues = len(aligned_indices)
    if n_residues < min_residues:
        return None
    return MatchedProtein(
        record_a=record_a,
        record_b=record_b,
        match_mode="mmseqs",
        n_residues_compared=n_residues,
        aligned_indices=aligned_indices,
        percent_identity=hit.percent_identity,
        query_coverage=hit.query_coverage,
        target_coverage=hit.target_coverage,
        alignment_length=hit.alignment_length,
    )


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


def fasta_header(dataset: str, index: int) -> str:
    return f"{dataset}__{index}"


def write_records_fasta(records: list[ProteinRecord], dataset: str, path: Path) -> dict[str, int]:
    header_to_index = {}
    with path.open("w") as handle:
        for index, record in enumerate(records):
            header = fasta_header(dataset, index)
            header_to_index[header] = index
            handle.write(f">{header}\n")
            sequence = record.sequence.replace("\n", "").replace("\r", "")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    return header_to_index


def aligned_residue_pairs(
    query_alignment: str,
    target_alignment: str,
    query_start: int,
    target_start: int,
) -> tuple[tuple[int, int], ...]:
    """Return zero-based residue index pairs from MMseqs aligned strings."""
    query_index = query_start - 1
    target_index = target_start - 1
    pairs = []
    for query_residue, target_residue in zip(query_alignment, target_alignment):
        query_has_residue = query_residue != "-"
        target_has_residue = target_residue != "-"
        if query_has_residue and target_has_residue:
            pairs.append((query_index, target_index))
        if query_has_residue:
            query_index += 1
        if target_has_residue:
            target_index += 1
    return tuple(pairs)


def normalized_percent_identity(value: str) -> float:
    identity = float(value)
    if identity <= 1.0:
        return identity * 100.0
    return identity


def run_mmseqs_search(
    records_a: list[ProteinRecord],
    records_b: list[ProteinRecord],
    dataset_a: str,
    dataset_b: str,
    mmseqs_binary: str,
    min_identity: float,
    threads: int,
) -> list[MmseqsHit]:
    if shutil.which(mmseqs_binary) is None:
        raise FileNotFoundError(
            f"MMseqs binary '{mmseqs_binary}' was not found. Install MMseqs2 or pass --mmseqs-binary."
        )

    fields = [
        "query",
        "target",
        "pident",
        "alnlen",
        "qstart",
        "tstart",
        "qlen",
        "tlen",
        "bits",
        "qaln",
        "taln",
    ]
    with TemporaryDirectory(prefix="annotation_ceiling_mmseqs_") as tmpdir:
        tmp_path = Path(tmpdir)
        query_fasta = tmp_path / "query.fasta"
        target_fasta = tmp_path / "target.fasta"
        output_tsv = tmp_path / "matches.tsv"
        search_tmp = tmp_path / "tmp"
        query_lookup = write_records_fasta(records_a, dataset_a, query_fasta)
        target_lookup = write_records_fasta(records_b, dataset_b, target_fasta)

        command = [
            mmseqs_binary,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(output_tsv),
            str(search_tmp),
            "--min-seq-id",
            f"{min_identity / 100.0:.4f}",
            "--format-output",
            ",".join(fields),
            "--threads",
            str(threads),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "MMseqs search failed:\n"
                f"command: {' '.join(command)}\n"
                f"stdout: {completed.stdout}\n"
                f"stderr: {completed.stderr}"
            )
        if not output_tsv.exists():
            return []

        hits = []
        with output_tsv.open() as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != len(fields):
                    continue
                query, target, pident, alnlen, qstart, tstart, qlen, tlen, bits, qaln, taln = parts
                if query not in query_lookup or target not in target_lookup:
                    continue
                aligned_indices = aligned_residue_pairs(qaln, taln, int(qstart), int(tstart))
                query_aligned = sum(1 for residue in qaln if residue != "-")
                target_aligned = sum(1 for residue in taln if residue != "-")
                qlen_value = int(qlen)
                tlen_value = int(tlen)
                hits.append(
                    MmseqsHit(
                        record_a_index=query_lookup[query],
                        record_b_index=target_lookup[target],
                        percent_identity=normalized_percent_identity(pident),
                        query_coverage=query_aligned / qlen_value if qlen_value else 0.0,
                        target_coverage=target_aligned / tlen_value if tlen_value else 0.0,
                        alignment_length=int(alnlen),
                        bit_score=float(bits),
                        aligned_indices=aligned_indices,
                    )
                )
    return hits


def select_best_mmseqs_hits(hits: list[MmseqsHit]) -> list[MmseqsHit]:
    """Keep one best target per query to avoid double-counting homolog families."""
    best_by_query: dict[int, MmseqsHit] = {}
    for hit in hits:
        current = best_by_query.get(hit.record_a_index)
        sort_key = (hit.bit_score, hit.percent_identity, len(hit.aligned_indices))
        current_key = (
            current.bit_score,
            current.percent_identity,
            len(current.aligned_indices),
        ) if current else None
        if current is None or sort_key > current_key:
            best_by_query[hit.record_a_index] = hit
    return list(best_by_query.values())


def find_mmseqs_overlaps(
    records_a: list[ProteinRecord],
    records_b: list[ProteinRecord],
    hits: list[MmseqsHit],
    min_identity: float,
    min_coverage: float,
    min_residues: int,
    top_hit_only: bool,
) -> list[MatchedProtein]:
    filtered_hits = [
        hit
        for hit in hits
        if hit.percent_identity + 1e-9 >= min_identity
        and hit.query_coverage >= min_coverage
        and hit.target_coverage >= min_coverage
    ]
    if top_hit_only:
        filtered_hits = select_best_mmseqs_hits(filtered_hits)

    matches = []
    used: set[tuple[int, int]] = set()
    for hit in sorted(
        filtered_hits,
        key=lambda item: (-item.percent_identity, -item.bit_score, item.record_a_index, item.record_b_index),
    ):
        key = (hit.record_a_index, hit.record_b_index)
        if key in used:
            continue
        match = build_mmseqs_match(
            records_a[hit.record_a_index],
            records_b[hit.record_b_index],
            hit,
            min_residues,
        )
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
        directed_a = labels_in_common_direction(raw_a, dataset_a)
        directed_b = labels_in_common_direction(raw_b, dataset_b)
        if match.aligned_indices is None:
            mask = comparable_mask(raw_a, raw_b)
            labels_a.append(directed_a[: len(mask)][mask])
            labels_b.append(directed_b[: len(mask)][mask])
        else:
            indices_a = [index_a for index_a, _ in match.aligned_indices]
            indices_b = [index_b for _, index_b in match.aligned_indices]
            labels_a.append(directed_a[indices_a])
            labels_b.append(directed_b[indices_b])
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
        directed_a = labels_in_common_direction(raw_a, dataset_a)
        directed_b = labels_in_common_direction(raw_b, dataset_b)
        if match.aligned_indices is None:
            mask = comparable_mask(raw_a, raw_b)
            if not np.any(mask):
                continue
            protein_labels_a.append(float(np.mean(directed_a[: len(mask)][mask])))
            protein_labels_b.append(float(np.mean(directed_b[: len(mask)][mask])))
        else:
            indices_a = [index_a for index_a, _ in match.aligned_indices]
            indices_b = [index_b for _, index_b in match.aligned_indices]
            if not indices_a:
                continue
            protein_labels_a.append(float(np.mean(directed_a[indices_a])))
            protein_labels_b.append(float(np.mean(directed_b[indices_b])))
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
    comparison_level: str,
    match_mode: str,
    min_identity: float | None,
    min_coverage: float | None,
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
        "comparison_level": comparison_level,
        "match_mode": match_mode,
        "min_identity": min_identity if min_identity is not None else "",
        "min_coverage": min_coverage if min_coverage is not None else "",
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


def primary_metric_for_types(annotation_type_a: str, annotation_type_b: str) -> str:
    if annotation_type_a == "continuous" and annotation_type_b == "continuous":
        return "spearman"
    if annotation_type_a == "binary" and annotation_type_b == "binary":
        return "mcc"
    return "auroc"


def write_ceiling_plots(rows: list[dict[str, str | int | float]], output_dir: Path) -> None:
    """Write exact-match and MMseqs identity-threshold comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        print(f"Skipping plots because plotting dependencies are missing: {exc}", file=sys.stderr)
        return

    if not rows:
        return

    dataset_order = DATASETS
    summary = pd.DataFrame(rows)
    summary["pair"] = summary["dataset_a"] + " vs " + summary["dataset_b"]
    summary["value"] = pd.to_numeric(summary["value"], errors="coerce")
    summary["n_proteins_overlap"] = pd.to_numeric(summary["n_proteins_overlap"], errors="coerce")
    summary["n_residues_compared"] = pd.to_numeric(summary["n_residues_compared"], errors="coerce")
    summary["min_identity"] = pd.to_numeric(summary["min_identity"], errors="coerce")

    def matrix_for(level_rows, value_column: str):
        matrix = pd.DataFrame(np.nan, index=dataset_order, columns=dataset_order)
        pair_rows = level_rows.drop_duplicates(["dataset_a", "dataset_b"])
        for row in pair_rows.itertuples(index=False):
            value = getattr(row, value_column)
            matrix.loc[row.dataset_a, row.dataset_b] = value
            matrix.loc[row.dataset_b, row.dataset_a] = value
        return matrix

    exact = summary[summary["comparison_level"] == "exact"].copy()
    if not exact.empty:
        protein_overlap = matrix_for(exact, "n_proteins_overlap")
        residue_overlap = matrix_for(exact, "n_residues_compared")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(
            protein_overlap,
            annot=True,
            fmt=".0f",
            cmap="mako",
            mask=protein_overlap.isna(),
            ax=axes[0],
        )
        axes[0].set_title("Exact Overlapping Proteins")
        sns.heatmap(
            residue_overlap,
            annot=True,
            fmt=".0f",
            cmap="mako",
            mask=residue_overlap.isna(),
            ax=axes[1],
        )
        axes[1].set_title("Exact Residues Compared")
        plt.tight_layout()
        fig.savefig(output_dir / "ceiling_overlap_sizes.png", dpi=200)
        plt.close(fig)

        spearman = exact[
            (exact["annotation_type_a"] == "continuous")
            & (exact["annotation_type_b"] == "continuous")
            & (exact["metric"] == "spearman")
            & exact["value"].notna()
        ]
        if not spearman.empty:
            spearman_matrix = matrix_for(spearman, "value")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                spearman_matrix,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                vmin=-1,
                vmax=1,
                mask=spearman_matrix.isna(),
                ax=ax,
            )
            ax.set_title("Exact Continuous-Continuous Ceiling (Spearman)")
            plt.tight_layout()
            fig.savefig(output_dir / "ceiling_continuous_spearman.png", dpi=200)
            plt.close(fig)

        disprot = exact[
            ((exact["dataset_a"] == "disprot") | (exact["dataset_b"] == "disprot"))
            & exact["metric"].isin(["auroc", "average_precision", "f1_thresholded", "spearman"])
            & exact["value"].notna()
        ].copy()
        if not disprot.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * disprot["pair"].nunique())))
            sns.barplot(data=disprot, x="value", y="pair", hue="metric", ax=ax)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Metric value")
            ax.set_ylabel("Dataset pair")
            ax.set_title("Exact DisProt Annotation Agreement")
            ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            fig.savefig(output_dir / "ceiling_disprot_agreement.png", dpi=200)
            plt.close(fig)

    mmseqs = summary[summary["comparison_level"].astype(str).str.startswith("mmseqs_")].copy()
    if mmseqs.empty:
        return

    mmseqs_pair_rows = mmseqs.drop_duplicates(["comparison_level", "dataset_a", "dataset_b"]).copy()
    mmseqs_pair_rows["identity_threshold"] = mmseqs_pair_rows["min_identity"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(
        data=mmseqs_pair_rows,
        x="identity_threshold",
        y="n_proteins_overlap",
        hue="pair",
        marker="o",
        ax=axes[0],
    )
    axes[0].invert_xaxis()
    axes[0].set_title("MMseqs Protein Overlap By Identity Threshold")
    axes[0].set_xlabel("Minimum identity (%)")
    axes[0].set_ylabel("Proteins compared")
    sns.lineplot(
        data=mmseqs_pair_rows,
        x="identity_threshold",
        y="n_residues_compared",
        hue="pair",
        marker="o",
        ax=axes[1],
    )
    axes[1].invert_xaxis()
    axes[1].set_title("MMseqs Residue Overlap By Identity Threshold")
    axes[1].set_xlabel("Minimum identity (%)")
    axes[1].set_ylabel("Residues compared")
    for ax in axes:
        ax.legend(title="Pair", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / "ceiling_mmseqs_overlap_by_identity.png", dpi=200)
    plt.close(fig)

    primary_rows = []
    for row in mmseqs_pair_rows.itertuples(index=False):
        pair_metrics = mmseqs[
            (mmseqs["comparison_level"] == row.comparison_level)
            & (mmseqs["dataset_a"] == row.dataset_a)
            & (mmseqs["dataset_b"] == row.dataset_b)
        ]
        if pair_metrics.empty:
            continue
        metric = primary_metric_for_types(
            pair_metrics.iloc[0]["annotation_type_a"],
            pair_metrics.iloc[0]["annotation_type_b"],
        )
        metric_row = pair_metrics[pair_metrics["metric"] == metric]
        value = np.nan if metric_row.empty else metric_row.iloc[0]["value"]
        primary_rows.append(
            {
                "pair": row.pair,
                "comparison_level": row.comparison_level,
                "identity_threshold": row.identity_threshold,
                "metric": metric,
                "value": value,
                "n_residues_compared": row.n_residues_compared,
            }
        )
    primary = pd.DataFrame(primary_rows)
    primary = primary[primary["value"].notna()]
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=primary,
            x="identity_threshold",
            y="value",
            hue="pair",
            style="metric",
            marker="o",
            ax=ax,
        )
        ax.invert_xaxis()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("MMseqs Primary Annotation Agreement By Identity Threshold")
        ax.set_xlabel("Minimum identity (%)")
        ax.set_ylabel("Primary agreement")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(output_dir / "ceiling_mmseqs_primary_agreement_by_identity.png", dpi=200)
        plt.close(fig)

    continuous = mmseqs[
        (mmseqs["annotation_type_a"] == "continuous")
        & (mmseqs["annotation_type_b"] == "continuous")
        & (mmseqs["metric"] == "spearman")
        & mmseqs["value"].notna()
    ].copy()
    if not continuous.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=continuous,
            x="min_identity",
            y="value",
            hue="pair",
            marker="o",
            ax=ax,
        )
        ax.invert_xaxis()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("MMseqs Continuous-Continuous Spearman By Identity Threshold")
        ax.set_xlabel("Minimum identity (%)")
        ax.set_ylabel("Spearman")
        ax.legend(title="Pair", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(output_dir / "ceiling_mmseqs_continuous_spearman_by_identity.png", dpi=200)
        plt.close(fig)

    disprot = mmseqs[
        ((mmseqs["dataset_a"] == "disprot") | (mmseqs["dataset_b"] == "disprot"))
        & mmseqs["metric"].isin(["auroc", "average_precision", "f1_thresholded", "spearman"])
        & mmseqs["value"].notna()
    ].copy()
    if not disprot.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=disprot,
            x="min_identity",
            y="value",
            hue="pair",
            style="metric",
            marker="o",
            ax=ax,
        )
        ax.invert_xaxis()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("MMseqs DisProt Agreement By Identity Threshold")
        ax.set_xlabel("Minimum identity (%)")
        ax.set_ylabel("Metric value")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(output_dir / "ceiling_mmseqs_disprot_agreement_by_identity.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/annotation_ceiling"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--min-residues", type=int, default=10)
    parser.add_argument("--use-mmseqs", action="store_true", help="Add MMseqs2 local-alignment ceiling levels.")
    parser.add_argument("--mmseqs-binary", default="mmseqs")
    parser.add_argument(
        "--mmseqs-identities",
        type=float,
        nargs="+",
        default=[100.0, 98.0, 95.0, 90.0, 85.0, 80.0],
        help="Percent-identity cutoffs to evaluate when --use-mmseqs is set.",
    )
    parser.add_argument(
        "--mmseqs-min-coverage",
        type=float,
        default=0.8,
        help="Minimum aligned coverage required on both proteins for MMseqs hits.",
    )
    parser.add_argument("--mmseqs-threads", type=int, default=1)
    parser.add_argument(
        "--mmseqs-all-hits",
        action="store_true",
        help="Use all passing MMseqs hits instead of only the best target per query.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip writing ceiling PNG plots.")
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
        pair_evaluations: list[tuple[str, list[MatchedProtein], float | None, float | None]] = [
            (
                "exact",
                find_overlaps(
                    records_by_dataset[dataset_a],
                    records_by_dataset[dataset_b],
                    args.min_residues,
                ),
                None,
                None,
            )
        ]

        if args.use_mmseqs:
            identities = sorted(set(args.mmseqs_identities), reverse=True)
            if not identities:
                raise SystemExit("--mmseqs-identities must contain at least one threshold")
            min_identity = min(identities)
            mmseqs_hits = run_mmseqs_search(
                records_by_dataset[dataset_a],
                records_by_dataset[dataset_b],
                dataset_a,
                dataset_b,
                args.mmseqs_binary,
                min_identity,
                args.mmseqs_threads,
            )
            for identity in identities:
                pair_evaluations.append(
                    (
                        f"mmseqs_{identity:g}",
                        find_mmseqs_overlaps(
                            records_by_dataset[dataset_a],
                            records_by_dataset[dataset_b],
                            mmseqs_hits,
                            identity,
                            args.mmseqs_min_coverage,
                            args.min_residues,
                            top_hit_only=not args.mmseqs_all_hits,
                        ),
                        identity,
                        args.mmseqs_min_coverage,
                    )
                )

        for comparison_level, matches, min_identity, min_coverage in pair_evaluations:
            match_mode = pair_match_mode(matches)
            n_residues = sum(match.n_residues_compared for match in matches)

            if args.verbose:
                print(
                    f"{dataset_a} vs {dataset_b} [{comparison_level}]: {len(matches)} proteins, "
                    f"{n_residues} residues, match={match_mode}"
                )

            if not matches:
                overlap_note = (
                    "no exact ID or exact sequence overlap with enough comparable residues"
                    if comparison_level == "exact"
                    else "no MMseqs hit passed identity, coverage, and comparable-residue filters"
                )
                summary_rows.append(
                    make_summary_row(
                        dataset_a,
                        dataset_b,
                        comparison_level,
                        match_mode,
                        min_identity,
                        min_coverage,
                        0,
                        0,
                        "no_overlap",
                        math.nan,
                        args.threshold,
                        overlap_note,
                    )
                )
                continue

            for match in matches:
                detail_rows.append(
                    {
                        "dataset_a": dataset_a,
                        "dataset_b": dataset_b,
                        "comparison_level": comparison_level,
                        "protein_id_a": match.record_a.protein_id,
                        "protein_id_b": match.record_b.protein_id,
                        "sequence_length": len(match.record_a.sequence),
                        "n_residues_compared": match.n_residues_compared,
                        "match_mode": match.match_mode,
                        "percent_identity": format_value(match.percent_identity)
                        if match.percent_identity is not None
                        else "",
                        "query_coverage": format_value(match.query_coverage)
                        if match.query_coverage is not None
                        else "",
                        "target_coverage": format_value(match.target_coverage)
                        if match.target_coverage is not None
                        else "",
                        "alignment_length": match.alignment_length or "",
                    }
                )

            labels_a, labels_b = flatten_matched_labels(matches, dataset_a, dataset_b)
            metrics = compute_pair_metrics(labels_a, labels_b, dataset_a, dataset_b, args.threshold)
            base_notes = []
            if dataset_a == "plddt" or dataset_b == "plddt":
                base_notes.append("pLDDT converted to disorder score as 1 - pLDDT / 100")
            if dataset_a in NEGATED_DATASETS - {"plddt"} or dataset_b in NEGATED_DATASETS - {"plddt"}:
                base_notes.append("CheZOD labels negated following existing project convention")
            if comparison_level.startswith("mmseqs"):
                base_notes.append(
                    "MMseqs local alignment; only aligned residue pairs without gaps and with valid labels compared"
                )
            for metric, value, metric_note in metrics:
                notes = "; ".join(note for note in [*base_notes, metric_note] if note)
                summary_rows.append(
                    make_summary_row(
                        dataset_a,
                        dataset_b,
                        comparison_level,
                        match_mode,
                        min_identity,
                        min_coverage,
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
    if not args.no_plots:
        write_ceiling_plots(summary_rows, output_dir)

    print(f"Wrote {output_dir / 'annotation_ceiling_summary.csv'}")
    print(f"Wrote {output_dir / 'annotation_ceiling_summary.json'}")
    print(f"Wrote {output_dir / 'overlap_details.csv'}")
    if not args.no_plots:
        print(f"Wrote ceiling plots to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    except KeyboardInterrupt:
        sys.exit(130)
