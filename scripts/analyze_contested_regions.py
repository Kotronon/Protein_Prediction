#!/usr/bin/env python3
"""Build contested-region tables and presentation figures.

The script works with UdonPred-style CAID prediction folders. For the current
branch it can analyse existing cross-dataset prediction outputs such as:

    results/udonpred_matrix/predictions/{train_dataset}_{target_dataset}/*.caid

It also accepts a generic layout:

    prediction_root/{predictor_name}/*.caid

Scores are converted so larger values mean more disorder, calibrated per
predictor, summarized in sliding windows, classified by disagreement type, and
written as CSVs plus slide-ready plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
NEGATED_PREDICTORS = {"chezod", "plddt"}
MASK_VALUE = 999.0
WINDOW_FEATURE_COLUMNS = [
    "hydropathy",
    "percent_hydrophobic",
    "net_charge",
    "positive_fraction",
    "negative_fraction",
    "frac_proline",
    "frac_glycine",
    "frac_pro_gly",
    "frac_charged",
    "frac_polar",
    "frac_aromatic",
    "frac_nonstandard",
    "low_complexity",
    "longest_hydrophobic_run",
    "tmh_like_score",
    "signal_peptide_like_score",
    "coil_like_score",
    "nors_like_score",
    "possible_more_score",
]
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
STANDARD_AA = set(HYDROPATHY)
POLAR = set("STNQCY")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWYV")
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


@dataclass(frozen=True)
class PredictionRecord:
    protein_id: str
    sequence: str
    scores: np.ndarray


@dataclass(frozen=True)
class LabelRecord:
    protein_id: str
    sequence: str
    labels: np.ndarray


def read_caid(path: Path) -> PredictionRecord:
    with path.open() as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Invalid CAID file: {path}")
    protein_id = lines[0].lstrip(">")
    residues: list[str] = []
    scores: list[float] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        residues.append(parts[1])
        scores.append(float(parts[2]))
    return PredictionRecord(protein_id, "".join(residues), np.asarray(scores, dtype=np.float64))


def read_jsonl_labels(path: Path) -> dict[str, LabelRecord]:
    records: dict[str, LabelRecord] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            protein_id = str(item["id"])
            sequence = str(item["x_0"])
            labels = np.asarray(item["y"], dtype=np.float64)
            records[protein_id] = LabelRecord(protein_id, sequence, labels)
    return records


def labels_in_disorder_direction(labels: np.ndarray, dataset: str) -> np.ndarray:
    labels = labels.astype(np.float64, copy=True)
    if dataset == "plddt":
        mask = valid_mask(labels)
        converted = labels.copy()
        converted[mask] = 1.0 - (converted[mask] / 100.0)
        return converted
    if dataset == "chezod":
        return -labels
    return labels


def valid_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != MASK_VALUE)


def normalize_protein_id(protein_id: str) -> str:
    """Match the TriZOD ID conversion used in run_udonpred_matrix.py."""
    if len(protein_id) >= 7 and protein_id.isdigit():
        base_len = len(protein_id) - 3
        return protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])
    return protein_id


def discover_prediction_dirs(
    prediction_root: Path,
    layout: str,
    target_dataset: str | None,
    exclude_predictors: set[str],
) -> dict[str, Path]:
    if layout == "udonpred_matrix":
        if not target_dataset:
            raise ValueError("--target-dataset is required for udonpred_matrix layout")
        dirs: dict[str, Path] = {}
        for predictor in DATASETS:
            if predictor in exclude_predictors:
                continue
            path = prediction_root / f"{predictor}_{target_dataset}"
            if path.exists():
                dirs[predictor] = path
        return dirs

    dirs = {
        path.name: path
        for path in sorted(prediction_root.iterdir())
        if path.is_dir() and path.name not in exclude_predictors and any(path.glob("*.caid"))
    }
    return dirs


def load_predictions(prediction_dirs: dict[str, Path]) -> dict[str, dict[str, PredictionRecord]]:
    predictions: dict[str, dict[str, PredictionRecord]] = {}
    for predictor, directory in prediction_dirs.items():
        records: dict[str, PredictionRecord] = {}
        for path in sorted(directory.glob("*.caid")):
            record = read_caid(path)
            records[normalize_protein_id(record.protein_id)] = record
        if records:
            predictions[predictor] = records
    if len(predictions) < 2:
        raise ValueError("Need at least two predictors with CAID files")
    return predictions


def common_protein_ids(predictions: dict[str, dict[str, PredictionRecord]]) -> list[str]:
    ids: set[str] | None = None
    for records in predictions.values():
        current = set(records)
        ids = current if ids is None else ids & current
    return sorted(ids or set())


def directed_scores(predictor: str, scores: np.ndarray) -> np.ndarray:
    return -scores if predictor in NEGATED_PREDICTORS else scores


def calibration_tables(
    predictions: dict[str, dict[str, PredictionRecord]],
    protein_ids: list[str],
) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    z_params: dict[str, tuple[float, float]] = {}
    sorted_scores: dict[str, np.ndarray] = {}
    for predictor, records in predictions.items():
        values = np.concatenate([directed_scores(predictor, records[pid].scores) for pid in protein_ids])
        values = values[np.isfinite(values)]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(std) or std == 0:
            std = 1.0
        z_params[predictor] = (mean, std)
        sorted_scores[predictor] = np.sort(values)
    return z_params, sorted_scores


def percentile_values(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    ranks = np.searchsorted(sorted_reference, values, side="right")
    return ranks / max(len(sorted_reference), 1)


def sequence_features(sequence: str) -> dict[str, float]:
    seq = sequence.upper()
    length = len(seq)
    if length == 0:
        return {column: np.nan for column in WINDOW_FEATURE_COLUMNS}
    counts = Counter(seq)
    standard = [aa for aa in seq if aa in STANDARD_AA]
    hydropathy = np.mean([HYDROPATHY[aa] for aa in standard]) if standard else np.nan
    hydrophobic_flags = [aa in HYDROPHOBIC for aa in seq]
    longest_hydrophobic_run = 0
    current_run = 0
    for flag in hydrophobic_flags:
        current_run = current_run + 1 if flag else 0
        longest_hydrophobic_run = max(longest_hydrophobic_run, current_run)
    percent_hydrophobic = sum(hydrophobic_flags) / length
    tmh_like_score = tmh_score(seq)
    signal_peptide_like_score = signal_peptide_score(seq)
    coil_like_score = coil_score(seq)
    nors_like_score = float(np.clip(coil_like_score * (1.0 - tmh_like_score), 0.0, 1.0))
    return {
        "hydropathy": float(hydropathy),
        "percent_hydrophobic": percent_hydrophobic,
        "net_charge": (sum(counts[aa] for aa in POSITIVE) - sum(counts[aa] for aa in NEGATIVE)) / length,
        "positive_fraction": sum(counts[aa] for aa in POSITIVE) / length,
        "negative_fraction": sum(counts[aa] for aa in NEGATIVE) / length,
        "frac_proline": counts["P"] / length,
        "frac_glycine": counts["G"] / length,
        "frac_pro_gly": (counts["P"] + counts["G"]) / length,
        "frac_charged": sum(counts[aa] for aa in POSITIVE | NEGATIVE) / length,
        "frac_polar": sum(counts[aa] for aa in POLAR) / length,
        "frac_aromatic": sum(counts[aa] for aa in AROMATIC) / length,
        "frac_nonstandard": sum(1 for aa in seq if aa not in STANDARD_AA) / length,
        "low_complexity": max(counts.values()) / length,
        "longest_hydrophobic_run": float(longest_hydrophobic_run),
        "tmh_like_score": tmh_like_score,
        "signal_peptide_like_score": signal_peptide_like_score,
        "coil_like_score": coil_like_score,
        "nors_like_score": nors_like_score,
        # Filled after predictor consensus is available. Keep the column stable.
        "possible_more_score": 0.0,
    }


def sliding_mean(values: list[float], width: int) -> list[float]:
    if not values:
        return []
    if len(values) <= width:
        return [float(np.mean(values))]
    return [float(np.mean(values[i : i + width])) for i in range(0, len(values) - width + 1)]


def tmh_score(sequence: str) -> float:
    """Hydrophobic-segment proxy inspired by lecture TMH controls."""
    seq = sequence.upper()
    kd = [HYDROPATHY.get(aa, 0.0) for aa in seq]
    hydrophobic = [1.0 if aa in HYDROPHOBIC else 0.0 for aa in seq]
    best = 0.0
    for width in (17, 19, 21, 23):
        for mean_kd, frac_hydro in zip(sliding_mean(kd, width), sliding_mean(hydrophobic, width)):
            kd_component = np.clip((mean_kd - 1.0) / 2.2, 0.0, 1.0)
            hydro_component = np.clip((frac_hydro - 0.45) / 0.35, 0.0, 1.0)
            best = max(best, float(0.55 * kd_component + 0.45 * hydro_component))
    return best


def signal_peptide_score(sequence: str) -> float:
    """Simple N-terminal signal-peptide-like proxy for weekly analysis."""
    seq = sequence.upper()
    nterm = seq[:30]
    if len(nterm) < 15:
        return 0.0
    hydrophobic_core = tmh_score(nterm[5:25])
    n_region_positive = sum(1 for aa in nterm[:8] if aa in POSITIVE) / 8.0
    return float(np.clip(0.75 * hydrophobic_core + 0.25 * n_region_positive, 0.0, 1.0))


def coil_score(sequence: str) -> float:
    """Transparent no-regular-secondary-structure proxy.

    Higher values indicate low helix/sheet propensity and more proline/glycine,
    echoing the lecture's NORS/no-regular-secondary-structure framing. This is
    not a replacement for a real secondary-structure predictor.
    """
    seq = sequence.upper()
    standard = [aa for aa in seq if aa in STANDARD_AA]
    if not standard:
        return 0.0
    helix = np.mean([HELIX_PROPENSITY[aa] for aa in standard])
    sheet = np.mean([SHEET_PROPENSITY[aa] for aa in standard])
    regular = (helix + sheet) / 2.0
    low_regular = np.clip((1.05 - regular) / 0.45, 0.0, 1.0)
    pro_gly = (seq.count("P") + seq.count("G")) / len(seq)
    charge = sum(1 for aa in seq if aa in POSITIVE | NEGATIVE) / len(seq)
    return float(np.clip(0.55 * low_regular + 0.25 * pro_gly + 0.20 * charge, 0.0, 1.0))


def nmr_suitability(features: dict[str, float]) -> float:
    hydropathy = features["hydropathy"]
    hydrophobic_penalty = 0.0 if not np.isfinite(hydropathy) else max(0.0, (hydropathy - 1.5) / 3.0)
    low_complexity_penalty = max(0.0, (features["low_complexity"] - 0.35) / 0.65)
    nonstandard_penalty = min(1.0, features["frac_nonstandard"] * 5.0)
    tmh_penalty = features.get("tmh_like_score", 0.0)
    signal_penalty = features.get("signal_peptide_like_score", 0.0)
    suitability = (
        1.0
        - 0.35 * hydrophobic_penalty
        - 0.20 * low_complexity_penalty
        - 0.15 * nonstandard_penalty
        - 0.20 * tmh_penalty
        - 0.10 * signal_penalty
    )
    return float(np.clip(suitability, 0.0, 1.0))


def classify_window(row: pd.Series) -> str:
    if row.get("tmh_like_score", 0.0) >= 0.65 or row.get("signal_peptide_like_score", 0.0) >= 0.65:
        return "tmh_or_signal_peptide_like_artifact"
    if row.get("possible_more_score", 0.0) >= 0.78 and row.get("mean_disagreement", 0.0) >= 0.65:
        return "possible_molecular_recognition_element"
    if row.get("nors_like_score", 0.0) >= 0.60 and row.get("window_length", 0) >= 30:
        return "nors_like_no_regular_structure"
    if row.get("pdbflex_delta_vs_median_z", 0.0) >= 1.0:
        return "pdbflex_high_outlier"
    if row.get("plddt_delta_vs_median_z", 0.0) >= 1.0:
        return "plddt_uncertainty_outlier"
    if abs(row.get("nmr_vs_curated_delta", 0.0)) >= 1.0:
        return "nmr_vs_curated_conflict"
    if row.get("n_predictors_above_consensus", 0) >= 2 and row.get("n_predictors_below_consensus", 0) >= 2:
        return "broad_multimodel_spread"
    return "general_predictor_spread"


def concept_family_values(row: dict[str, object], predictors: list[str]) -> dict[str, float]:
    families = {
        "nmr": ["trizod", "chezod"],
        "curated_or_derived_disorder": ["disprot", "softdis", "atlas"],
        "alphafold_confidence_proxy": ["plddt"],
        "structural_flexibility": ["pdbflex"],
    }
    values = {}
    for family, members in families.items():
        member_values = [float(row[f"{member}_mean_z"]) for member in members if member in predictors]
        if member_values:
            values[f"{family}_mean_z"] = float(np.mean(member_values))
    if values:
        family_scores = list(values.values())
        values["concept_family_spread_z"] = float(max(family_scores) - min(family_scores))
    else:
        values["concept_family_spread_z"] = 0.0
    return values


def possible_more_score(row: dict[str, object]) -> float:
    """Heuristic MoRE candidate score from lecture concepts.

    MoRE candidates should be disorder-relevant but not just low-complexity
    tails: near-boundary consensus, meaningful disagreement, not strongly TMH-
    like, and moderate sequence complexity.
    """
    consensus = float(row["consensus_percentile"])
    moderate_disorder = float(np.clip(1.0 - abs(consensus - 0.62) / 0.38, 0.0, 1.0))
    boundary = float(row["boundary_uncertainty"])
    disagreement = float(np.clip(row["mean_disagreement"] / 1.0, 0.0, 1.0))
    complexity_ok = float(np.clip(1.0 - max(0.0, row["low_complexity"] - 0.30) / 0.45, 0.0, 1.0))
    not_membrane = 1.0 - max(float(row["tmh_like_score"]), float(row["signal_peptide_like_score"]))
    pro_gly_ok = float(np.clip(1.0 - max(0.0, row["frac_pro_gly"] - 0.45) / 0.35, 0.0, 1.0))
    return float(
        np.clip(
            0.25 * moderate_disorder
            + 0.20 * boundary
            + 0.20 * disagreement
            + 0.15 * complexity_ok
            + 0.10 * not_membrane
            + 0.10 * pro_gly_ok,
            0.0,
            1.0,
        )
    )


def build_window_table(
    predictions: dict[str, dict[str, PredictionRecord]],
    protein_ids: list[str],
    z_params: dict[str, tuple[float, float]],
    sorted_scores: dict[str, np.ndarray],
    window_size: int,
    step: int,
    labels: dict[str, LabelRecord] | None,
    label_dataset: str | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    predictors = sorted(predictions)
    windows: list[dict[str, object]] = []
    tracks: dict[str, pd.DataFrame] = {}

    for protein_id in protein_ids:
        first = predictions[predictors[0]][protein_id]
        length = len(first.sequence)
        per_predictor_z: dict[str, np.ndarray] = {}
        per_predictor_pct: dict[str, np.ndarray] = {}
        per_predictor_raw: dict[str, np.ndarray] = {}

        for predictor in predictors:
            record = predictions[predictor][protein_id]
            if len(record.scores) != length:
                raise ValueError(f"Length mismatch for {protein_id} in {predictor}")
            raw = directed_scores(predictor, record.scores)
            mean, std = z_params[predictor]
            per_predictor_raw[predictor] = raw
            per_predictor_z[predictor] = (raw - mean) / std
            per_predictor_pct[predictor] = percentile_values(raw, sorted_scores[predictor])

        z_matrix = np.vstack([per_predictor_z[predictor] for predictor in predictors])
        pct_matrix = np.vstack([per_predictor_pct[predictor] for predictor in predictors])
        residue_disagreement = np.percentile(z_matrix, 75, axis=0) - np.percentile(z_matrix, 25, axis=0)
        consensus_pct = np.mean(pct_matrix, axis=0)
        median_z = np.median(z_matrix, axis=0)

        track = pd.DataFrame(
            {
                "protein_id": protein_id,
                "position": np.arange(1, length + 1),
                "residue": list(first.sequence),
                "residue_disagreement": residue_disagreement,
                "consensus_percentile": consensus_pct,
                "median_z": median_z,
                **{f"{predictor}_z": per_predictor_z[predictor] for predictor in predictors},
                **{f"{predictor}_raw_disorder": per_predictor_raw[predictor] for predictor in predictors},
            }
        )
        tracks[protein_id] = track

        directed_labels: np.ndarray | None = None
        label_mask: np.ndarray | None = None
        if labels and protein_id in labels and label_dataset:
            label_record = labels[protein_id]
            if len(label_record.labels) == length:
                directed_labels = labels_in_disorder_direction(label_record.labels, label_dataset)
                label_mask = valid_mask(label_record.labels)

        for start0 in range(0, max(length - window_size + 1, 1), step):
            end0 = min(start0 + window_size, length)
            if end0 - start0 < min(window_size, length):
                continue
            window_sequence = first.sequence[start0:end0]
            z_window = z_matrix[:, start0:end0]
            pct_window = pct_matrix[:, start0:end0]
            disagreement = residue_disagreement[start0:end0]
            consensus = consensus_pct[start0:end0]
            predictor_mean_z = z_window.mean(axis=1)
            order = np.argsort(predictor_mean_z)
            median_predictor_z = float(np.median(predictor_mean_z))
            top_predictor = predictors[int(order[-1])]
            bottom_predictor = predictors[int(order[0])]
            features = sequence_features(window_sequence)

            row: dict[str, object] = {
                "protein_id": protein_id,
                "start": start0 + 1,
                "end": end0,
                "window_length": end0 - start0,
                "sequence": window_sequence,
                "mean_disagreement": float(np.mean(disagreement)),
                "max_disagreement": float(np.max(disagreement)),
                "consensus_percentile": float(np.mean(consensus)),
                "boundary_uncertainty": float(np.clip(1.0 - 2.0 * abs(np.mean(consensus) - 0.5), 0.0, 1.0)),
                "top_predictor": top_predictor,
                "bottom_predictor": bottom_predictor,
                "top_minus_bottom_z": float(predictor_mean_z[order[-1]] - predictor_mean_z[order[0]]),
                "n_predictors_above_consensus": int(np.sum(predictor_mean_z > median_predictor_z + 0.5)),
                "n_predictors_below_consensus": int(np.sum(predictor_mean_z < median_predictor_z - 0.5)),
                "nmr_suitability": nmr_suitability(features),
            }
            for predictor, value in zip(predictors, predictor_mean_z):
                row[f"{predictor}_mean_z"] = float(value)
                row[f"{predictor}_delta_vs_median_z"] = float(value - median_predictor_z)

            row.update(concept_family_values(row, predictors))
            nmr_values = [row[f"{p}_mean_z"] for p in ("trizod", "chezod") if p in predictors]
            curated_values = [row[f"{p}_mean_z"] for p in ("disprot", "softdis", "atlas") if p in predictors]
            row["nmr_vs_curated_delta"] = (
                float(np.mean(nmr_values) - np.mean(curated_values))
                if nmr_values and curated_values
                else 0.0
            )
            row.update(features)
            row["possible_more_score"] = possible_more_score(row)
            if "plddt" in predictors:
                plddt_delta = float(row.get("plddt_delta_vs_median_z", 0.0))
                consensus_value = float(row["consensus_percentile"])
                if plddt_delta >= 1.0 and consensus_value < 0.45:
                    row["plddt_confidence_class"] = "plddt_low_confidence_without_disorder_consensus"
                elif plddt_delta >= 1.0:
                    row["plddt_confidence_class"] = "plddt_low_confidence_with_disorder_consensus"
                elif plddt_delta <= -1.0 and consensus_value > 0.55:
                    row["plddt_confidence_class"] = "plddt_high_confidence_against_disorder_consensus"
                else:
                    row["plddt_confidence_class"] = "plddt_not_outlier"
            if directed_labels is not None and label_mask is not None:
                mask = label_mask[start0:end0]
                if np.any(mask):
                    label_values = directed_labels[start0:end0][mask]
                    row["label_mean"] = float(np.mean(label_values))
                    row["label_coverage"] = float(np.mean(mask))
                    if label_dataset == "disprot":
                        row["label_positive_rate"] = float(np.mean(label_values > 0.5))
                    else:
                        label_consensus = float(np.mean(label_values))
                        row["label_boundary_uncertainty"] = float(
                            np.clip(1.0 - 2.0 * abs(label_consensus - np.nanmedian(directed_labels[label_mask])), 0.0, 1.0)
                        )
            windows.append(row)

    table = pd.DataFrame(windows)
    if table.empty:
        raise ValueError("No windows produced")
    table["disagreement_type"] = table.apply(classify_window, axis=1)
    table["priority_score"] = (
        table["mean_disagreement"]
        * table["boundary_uncertainty"]
        * table["nmr_suitability"]
    )
    return table.sort_values("priority_score", ascending=False), tracks


def add_human_overlap(table: pd.DataFrame, overlap_path: Path | None) -> pd.DataFrame:
    if not overlap_path or not overlap_path.exists():
        table["human_accessions"] = ""
        table["human_genes"] = ""
        table["has_human_overlap"] = False
        return table
    overlap = pd.read_csv(overlap_path)
    if "udonpred_id" not in overlap:
        return table
    cols = ["udonpred_id", "human_accessions", "human_genes", "example_human_header"]
    overlap = overlap[[col for col in cols if col in overlap.columns]].drop_duplicates("udonpred_id")
    merged = table.merge(overlap, left_on="protein_id", right_on="udonpred_id", how="left")
    merged = merged.drop(columns=[col for col in ["udonpred_id"] if col in merged.columns])
    merged["has_human_overlap"] = merged.get("human_accessions", pd.Series(index=merged.index)).notna()
    return merged


def write_tables(table: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "contested_windows.csv", index=False)
    table.head(top_n).to_csv(output_dir / "top_contested_windows.csv", index=False)

    type_summary = (
        table.groupby("disagreement_type", dropna=False)
        .agg(
            n_windows=("priority_score", "size"),
            median_priority=("priority_score", "median"),
            median_disagreement=("mean_disagreement", "median"),
            median_boundary_uncertainty=("boundary_uncertainty", "median"),
            median_nmr_suitability=("nmr_suitability", "median"),
        )
        .reset_index()
        .sort_values("median_priority", ascending=False)
    )
    type_summary.to_csv(output_dir / "disagreement_type_summary.csv", index=False)

    feature_summary = feature_enrichment(table)
    feature_summary.to_csv(output_dir / "feature_enrichment_top_vs_background.csv", index=False)

    validation = validation_summary(table)
    validation.to_csv(output_dir / "validation_summary.csv", index=False)

    lecture_summary = lecture_extension_summary(table)
    lecture_summary.to_csv(output_dir / "lecture_extension_summary.csv", index=False)


def feature_enrichment(table: pd.DataFrame) -> pd.DataFrame:
    threshold = table["priority_score"].quantile(0.95)
    top = table[table["priority_score"] >= threshold]
    background = table[table["priority_score"] < threshold]
    rows = []
    for feature in WINDOW_FEATURE_COLUMNS:
        rows.append(
            {
                "feature": feature,
                "top5_mean": float(top[feature].mean()),
                "background_mean": float(background[feature].mean()),
                "delta_top_minus_background": float(top[feature].mean() - background[feature].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_top_minus_background", key=lambda s: s.abs(), ascending=False)


def validation_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "n_windows", "value": len(table)},
        {"metric": "n_proteins", "value": table["protein_id"].nunique()},
        {"metric": "median_priority_score", "value": float(table["priority_score"].median())},
        {"metric": "top5pct_priority_threshold", "value": float(table["priority_score"].quantile(0.95))},
        {"metric": "fraction_with_human_overlap", "value": float(table.get("has_human_overlap", False).mean())},
    ]
    if "label_boundary_uncertainty" in table:
        subset = table[["mean_disagreement", "label_boundary_uncertainty"]].dropna()
        if len(subset) >= 3:
            rows.append(
                {
                    "metric": "spearman_disagreement_vs_label_boundary_uncertainty",
                    "value": float(subset.corr(method="spearman").iloc[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def lecture_extension_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [
        "tmh_like_score",
        "signal_peptide_like_score",
        "nors_like_score",
        "coil_like_score",
        "possible_more_score",
        "concept_family_spread_z",
    ]:
        if column in table:
            rows.append(
                {
                    "lecture_control": column,
                    "median_all_windows": float(table[column].median()),
                    "median_top5pct_priority": float(table.loc[table["priority_score"] >= table["priority_score"].quantile(0.95), column].median()),
                    "top5pct_minus_all": float(
                        table.loc[table["priority_score"] >= table["priority_score"].quantile(0.95), column].median()
                        - table[column].median()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("top5pct_minus_all", key=lambda s: s.abs(), ascending=False)


def plot_outputs(table: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    top = table.head(top_n).copy()
    top["region"] = top["protein_id"] + ":" + top["start"].astype(str) + "-" + top["end"].astype(str)
    top = top.sort_values("priority_score")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
    ax.barh(top["region"], top["priority_score"], color="#C44E52")
    ax.set_xlabel("Priority score")
    ax.set_title("Top contested windows")
    for y_pos, value in enumerate(top["priority_score"]):
        ax.text(value + max(top["priority_score"].max() * 0.01, 0.01), y_pos, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "top_contested_windows.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    type_counts = table["disagreement_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(type_counts.index, type_counts.values, color="#4C78A8")
    ax.set_xlabel("Number of windows")
    ax.set_title("Contested-window disagreement types")
    fig.tight_layout()
    fig.savefig(output_dir / "disagreement_type_counts.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    features = feature_enrichment(table)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = np.where(features["delta_top_minus_background"] >= 0, "#54A24B", "#E45756")
    ax.barh(features["feature"], features["delta_top_minus_background"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Top 5% minus background")
    ax.set_title("Sequence features enriched in top contested windows")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_enrichment_top_vs_background.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(
        table["boundary_uncertainty"],
        table["mean_disagreement"],
        c=table["nmr_suitability"],
        cmap="viridis",
        s=14,
        alpha=0.55,
    )
    ax.set_xlabel("Boundary uncertainty")
    ax.set_ylabel("Predictor disagreement")
    ax.set_title("Priority favors disagreement near the decision boundary")
    colorbar = fig.colorbar(ax.collections[0], ax=ax)
    colorbar.set_label("NMR suitability")
    fig.tight_layout()
    fig.savefig(output_dir / "disagreement_vs_boundary_uncertainty.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    lecture_columns = [
        "tmh_like_score",
        "signal_peptide_like_score",
        "nors_like_score",
        "possible_more_score",
        "concept_family_spread_z",
    ]
    lecture_summary = lecture_extension_summary(table)
    if not lecture_summary.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        colors = np.where(lecture_summary["top5pct_minus_all"] >= 0, "#54A24B", "#E45756")
        ax.barh(lecture_summary["lecture_control"], lecture_summary["top5pct_minus_all"], color=colors)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Median top 5% priority minus median all windows")
        ax.set_title("Lecture-derived controls in top contested windows")
        fig.tight_layout()
        fig.savefig(output_dir / "lecture_controls_top_vs_all.png", bbox_inches="tight", dpi=220)
        plt.close(fig)

    available = [column for column in lecture_columns if column in table]
    if available:
        top = table.nlargest(min(300, len(table)), "priority_score")
        fig, axes = plt.subplots(len(available), 1, figsize=(9, 2.1 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        x = np.arange(len(top))
        labels = top["protein_id"] + ":" + top["start"].astype(str) + "-" + top["end"].astype(str)
        for ax, column in zip(axes, available):
            ax.plot(x, top[column], lw=1.3)
            ax.set_ylabel(column.replace("_", "\n"), rotation=0, ha="right", va="center")
            ax.grid(axis="y", alpha=0.2)
        axes[-1].set_xticks(x[:: max(1, len(x) // 12)])
        axes[-1].set_xticklabels(labels.iloc[:: max(1, len(x) // 12)], rotation=45, ha="right", fontsize=8)
        axes[0].set_title("Lecture-derived controls across top-priority windows")
        fig.tight_layout()
        fig.savefig(output_dir / "lecture_controls_top_windows_profile.png", bbox_inches="tight", dpi=220)
        plt.close(fig)


def plot_case_studies(
    table: pd.DataFrame,
    tracks: dict[str, pd.DataFrame],
    output_dir: Path,
    case_studies: int,
) -> None:
    case_dir = output_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)
    for _, row in table.head(case_studies).iterrows():
        protein_id = row["protein_id"]
        track = tracks[protein_id]
        start = max(int(row["start"]) - 20, 1)
        end = min(int(row["end"]) + 20, int(track["position"].max()))
        subset = track[(track["position"] >= start) & (track["position"] <= end)]
        predictor_cols = [col for col in subset.columns if col.endswith("_z") and col not in {"median_z"}]

        fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        for col in predictor_cols:
            axes[0].plot(subset["position"], subset[col], lw=1.3, alpha=0.8, label=col.removesuffix("_z"))
        axes[0].axvspan(row["start"], row["end"], color="#F58518", alpha=0.18)
        axes[0].set_ylabel("Calibrated score (z)")
        axes[0].set_title(
            f"{protein_id}:{int(row['start'])}-{int(row['end'])} | {row['disagreement_type']} | priority={row['priority_score']:.2f}"
        )
        axes[0].legend(ncol=4, fontsize=8, frameon=False)

        axes[1].plot(subset["position"], subset["residue_disagreement"], color="#C44E52", lw=1.8)
        axes[1].axvspan(row["start"], row["end"], color="#F58518", alpha=0.18)
        axes[1].set_ylabel("Disagreement")
        axes[1].set_xlabel("Residue position")
        axes[1].set_xlim(start, end)
        fig.tight_layout()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", protein_id)
        fig.savefig(case_dir / f"{safe_id}_{int(row['start'])}_{int(row['end'])}.png", bbox_inches="tight", dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, default=Path("results/udonpred_matrix/predictions"))
    parser.add_argument("--layout", choices=["udonpred_matrix", "generic"], default="udonpred_matrix")
    parser.add_argument("--target-dataset", default="plddt", help="Target dataset for udonpred_matrix layout")
    parser.add_argument("--label-jsonl", type=Path, default=None)
    parser.add_argument("--human-overlap", type=Path, default=Path("results/human_proteome_annotation_ceiling/human_proteome_overlap_details.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--case-studies", type=int, default=5)
    parser.add_argument(
        "--exclude-predictors",
        nargs="*",
        default=[],
        help="Predictor names to remove before scoring, e.g. --exclude-predictors pdbflex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results/contested_regions") / str(args.target_dataset or "generic")
    label_jsonl = args.label_jsonl
    if label_jsonl is None and args.target_dataset:
        candidate = Path("UdonPred") / "data" / args.target_dataset / "test.jsonl"
        label_jsonl = candidate if candidate.exists() else None

    exclude_predictors = {name.lower() for name in args.exclude_predictors}
    prediction_dirs = discover_prediction_dirs(
        args.prediction_root,
        args.layout,
        args.target_dataset,
        exclude_predictors,
    )
    predictions = load_predictions(prediction_dirs)
    protein_ids = common_protein_ids(predictions)
    if not protein_ids:
        raise ValueError("No proteins are shared by all predictors")

    labels = read_jsonl_labels(label_jsonl) if label_jsonl else {}
    z_params, sorted_scores = calibration_tables(predictions, protein_ids)
    table, tracks = build_window_table(
        predictions=predictions,
        protein_ids=protein_ids,
        z_params=z_params,
        sorted_scores=sorted_scores,
        window_size=args.window_size,
        step=args.step,
        labels=labels,
        label_dataset=args.target_dataset,
    )
    table = add_human_overlap(table, args.human_overlap)
    write_tables(table, output_dir, args.top_n)
    plot_outputs(table, output_dir, args.top_n)
    plot_case_studies(table, tracks, output_dir, args.case_studies)

    print(f"Predictors: {', '.join(sorted(predictions))}")
    print(f"Shared proteins: {len(protein_ids)}")
    print(f"Windows: {len(table)}")
    print(f"Wrote contested-region outputs to {output_dir}")


if __name__ == "__main__":
    main()
