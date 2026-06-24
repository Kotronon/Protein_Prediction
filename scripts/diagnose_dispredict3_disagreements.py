#!/usr/bin/env python3
"""Diagnose why DisPredict3 differs from other disorder predictors.

The script decomposes disagreement into technical input checks, per-protein
differences, high-disagreement residue windows, and overlap of high-scoring
residue segments.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path

import numpy as np

from compare_predictors import (
    PredictionRecord,
    correlation,
    load_predictions,
    maybe_negate_scores,
    parse_predictor_argument,
    zscore,
)
from protein_prediction_config import NEGATED_UDONPRED_DATASETS, UDONPRED_DISORDER_MODELS


UDONPRED_MODELS = UDONPRED_DISORDER_MODELS
DEFAULT_NEGATED = {"adopt"} | set(NEGATED_UDONPRED_DATASETS)
DEFAULT_EXTERNAL_PREDICTORS = {
    "DisoFLAG": Path("results/human_proteome/DisoFLAG/caid"),
    "DisorderUnetLM": Path("results/human_proteome/DisorderUnetLM/disorder"),
    "PUNCH2_light": Path("results/human_proteome/PUNCH2_light/disorder"),
}


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.8g}" if isinstance(value, float) and math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def load_all_predictors(args: argparse.Namespace) -> dict[str, dict[str, PredictionRecord]]:
    negated_predictors = {name.lower() for name in args.negate}
    requested = set(args.only_predictor or [])

    def should_load(name: str) -> bool:
        return not requested or name in requested

    predictors: dict[str, dict[str, PredictionRecord]] = {
        "DisPredict3": load_predictions(args.dispredict3_dir)
    }

    for model in UDONPRED_MODELS:
        if not should_load(model):
            continue
        records = load_predictions(args.udonpred_root / model)
        predictors[model] = maybe_negate_scores(records, model, negated_predictors)

    external_predictors = {} if args.skip_default_external_predictors else dict(DEFAULT_EXTERNAL_PREDICTORS)
    if args.external_predictor:
        external_predictors.update(dict(args.external_predictor))

    for name, path in external_predictors.items():
        if not should_load(name):
            continue
        if path.exists():
            predictors[name] = maybe_negate_scores(load_predictions(path), name, negated_predictors)
        else:
            print(f"Skipping missing external predictor {name}: {path}")

    return predictors


def technical_qc_rows(
    dispredict3: dict[str, PredictionRecord],
    predictors: dict[str, dict[str, PredictionRecord]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    d3_ids = set(dispredict3)
    for name, records in predictors.items():
        if name == "DisPredict3":
            continue
        other_ids = set(records)
        common_ids = sorted(d3_ids & other_ids)
        length_mismatches = 0
        overlap_sequence_mismatches = 0
        exact_sequence_mismatches = 0
        matched_residues = 0
        truncated_residues = 0

        for protein_id in common_ids:
            d3 = dispredict3[protein_id]
            other = records[protein_id]
            length = min(len(d3.scores), len(other.scores))
            matched_residues += length
            truncated_residues += abs(len(d3.scores) - len(other.scores))
            if len(d3.scores) != len(other.scores):
                length_mismatches += 1
            if d3.sequence[:length] != other.sequence[:length]:
                overlap_sequence_mismatches += 1
            if d3.sequence != other.sequence:
                exact_sequence_mismatches += 1

        rows.append(
            {
                "predictor": name,
                "dispredict3_proteins": len(d3_ids),
                "predictor_proteins": len(other_ids),
                "common_proteins": len(common_ids),
                "only_dispredict3": len(d3_ids - other_ids),
                "only_predictor": len(other_ids - d3_ids),
                "length_mismatched_common_proteins": length_mismatches,
                "truncated_residues_due_to_length_mismatch": truncated_residues,
                "overlap_sequence_mismatched_proteins": overlap_sequence_mismatches,
                "exact_sequence_mismatched_proteins": exact_sequence_mismatches,
                "matched_residues_after_min_length": matched_residues,
            }
        )
    return rows


def finite_pair_scores(
    left: PredictionRecord,
    right: PredictionRecord,
) -> tuple[np.ndarray, np.ndarray, int]:
    length = min(len(left.scores), len(right.scores))
    if length == 0:
        return np.asarray([]), np.asarray([]), 0
    x = left.scores[:length]
    y = right.scores[:length]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], length


def per_protein_disagreement_rows(
    dispredict3: dict[str, PredictionRecord],
    predictors: dict[str, dict[str, PredictionRecord]],
    include_spearman: bool,
    min_residues: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, records in predictors.items():
        if name == "DisPredict3":
            continue
        for protein_id in sorted(set(dispredict3) & set(records)):
            d3 = dispredict3[protein_id]
            other = records[protein_id]
            x, y, overlap_length = finite_pair_scores(d3, other)
            if len(x) < min_residues:
                continue
            zx = zscore(x)
            zy = zscore(y)
            residue_spearman = correlation(x, y, "spearman") if include_spearman else math.nan
            rows.append(
                {
                    "predictor": name,
                    "protein_id": protein_id,
                    "dispredict3_length": len(d3.scores),
                    "predictor_length": len(other.scores),
                    "overlap_length": overlap_length,
                    "used_residues": len(x),
                    "length_mismatch": len(d3.scores) != len(other.scores),
                    "sequence_mismatch_on_overlap": d3.sequence[:overlap_length] != other.sequence[:overlap_length],
                    "dispredict3_mean": float(np.mean(x)),
                    "predictor_mean": float(np.mean(y)),
                    "mean_difference_predictor_minus_dispredict3": float(np.mean(y) - np.mean(x)),
                    "residue_spearman": residue_spearman,
                    "zscore_mae": float(np.mean(np.abs(zx - zy))),
                    "dispredict3_z_high_other_z_low_fraction": float(np.mean((zx >= 1.0) & (zy <= 0.0))),
                    "dispredict3_z_low_other_z_high_fraction": float(np.mean((zx <= 0.0) & (zy >= 1.0))),
                }
            )
    return rows


def top_rows_by_predictor(
    rows: list[dict[str, object]],
    metric: str,
    top_n: int,
) -> list[dict[str, object]]:
    top_rows: list[dict[str, object]] = []
    predictors = sorted({str(row["predictor"]) for row in rows})
    for predictor in predictors:
        predictor_rows = [row for row in rows if row["predictor"] == predictor]
        predictor_rows.sort(key=lambda row: float(row[metric]), reverse=True)
        for rank, row in enumerate(predictor_rows[:top_n], start=1):
            ranked = {"rank": rank, **row}
            top_rows.append(ranked)
    return top_rows


def high_disagreement_region_rows(
    dispredict3: dict[str, PredictionRecord],
    predictors: dict[str, dict[str, PredictionRecord]],
    window_size: int,
    top_n: int,
) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    for name, records in predictors.items():
        if name == "DisPredict3":
            continue
        heap: list[tuple[float, int, dict[str, object]]] = []
        counter = 0
        for protein_id in sorted(set(dispredict3) & set(records)):
            d3 = dispredict3[protein_id]
            other = records[protein_id]
            x, y, _ = finite_pair_scores(d3, other)
            if len(x) < window_size:
                continue
            zx = zscore(x)
            zy = zscore(y)
            signed = zx - zy
            abs_diff = np.abs(signed)
            for start in range(0, len(abs_diff) - window_size + 1, window_size):
                end = start + window_size
                mean_signed = float(np.mean(signed[start:end]))
                if mean_signed > 0:
                    direction = "DisPredict3 higher"
                elif mean_signed < 0:
                    direction = f"{name} higher"
                else:
                    direction = "balanced"
                mean_abs = float(np.mean(abs_diff[start:end]))
                row = {
                    "predictor": name,
                    "protein_id": protein_id,
                    "start_residue": start + 1,
                    "end_residue": end,
                    "window_size": end - start,
                    "mean_abs_zscore_difference": mean_abs,
                    "max_abs_zscore_difference": float(np.max(abs_diff[start:end])),
                    "mean_signed_zscore_difference": mean_signed,
                    "direction": direction,
                    "dispredict3_mean_z": float(np.mean(zx[start:end])),
                    "predictor_mean_z": float(np.mean(zy[start:end])),
                    "sequence": d3.sequence[start:end],
                }
                counter += 1
                if len(heap) < top_n:
                    heapq.heappush(heap, (mean_abs, counter, row))
                elif mean_abs > heap[0][0]:
                    heapq.heapreplace(heap, (mean_abs, counter, row))
        rows = [entry[2] for entry in sorted(heap, reverse=True)]
        all_rows.extend({"rank": rank, **row} for rank, row in enumerate(rows, start=1))
    return all_rows


def high_score_thresholds(
    predictors: dict[str, dict[str, PredictionRecord]],
    quantile: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for name, records in predictors.items():
        chunks = [
            record.scores[np.isfinite(record.scores)]
            for record in records.values()
            if np.any(np.isfinite(record.scores))
        ]
        if chunks:
            thresholds[name] = float(np.quantile(np.concatenate(chunks), quantile))
    return thresholds


def segment_count(mask: np.ndarray, min_length: int) -> int:
    count = 0
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                count += 1
            start = None
    if start is not None and len(mask) - start >= min_length:
        count += 1
    return count


def segment_overlap_rows(
    dispredict3: dict[str, PredictionRecord],
    predictors: dict[str, dict[str, PredictionRecord]],
    quantile: float,
    min_segment_length: int,
) -> list[dict[str, object]]:
    thresholds = high_score_thresholds(predictors, quantile)
    rows: list[dict[str, object]] = []
    d3_threshold = thresholds["DisPredict3"]

    for name, records in predictors.items():
        if name == "DisPredict3" or name not in thresholds:
            continue
        intersection = 0
        union = 0
        d3_high = 0
        other_high = 0
        total = 0
        per_protein_jaccards = []
        d3_segments = 0
        other_segments = 0
        shared_segments = 0

        for protein_id in sorted(set(dispredict3) & set(records)):
            d3 = dispredict3[protein_id]
            other = records[protein_id]
            x, y, _ = finite_pair_scores(d3, other)
            if len(x) == 0:
                continue
            x_high = x >= d3_threshold
            y_high = y >= thresholds[name]
            both = x_high & y_high
            either = x_high | y_high
            intersection += int(np.sum(both))
            union += int(np.sum(either))
            d3_high += int(np.sum(x_high))
            other_high += int(np.sum(y_high))
            total += len(x)
            if np.any(either):
                per_protein_jaccards.append(float(np.sum(both) / np.sum(either)))
            d3_segments += segment_count(x_high, min_segment_length)
            other_segments += segment_count(y_high, min_segment_length)
            shared_segments += segment_count(both, min_segment_length)

        rows.append(
            {
                "predictor": name,
                "high_score_quantile": quantile,
                "dispredict3_threshold": d3_threshold,
                "predictor_threshold": thresholds[name],
                "matched_residues": total,
                "dispredict3_high_residues": d3_high,
                "predictor_high_residues": other_high,
                "shared_high_residues": intersection,
                "union_high_residues": union,
                "residue_jaccard": intersection / union if union else math.nan,
                "fraction_of_dispredict3_high_shared": intersection / d3_high if d3_high else math.nan,
                "fraction_of_predictor_high_shared": intersection / other_high if other_high else math.nan,
                "mean_per_protein_jaccard": float(np.mean(per_protein_jaccards)) if per_protein_jaccards else math.nan,
                "dispredict3_segments": d3_segments,
                "predictor_segments": other_segments,
                "shared_segments": shared_segments,
                "min_segment_length": min_segment_length,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispredict3-dir",
        type=Path,
        default=Path("results/human_proteome/Dispredict3_native/caid"),
    )
    parser.add_argument(
        "--udonpred-root",
        type=Path,
        default=Path("results/human_proteome/UdonPred"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dispredict3_disagreement_diagnostics"),
    )
    parser.add_argument("--external-predictor", action="append", type=parse_predictor_argument)
    parser.add_argument("--skip-default-external-predictors", action="store_true")
    parser.add_argument("--negate", nargs="*", default=sorted(DEFAULT_NEGATED))
    parser.add_argument(
        "--only-predictor",
        action="append",
        help=(
            "Limit diagnostics to this predictor name. Repeat for multiple predictors. "
            "DisPredict3 is always kept."
        ),
    )
    parser.add_argument("--top-proteins", type=int, default=100)
    parser.add_argument("--top-regions", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument(
        "--min-protein-residues",
        type=int,
        default=30,
        help="Minimum usable residues for per-protein disagreement rows.",
    )
    parser.add_argument("--high-score-quantile", type=float, default=0.90)
    parser.add_argument("--min-segment-length", type=int, default=10)
    parser.add_argument(
        "--include-per-protein-spearman",
        action="store_true",
        help="Also compute per-protein residue Spearman. This is much slower on the full proteome.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.high_score_quantile < 1:
        raise ValueError("--high-score-quantile must be between 0 and 1")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")

    predictors = load_all_predictors(args)
    if args.only_predictor:
        keep = {"DisPredict3", *args.only_predictor}
        missing = sorted(keep - set(predictors))
        if missing:
            raise ValueError(f"--only-predictor names not loaded: {', '.join(missing)}")
        predictors = {name: records for name, records in predictors.items() if name in keep}
    dispredict3 = predictors["DisPredict3"]

    qc_rows = technical_qc_rows(dispredict3, predictors)
    protein_rows = per_protein_disagreement_rows(
        dispredict3,
        predictors,
        include_spearman=args.include_per_protein_spearman,
        min_residues=args.min_protein_residues,
    )
    top_protein_rows = top_rows_by_predictor(protein_rows, "zscore_mae", args.top_proteins)
    region_rows = high_disagreement_region_rows(
        dispredict3,
        predictors,
        window_size=args.window_size,
        top_n=args.top_regions,
    )
    segment_rows = segment_overlap_rows(
        dispredict3,
        predictors,
        quantile=args.high_score_quantile,
        min_segment_length=args.min_segment_length,
    )

    write_csv(qc_rows, args.output_dir / "technical_qc.csv")
    write_csv(protein_rows, args.output_dir / "per_protein_disagreement.csv")
    write_csv(top_protein_rows, args.output_dir / "top_disagreement_proteins.csv")
    write_csv(region_rows, args.output_dir / "top_disagreement_regions.csv")
    write_csv(segment_rows, args.output_dir / "high_score_segment_overlap.csv")

    print(f"Wrote {args.output_dir / 'technical_qc.csv'}")
    print(f"Wrote {args.output_dir / 'per_protein_disagreement.csv'}")
    print(f"Wrote {args.output_dir / 'top_disagreement_proteins.csv'}")
    print(f"Wrote {args.output_dir / 'top_disagreement_regions.csv'}")
    print(f"Wrote {args.output_dir / 'high_score_segment_overlap.csv'}")


if __name__ == "__main__":
    main()
