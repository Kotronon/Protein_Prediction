#!/usr/bin/env python3
"""Focused diagnostics for DisPredict3 versus UdonPred and CAID-style predictors.
python scripts/analyze_dispredict3_vs_udonpred.py \
  --skip-default-external-predictors \
  --external-predictor ADOPT=results/human_proteome/ADOPT \
  --external-predictor SETH=results/human_proteome/SETH/seth_human_proteome.caid \
  --external-predictor metapredict=results/human_proteome/metapredict/metapredict_human_proteome.caid \
  --external-predictor IUPred3=results/human_proteome/IUPred3 \
  --external-predictor DisoFLAG=results/human_proteome/DisoFLAG/caid \
  --external-predictor DisorderUnetLM=results/human_proteome/DisorderUnetLM/disorder \
  --external-predictor PUNCH2_light=results/human_proteome/PUNCH2_light/disorder \
  --output-dir results/compare_predictors_with_all_predictors_wo_pdbflex    
"""

from __future__ import annotations

import argparse
import csv
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


UDONPRED_MODELS = ("trizod", "chezod", "softdis", "atlas", "plddt", "disprot")
DEFAULT_NEGATED = {"adopt", "chezod", "plddt"}
DEFAULT_EXTERNAL_PREDICTORS = {
    "DisoFLAG": Path("results/human_proteome/DisoFLAG/caid"),
    "DisorderUnetLM": Path("results/human_proteome/DisorderUnetLM/disorder"),
    "PUNCH2_light": Path("results/human_proteome/PUNCH2_light/disorder"),
}


def finite_scores(record: PredictionRecord) -> np.ndarray:
    return record.scores[np.isfinite(record.scores)]


def safe_se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def bootstrap_se(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)
    return float(np.std(means, ddof=1))


def summarize_predictor_scores(records: dict[str, PredictionRecord], name: str) -> dict[str, object]:
    per_protein_mean = []
    total_residues = 0
    score_chunks = []

    for record in records.values():
        scores = finite_scores(record)
        if len(scores) == 0:
            continue
        total_residues += len(scores)
        per_protein_mean.append(float(np.mean(scores)))
        score_chunks.append(scores)

    all_scores = np.concatenate(score_chunks) if score_chunks else np.asarray([], dtype=np.float64)
    protein_means = np.asarray(per_protein_mean, dtype=np.float64)

    return {
        "predictor": name,
        "proteins": len(per_protein_mean),
        "residues": total_residues,
        "score_mean": float(np.mean(all_scores)) if len(all_scores) else math.nan,
        "score_std": float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else math.nan,
        "score_p05": float(np.quantile(all_scores, 0.05)) if len(all_scores) else math.nan,
        "score_p50": float(np.quantile(all_scores, 0.50)) if len(all_scores) else math.nan,
        "score_p95": float(np.quantile(all_scores, 0.95)) if len(all_scores) else math.nan,
        "protein_mean_score_mean": float(np.mean(protein_means)) if len(protein_means) else math.nan,
        "protein_mean_score_se": safe_se(protein_means),
    }


def pair_diagnostics(
    dispredict3: dict[str, PredictionRecord],
    other: dict[str, PredictionRecord],
    other_name: str,
    group: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, object]:
    common_ids = sorted(set(dispredict3) & set(other))
    per_protein_spearman = []
    per_protein_zscore_mae = []
    dispredict3_means = []
    other_means = []
    matched_residues = 0

    for protein_id in common_ids:
        left = dispredict3[protein_id].scores
        right = other[protein_id].scores
        length = min(len(left), len(right))
        if length < 2:
            continue
        left = left[:length]
        right = right[:length]
        mask = np.isfinite(left) & np.isfinite(right)
        if np.sum(mask) < 2:
            continue
        left = left[mask]
        right = right[mask]
        matched_residues += len(left)

        rho = correlation(left, right, "spearman")
        if math.isfinite(rho):
            per_protein_spearman.append(rho)
        per_protein_zscore_mae.append(float(np.mean(np.abs(zscore(left) - zscore(right)))))
        dispredict3_means.append(float(np.mean(left)))
        other_means.append(float(np.mean(right)))

    protein_spearman_values = np.asarray(per_protein_spearman, dtype=np.float64)
    protein_zscore_mae = np.asarray(per_protein_zscore_mae, dtype=np.float64)
    dispredict3_means_array = np.asarray(dispredict3_means, dtype=np.float64)
    other_means_array = np.asarray(other_means, dtype=np.float64)
    mean_shift = other_means_array - dispredict3_means_array

    return {
        "predictor": other_name,
        "group": group,
        "common_proteins": len(common_ids),
        "matched_residues": matched_residues,
        "mean_per_protein_residue_spearman": float(np.mean(protein_spearman_values)),
        "mean_per_protein_residue_spearman_se": safe_se(protein_spearman_values),
        "mean_per_protein_residue_spearman_bootstrap_se": bootstrap_se(
            protein_spearman_values, rng, n_bootstrap
        ),
        "protein_mean_score_spearman": correlation(dispredict3_means_array, other_means_array, "spearman"),
        "mean_per_protein_zscore_mae": float(np.mean(protein_zscore_mae)),
        "mean_per_protein_zscore_mae_se": safe_se(protein_zscore_mae),
        "oriented_mean_score_shift_other_minus_dispredict3": float(np.mean(mean_shift)),
        "oriented_mean_score_shift_other_minus_dispredict3_se": safe_se(mean_shift),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        default=Path("results/compare_predictors_with_all_caid_predictors_wo_pdbflex"),
    )
    parser.add_argument(
        "--external-predictor",
        action="append",
        type=parse_predictor_argument,
        help=(
            "Additional CAID-style predictor as NAME=PATH. PATH can be one .caid file "
            "or a directory containing .caid files. Repeat for multiple predictors."
        ),
    )
    parser.add_argument(
        "--skip-default-external-predictors",
        action="store_true",
        help="Only compare UdonPred plus explicitly supplied --external-predictor values.",
    )
    parser.add_argument(
        "--negate",
        nargs="*",
        default=sorted(DEFAULT_NEGATED),
        help="Predictor names whose scores should be multiplied by -1 before comparison.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    negated_predictors = {name.lower() for name in args.negate}

    dispredict3 = load_predictions(args.dispredict3_dir)
    predictor_records = {"DisPredict3": dispredict3}
    for model in UDONPRED_MODELS:
        records = load_predictions(args.udonpred_root / model)
        predictor_records[model] = maybe_negate_scores(records, model, negated_predictors)

    external_predictors = {} if args.skip_default_external_predictors else dict(DEFAULT_EXTERNAL_PREDICTORS)
    if args.external_predictor:
        external_predictors.update(dict(args.external_predictor))

    for name, path in external_predictors.items():
        if path.exists():
            predictor_records[name] = maybe_negate_scores(
                load_predictions(path),
                name,
                negated_predictors,
            )
        else:
            print(f"Skipping missing external predictor {name}: {path}")

    score_rows = [
        summarize_predictor_scores(records, name)
        for name, records in predictor_records.items()
    ]
    pair_rows = [
        pair_diagnostics(dispredict3, predictor_records[model], model, "UdonPred", rng, args.bootstrap)
        for model in UDONPRED_MODELS
    ]
    pair_rows.extend(
        pair_diagnostics(dispredict3, predictor_records[name], name, "external", rng, args.bootstrap)
        for name in external_predictors
        if name in predictor_records
    )

    write_csv(score_rows, args.output_dir / "dispredict3_predictor_score_distribution.csv")
    write_csv(pair_rows, args.output_dir / "dispredict3_vs_predictors_proteinwise_se.csv")

    print(f"Wrote {args.output_dir / 'dispredict3_predictor_score_distribution.csv'}")
    print(f"Wrote {args.output_dir / 'dispredict3_vs_predictors_proteinwise_se.csv'}")


if __name__ == "__main__":
    main()
