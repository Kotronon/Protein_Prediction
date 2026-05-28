#!/usr/bin/env python3
"""Compute normalized UdonPred headroom against annotation ceilings.

This script combines three project outputs:

* observed UdonPred cross-dataset scores
* simple sequence baselines
* annotation agreement ceilings

The primary normalized quantity is:

    (UdonPred - best_simple_baseline) / (annotation_ceiling - best_simple_baseline)

Missing off-diagonal annotation ceilings stay missing. Diagonal cells use a
ceiling of 1.0 because a dataset's own annotation is the reference target.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
METRIC_COLUMNS = [
    "trizod",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot\n(AP)",
    "disprot\n(AUROC)",
]
DISPROT_AP = "disprot\n(AP)"
DISPROT_AUROC = "disprot\n(AUROC)"
EPSILON = 1e-12


def test_dataset_for_metric(metric: str) -> str:
    if metric in {DISPROT_AP, DISPROT_AUROC}:
        return "disprot"
    return metric


def ceiling_metric_for_column(metric: str) -> str:
    if metric == DISPROT_AP:
        return "average_precision"
    if metric == DISPROT_AUROC:
        return "auroc"
    return "spearman"


def read_score_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = pd.read_csv(path).set_index("train_dataset")
    matrix = matrix.reindex(index=DATASETS, columns=METRIC_COLUMNS)
    return matrix.apply(pd.to_numeric, errors="coerce")


def write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, float_format="%.6f")


def load_best_simple_baseline(path: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    baseline = pd.read_csv(path)
    baseline_scores = (
        baseline.set_index(["baseline", "train_dataset"])[METRIC_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
    )
    best_scores = baseline_scores.max(axis=0)
    best_rows = baseline_scores.idxmax(axis=0)
    best_summary = pd.DataFrame(
        {
            "test_metric": METRIC_COLUMNS,
            "best_simple_baseline": [best_rows[metric][0] for metric in METRIC_COLUMNS],
            "best_simple_baseline_train_dataset": [
                best_rows[metric][1] for metric in METRIC_COLUMNS
            ],
            "best_simple_baseline_score": [best_scores[metric] for metric in METRIC_COLUMNS],
        }
    )
    best_matrix = pd.DataFrame(
        {metric: best_scores[metric] for metric in METRIC_COLUMNS},
        index=DATASETS,
        columns=METRIC_COLUMNS,
    )
    return best_scores, best_matrix, best_summary


def lookup_ceiling(
    ceiling_summary: pd.DataFrame,
    train_dataset: str,
    test_dataset: str,
    metric: str,
) -> tuple[float, str, float, str]:
    rows = ceiling_summary[
        (
            (ceiling_summary["dataset_a"] == train_dataset)
            & (ceiling_summary["dataset_b"] == test_dataset)
        )
        | (
            (ceiling_summary["dataset_a"] == test_dataset)
            & (ceiling_summary["dataset_b"] == train_dataset)
        )
    ]
    metric_rows = rows[rows["metric"] == metric]
    if metric_rows.empty:
        return math.nan, "", math.nan, ""
    row = metric_rows.iloc[0]
    return (
        float(row["value"]) if pd.notna(row["value"]) else math.nan,
        str(row.get("match_mode", "")),
        float(row.get("n_residues_compared", math.nan)),
        f"{row['dataset_a']} vs {row['dataset_b']}",
    )


def build_ceiling_matrix(
    ceiling_summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ceiling_summary_path.exists():
        raise FileNotFoundError(ceiling_summary_path)
    ceiling_summary = pd.read_csv(ceiling_summary_path)
    if "value" in ceiling_summary:
        ceiling_summary["value"] = pd.to_numeric(ceiling_summary["value"], errors="coerce")

    matrix = pd.DataFrame(np.nan, index=DATASETS, columns=METRIC_COLUMNS, dtype=float)
    meta_rows: list[dict[str, object]] = []
    for train_dataset in DATASETS:
        for metric_column in METRIC_COLUMNS:
            test_dataset = test_dataset_for_metric(metric_column)
            ceiling_metric = ceiling_metric_for_column(metric_column)
            if train_dataset == test_dataset:
                value = 1.0
                match_mode = "diagonal"
                n_residues = math.nan
                source_pair = f"{train_dataset} vs {test_dataset}"
                source = "diagonal"
            else:
                value, match_mode, n_residues, source_pair = lookup_ceiling(
                    ceiling_summary,
                    train_dataset,
                    test_dataset,
                    ceiling_metric,
                )
                source = "annotation_overlap" if pd.notna(value) else "missing"
            matrix.loc[train_dataset, metric_column] = value
            meta_rows.append(
                {
                    "train_dataset": train_dataset,
                    "test_metric": metric_column,
                    "test_dataset": test_dataset,
                    "ceiling_metric": ceiling_metric,
                    "annotation_ceiling": value,
                    "ceiling_source": source,
                    "ceiling_pair": source_pair,
                    "match_mode": match_mode,
                    "n_residues_compared": n_residues,
                }
            )
    return matrix, pd.DataFrame(meta_rows)


def status_for_cell(udon: float, baseline: float, ceiling: float) -> str:
    if pd.isna(udon):
        return "missing_udon"
    if pd.isna(baseline):
        return "missing_baseline"
    if pd.isna(ceiling):
        return "missing_ceiling"
    if ceiling - baseline <= EPSILON:
        return "ceiling_not_above_baseline"
    return "ok"


def normalize_with_status(
    udon: pd.DataFrame,
    baseline: pd.DataFrame,
    ceiling: pd.DataFrame,
    ceiling_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_headroom = udon - baseline
    available_headroom = ceiling - baseline
    normalized = pd.DataFrame(np.nan, index=DATASETS, columns=METRIC_COLUMNS, dtype=float)
    status_rows: list[dict[str, object]] = []
    meta_index = ceiling_meta.set_index(["train_dataset", "test_metric"])

    for train_dataset in DATASETS:
        for metric in METRIC_COLUMNS:
            udon_value = udon.loc[train_dataset, metric]
            baseline_value = baseline.loc[train_dataset, metric]
            ceiling_value = ceiling.loc[train_dataset, metric]
            status = status_for_cell(udon_value, baseline_value, ceiling_value)
            if status == "ok":
                normalized.loc[train_dataset, metric] = (
                    raw_headroom.loc[train_dataset, metric]
                    / available_headroom.loc[train_dataset, metric]
                )
            meta = meta_index.loc[(train_dataset, metric)].to_dict()
            status_rows.append(
                {
                    "train_dataset": train_dataset,
                    "test_metric": metric,
                    "udon_score": udon_value,
                    "baseline_score": baseline_value,
                    "annotation_ceiling": ceiling_value,
                    "raw_headroom": raw_headroom.loc[train_dataset, metric],
                    "available_headroom": available_headroom.loc[train_dataset, metric],
                    "normalized_headroom": normalized.loc[train_dataset, metric],
                    "status": status,
                    **meta,
                }
            )
    return raw_headroom, available_headroom, normalized, pd.DataFrame(status_rows)


def build_normalized_summary(
    udon: pd.DataFrame,
    best_scores: pd.Series,
    best_summary: pd.DataFrame,
    cell_status: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    status_index = cell_status.set_index(["train_dataset", "test_metric"])
    summary_by_metric = best_summary.set_index("test_metric")

    for metric in METRIC_COLUMNS:
        test_dataset = test_dataset_for_metric(metric)
        best_udon_train = udon[metric].idxmax()
        same_dataset_train = test_dataset

        best_cell = status_index.loc[(best_udon_train, metric)]
        same_cell = status_index.loc[(same_dataset_train, metric)]
        ok_cells = cell_status[
            (cell_status["test_metric"] == metric) & (cell_status["status"] == "ok")
        ].copy()
        if ok_cells.empty:
            best_normalized_train = ""
            best_normalized_value = math.nan
        else:
            best_index = ok_cells["normalized_headroom"].astype(float).idxmax()
            best_normalized_train = ok_cells.loc[best_index, "train_dataset"]
            best_normalized_value = ok_cells.loc[best_index, "normalized_headroom"]

        baseline_row = summary_by_metric.loc[metric]
        rows.append(
            {
                "test_metric": metric,
                "best_udon_training_dataset": best_udon_train,
                "best_udon_score": udon.loc[best_udon_train, metric],
                "same_dataset_udon_score": udon.loc[same_dataset_train, metric],
                "best_simple_baseline": baseline_row["best_simple_baseline"],
                "best_simple_baseline_train_dataset": baseline_row[
                    "best_simple_baseline_train_dataset"
                ],
                "best_simple_baseline_score": best_scores[metric],
                "best_udon_annotation_ceiling": best_cell["annotation_ceiling"],
                "best_udon_available_headroom": best_cell["available_headroom"],
                "best_udon_raw_headroom": best_cell["raw_headroom"],
                "best_udon_normalized_headroom": best_cell["normalized_headroom"],
                "best_udon_status": best_cell["status"],
                "same_dataset_annotation_ceiling": same_cell["annotation_ceiling"],
                "same_dataset_available_headroom": same_cell["available_headroom"],
                "same_dataset_raw_headroom": same_cell["raw_headroom"],
                "same_dataset_normalized_headroom": same_cell["normalized_headroom"],
                "same_dataset_status": same_cell["status"],
                "best_normalized_training_dataset": best_normalized_train,
                "best_normalized_headroom": best_normalized_value,
            }
        )
    return pd.DataFrame(rows)


def seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path.parent))
    return int(match.group(1)) if match else None


def load_shuffled_matrices(shuffled_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(shuffled_dir.glob("seed_*/matrix.csv")):
        seed = seed_from_path(path)
        if seed is None:
            continue
        matrix = pd.read_csv(path)
        matrix["seed"] = seed
        rows.append(matrix)
    if not rows:
        return pd.DataFrame(columns=["seed", "train_dataset", *METRIC_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def compute_shuffled_outputs(
    shuffled_dir: Path,
    udon: pd.DataFrame,
    ceiling: pd.DataFrame,
    output_dir: Path,
) -> None:
    null_matrices = load_shuffled_matrices(shuffled_dir)
    if null_matrices.empty:
        summary = pd.DataFrame(
            columns=[
                "train_dataset",
                "test_metric",
                "n",
                "mean",
                "std",
                "q025",
                "q500",
                "q975",
                "is_exploratory",
            ]
        )
        mean_matrix = pd.DataFrame(np.nan, index=DATASETS, columns=METRIC_COLUMNS)
    else:
        long_null = null_matrices.melt(
            id_vars=["seed", "train_dataset"],
            value_vars=METRIC_COLUMNS,
            var_name="test_metric",
            value_name="score",
        ).dropna()
        long_null["score"] = pd.to_numeric(long_null["score"], errors="coerce")
        summary = (
            long_null.groupby(["train_dataset", "test_metric"])["score"]
            .agg(
                n="count",
                mean="mean",
                std="std",
                q025=lambda series: series.quantile(0.025),
                q500="median",
                q975=lambda series: series.quantile(0.975),
            )
            .reset_index()
        )
        summary["is_exploratory"] = summary["n"] < 2
        mean_matrix = (
            summary.pivot(index="train_dataset", columns="test_metric", values="mean")
            .reindex(index=DATASETS, columns=METRIC_COLUMNS)
            .apply(pd.to_numeric, errors="coerce")
        )

    udon_minus_null = udon - mean_matrix
    available_vs_null = ceiling - mean_matrix
    normalized_vs_null = pd.DataFrame(np.nan, index=DATASETS, columns=METRIC_COLUMNS, dtype=float)
    ok = (
        udon.notna()
        & mean_matrix.notna()
        & ceiling.notna()
        & ((ceiling - mean_matrix) > EPSILON)
    )
    normalized_vs_null[ok] = udon_minus_null[ok] / available_vs_null[ok]

    summary.to_csv(output_dir / "shuffled_null_summary.csv", index=False, float_format="%.6f")
    write_matrix(mean_matrix, output_dir / "shuffled_null_mean_matrix.csv")
    write_matrix(udon_minus_null, output_dir / "udon_minus_shuffled_null.csv")
    write_matrix(normalized_vs_null, output_dir / "normalized_headroom_vs_shuffled_null.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udon-matrix", type=Path, default=Path("results/udonpred_matrix/matrix.csv"))
    parser.add_argument(
        "--baseline-matrix",
        type=Path,
        default=Path("results/simple_baselines/matrix.csv"),
    )
    parser.add_argument(
        "--annotation-ceiling-summary",
        type=Path,
        default=Path("results/annotation_ceiling/annotation_ceiling_summary.csv"),
    )
    parser.add_argument(
        "--shuffled-dir",
        type=Path,
        default=Path("results/udonpred_shuffled_labels"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/normalized_headroom"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    udon = read_score_matrix(args.udon_matrix)
    best_scores, best_baseline_matrix, best_summary = load_best_simple_baseline(
        args.baseline_matrix
    )
    ceiling_matrix, ceiling_meta = build_ceiling_matrix(args.annotation_ceiling_summary)

    raw_headroom, available_headroom, normalized_headroom, cell_status = normalize_with_status(
        udon,
        best_baseline_matrix,
        ceiling_matrix,
        ceiling_meta,
    )
    normalized_summary = build_normalized_summary(
        udon,
        best_scores,
        best_summary,
        cell_status,
    )

    write_matrix(ceiling_matrix, output_dir / "ceiling_matrix.csv")
    write_matrix(best_baseline_matrix, output_dir / "best_simple_baseline_matrix.csv")
    write_matrix(raw_headroom, output_dir / "raw_headroom_vs_best_simple_baseline.csv")
    write_matrix(available_headroom, output_dir / "available_headroom_vs_best_simple_baseline.csv")
    write_matrix(
        normalized_headroom,
        output_dir / "normalized_headroom_vs_best_simple_baseline.csv",
    )
    best_summary.to_csv(
        output_dir / "best_simple_baseline_per_metric.csv",
        index=False,
        float_format="%.6f",
    )
    normalized_summary.to_csv(
        output_dir / "normalized_headroom_summary.csv",
        index=False,
        float_format="%.6f",
    )
    cell_status.to_csv(output_dir / "cell_status.csv", index=False, float_format="%.6f")

    compute_shuffled_outputs(args.shuffled_dir, udon, ceiling_matrix, output_dir)

    print(f"Wrote normalized headroom outputs to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
