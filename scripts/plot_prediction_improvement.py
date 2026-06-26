#!/usr/bin/env python3
"""Create visual summaries for prediction improvements.

Inputs are the CSV outputs from ``run_prediction_pipeline.py`` and
``run_udonpred_ensembles.py``. The figures are intentionally presentation-ready
comparisons:

* ensemble delta heatmap versus the best individual UdonPred head
* best individual head versus best validation-trained ensemble
* baseline -> UdonPred -> ensemble -> annotation ceiling headroom plot
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteinprediction-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "trizod",
    "trizod_updated",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot\n(AP)",
    "disprot\n(AUROC)",
]
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


def metric_columns_for_dataset(dataset: str) -> list[str]:
    return ["disprot\n(AP)", "disprot\n(AUROC)"] if dataset == "disprot" else [dataset]


def metric_columns_for_datasets(datasets: list[str]) -> list[str]:
    columns = []
    for dataset in datasets:
        columns.extend(metric_columns_for_dataset(dataset))
    return columns


def resolve_metric_columns(excluded: list[str], replace_trizod_with_updated: bool) -> list[str]:
    excluded_set = set(excluded)
    unknown = sorted(excluded_set - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets in --exclude-datasets: {', '.join(unknown)}")
    base = ["trizod_updated" if replace_trizod_with_updated else "trizod"]
    base.extend(["chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"])
    datasets = [dataset for dataset in base if dataset not in excluded_set]
    if not datasets:
        raise ValueError("At least one dataset must remain after --exclude-datasets")
    return metric_columns_for_datasets(datasets)


def display_metric(metric: str) -> str:
    return metric.replace("\n", " ")


def read_numeric_matrix(path: Path, index_col: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the prediction pipeline and ensemble step first."
        )
    frame = pd.read_csv(path, index_col=index_col)
    frame = frame.reindex(columns=METRIC_COLUMNS)
    return frame.apply(pd.to_numeric, errors="coerce")


def read_udon_matrix(path: Path) -> pd.DataFrame:
    return read_numeric_matrix(path, index_col="train_dataset")


def read_baseline_matrix(path: Path) -> pd.DataFrame:
    return read_numeric_matrix(path, index_col=False)


def best_series(frame: pd.DataFrame) -> pd.Series:
    return frame[METRIC_COLUMNS].max(axis=0, skipna=True)


def best_ensemble_matrix(ensemble: pd.DataFrame) -> pd.DataFrame:
    return ensemble.drop(index="best_individual_test_oracle", errors="ignore")


def read_best_individual_heads(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(index=METRIC_COLUMNS, dtype=object)
    frame = pd.read_csv(path, index_col=0)
    return frame.iloc[:, 0].reindex(METRIC_COLUMNS)


def best_strategy_summary(
    non_oracle_ensemble: pd.DataFrame,
    best_udon: pd.Series,
    best_heads: pd.Series,
    combinations: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    combinations = combinations or {}
    rows = []
    for metric in METRIC_COLUMNS:
        scores = non_oracle_ensemble[metric].dropna()
        strategy = scores.idxmax()
        score = float(scores.loc[strategy])
        delta = score - float(best_udon[metric])
        target = metric_to_target(metric)
        rows.append(
            {
                "metric": metric,
                "best_individual_head": best_heads.get(metric, ""),
                "best_individual_score": float(best_udon[metric]),
                "best_ensemble_strategy": strategy,
                "best_ensemble_combination": combination_for_strategy(
                    combinations, strategy, target
                ),
                "best_ensemble_score": score,
                "ensemble_minus_best_individual": delta,
                "interpretation": (
                    "improves" if delta > 1e-6 else "ties" if abs(delta) <= 1e-6 else "worse"
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_to_target(metric: str) -> str:
    return "disprot" if metric.startswith("disprot") else metric


def format_weighted_combination(row: pd.Series, predictor_columns: list[str]) -> str:
    parts = []
    for predictor in predictor_columns:
        value = row.get(predictor)
        if pd.isna(value):
            continue
        value = float(value)
        if abs(value) <= 1e-6:
            continue
        parts.append((predictor, value))
    if not parts:
        return ""
    parts.sort(key=lambda item: abs(item[1]), reverse=True)
    return " + ".join(f"{name} ({value:+.3f})" for name, value in parts)


def read_combination_lookup(
    weights_path: Path,
    subset_choices_path: Path,
    validation_selected_heads_path: Path,
) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    if weights_path.exists():
        weights = pd.read_csv(weights_path)
        predictor_columns = [
            column
            for column in weights.columns
            if column not in {"strategy", "target_dataset", "intercept"}
        ]
        for _, row in weights.iterrows():
            combination = format_weighted_combination(row, predictor_columns)
            lookup[(str(row["strategy"]), str(row["target_dataset"]))] = combination
    if subset_choices_path.exists():
        subsets = pd.read_csv(subset_choices_path)
        for _, row in subsets.iterrows():
            strategy = str(row["strategy"])
            target = str(row["target_dataset"])
            heads = str(row["heads"])
            predictor_columns = [
                column
                for column in subsets.columns
                if column not in {"strategy", "target_dataset", "validation_score", "heads"}
            ]
            weighted = format_weighted_combination(row, predictor_columns)
            lookup[(strategy, target)] = weighted or heads
    if validation_selected_heads_path.exists():
        selected = pd.read_csv(validation_selected_heads_path, index_col=0)
        for target, row in selected.iterrows():
            lookup[("validation_selected_single_head", str(target))] = str(row.iloc[0])
    return lookup


def combination_for_strategy(
    combinations: dict[tuple[str, str], str],
    strategy: str,
    target: str,
) -> str:
    if (strategy, target) in combinations:
        return combinations[(strategy, target)]
    if (strategy, "all") in combinations:
        return combinations[(strategy, "all")]
    for (candidate_strategy, _candidate_target), combination in combinations.items():
        if candidate_strategy == strategy:
            return combination
    return ""


def write_improvement_summary(
    baseline: pd.Series,
    udon: pd.Series,
    ensemble: pd.Series,
    ceiling: pd.Series,
    output: Path,
) -> None:
    summary = pd.DataFrame(
        {
            "metric": METRIC_COLUMNS,
            "best_simple_baseline": [baseline[metric] for metric in METRIC_COLUMNS],
            "best_individual_udonpred": [udon[metric] for metric in METRIC_COLUMNS],
            "best_validation_ensemble": [ensemble[metric] for metric in METRIC_COLUMNS],
            "annotation_ceiling": [ceiling[metric] for metric in METRIC_COLUMNS],
        }
    )
    summary["ensemble_minus_best_individual"] = (
        summary["best_validation_ensemble"] - summary["best_individual_udonpred"]
    )
    summary["udon_minus_baseline"] = (
        summary["best_individual_udonpred"] - summary["best_simple_baseline"]
    )
    summary["ensemble_minus_baseline"] = (
        summary["best_validation_ensemble"] - summary["best_simple_baseline"]
    )
    summary.to_csv(output, index=False, float_format="%.6f")


def plot_delta_heatmap(delta: pd.DataFrame, output: Path) -> None:
    data = delta.to_numpy(dtype=float)
    labels_y = delta.index.tolist()
    labels_x = [display_metric(metric) for metric in delta.columns]
    finite = data[np.isfinite(data)]
    limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite)))) if len(finite) else 1.0
    limit = max(limit, 1e-6)

    fig_width = max(10, 0.95 * len(labels_x))
    fig_height = max(4.5, 0.55 * len(labels_y) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(data, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(labels_x)), labels_x, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels_y)), labels_y)
    ax.set_title("Validation-trained ensemble delta vs best individual UdonPred head", loc="left")
    ax.set_xlabel("Test dataset / metric")
    ax.set_ylabel("Ensemble strategy")

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            if math.isfinite(value):
                ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("score delta")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_barplot(udon: pd.Series, ensemble: pd.Series, output: Path) -> None:
    x = np.arange(len(METRIC_COLUMNS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, [udon[m] for m in METRIC_COLUMNS], width, label="Best individual head")
    ax.bar(
        x + width / 2,
        [ensemble[m] for m in METRIC_COLUMNS],
        width,
        label="Best validation ensemble",
    )
    for i, metric in enumerate(METRIC_COLUMNS):
        delta = ensemble[metric] - udon[metric]
        y = max(udon[metric], ensemble[metric])
        ax.text(i, y + 0.015, f"{delta:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [display_metric(metric) for metric in METRIC_COLUMNS], rotation=35, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Best individual UdonPred head vs best validation-trained ensemble", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_delta_bar(delta: pd.Series, output: Path) -> None:
    values = [delta[metric] for metric in METRIC_COLUMNS]
    colors = ["#2ca25f" if value > 0 else "#de2d26" if value < 0 else "#969696" for value in values]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(METRIC_COLUMNS))
    ax.bar(x, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    for i, value in enumerate(values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.002 if value >= 0 else -0.002
        ax.text(i, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(x, [display_metric(metric) for metric in METRIC_COLUMNS], rotation=35, ha="right")
    ax.set_ylabel("score delta")
    ax.set_title("Best ensemble gain is small and target-specific", loc="left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_strategy_by_metric(summary: pd.DataFrame, output: Path) -> None:
    values = summary["ensemble_minus_best_individual"].to_numpy(dtype=float)
    colors = ["#2ca25f" if value > 0 else "#de2d26" if value < 0 else "#969696" for value in values]
    labels = [display_metric(metric) for metric in summary["metric"]]
    strategies = summary["best_ensemble_strategy"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(summary))
    ax.bar(x, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    for i, (value, strategy) in enumerate(zip(values, strategies)):
        va = "bottom" if value >= 0 else "top"
        offset = 0.002 if value >= 0 else -0.002
        short_strategy = strategy.replace("_validation", "").replace("_stacking", "")
        ax.text(i, value + offset, f"{value:+.3f}\n{short_strategy}", ha="center", va=va, fontsize=7)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("score delta")
    ax.set_title("Which ensemble strategy is best for each metric?", loc="left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_combination_by_metric(summary: pd.DataFrame, output: Path) -> None:
    data = summary.copy()
    data["display_metric"] = data["metric"].map(display_metric)
    values = data["ensemble_minus_best_individual"].to_numpy(dtype=float)
    colors = ["#2ca25f" if value > 0 else "#de2d26" if value < 0 else "#969696" for value in values]
    y = np.arange(len(data))
    fig_height = max(4.8, 0.9 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    labels = []
    for _, row in data.iterrows():
        strategy = str(row["best_ensemble_strategy"]).replace("_validation", "")
        combination = str(row["best_ensemble_combination"])
        if combination and combination != "nan":
            labels.append(f"{display_metric(row['metric'])}\n{strategy}: {combination}")
        else:
            labels.append(f"{display_metric(row['metric'])}\n{strategy}")
    ax.set_yticks(y, labels)
    ax.set_xlabel("score delta vs best individual head")
    ax.set_title("Best validation ensemble and its predictor mix", loc="left")
    for i, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = 0.002 if value >= 0 else -0.002
        ax.text(value + offset, i, f"{value:+.3f}", ha=ha, va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_disprot_focus_decision(
    delta: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    output: Path,
) -> None:
    columns = [column for column in ["disprot\n(AP)", "disprot\n(AUROC)"] if column in delta.columns]
    if not columns:
        return

    plot_delta = delta[columns].copy()
    labels = [
        strategy.replace("_validation", "").replace("_stacking", "")
        for strategy in plot_delta.index
    ]
    y = np.arange(len(plot_delta))
    fig, axes = plt.subplots(1, len(columns), figsize=(13, max(5, 0.45 * len(plot_delta) + 1.6)))
    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):
        values = plot_delta[column].to_numpy(dtype=float)
        colors = [
            "#2ca25f" if value > 1e-6 else "#de2d26" if value < -1e-6 else "#969696"
            for value in values
        ]
        ax.barh(y, values, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.set_xlabel("delta vs DisProt head")
        ax.set_title(display_metric(column), loc="left")
        for i, value in enumerate(values):
            ha = "left" if value >= 0 else "right"
            offset = 0.00012 if value >= 0 else -0.00012
            ax.text(value + offset, i, f"{value:+.4f}", va="center", ha=ha, fontsize=8)
        ax.grid(axis="x", alpha=0.25)

    disprot_summary = strategy_summary[strategy_summary["metric"].isin(columns)]
    if not disprot_summary.empty:
        lines = []
        for _, row in disprot_summary.iterrows():
            combination = row.get("best_ensemble_combination", "")
            lines.append(
                f"{display_metric(row['metric'])}: {row['best_ensemble_strategy']} | {combination}"
            )
        fig.suptitle(
            "DisProt focus: ensemble gain is marginal; best mix is almost the DisProt head",
            x=0.02,
            ha="left",
        )
        fig.text(0.02, 0.01, "\n".join(lines), ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_summary(ensemble_summary: pd.DataFrame, output: Path) -> None:
    summary = ensemble_summary.drop(index="best_individual_test_oracle", errors="ignore")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    y = np.arange(len(summary))
    values = summary["mean_delta_vs_best_individual"].to_numpy(dtype=float)
    colors = ["#2ca25f" if value > 0 else "#de2d26" if value < 0 else "#969696" for value in values]
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, summary.index.tolist())
    ax.set_xlabel("mean delta vs best individual head")
    ax.set_title("No ensemble strategy improves performance broadly", loc="left")
    for i, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = 0.004 if value >= 0 else -0.004
        ax.text(value + offset, i, f"{value:+.3f}", va="center", ha=ha, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_headroom(
    baseline: pd.Series,
    udon: pd.Series,
    ensemble: pd.Series,
    ceiling: pd.Series,
    output: Path,
) -> None:
    stages = ["Simple baseline", "Best UdonPred", "Best ensemble", "Annotation ceiling"]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in METRIC_COLUMNS:
        values = [baseline[metric], udon[metric], ensemble[metric], ceiling[metric]]
        ax.plot(x, values, marker="o", linewidth=1.8, alpha=0.78, label=display_metric(metric))
    ax.set_xticks(x, stages)
    ax.set_ylabel("score")
    ax.set_title("Performance progression and remaining annotation headroom", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udon-matrix", type=Path, default=Path("results/udonpred_matrix/matrix.csv"))
    parser.add_argument(
        "--baseline-matrix",
        type=Path,
        default=Path("results/simple_baselines/matrix.csv"),
    )
    parser.add_argument(
        "--ensemble-matrix",
        type=Path,
        default=Path("results/ensembles/ensemble_matrix.csv"),
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=Path("results/ensembles/ensemble_summary.csv"),
    )
    parser.add_argument(
        "--best-individual-heads",
        type=Path,
        default=Path("results/ensembles/best_individual_heads.csv"),
    )
    parser.add_argument(
        "--ensemble-weights",
        type=Path,
        default=Path("results/ensembles/ensemble_weights.csv"),
    )
    parser.add_argument(
        "--ensemble-subset-choices",
        type=Path,
        default=Path("results/ensembles/ensemble_subset_choices.csv"),
    )
    parser.add_argument(
        "--validation-selected-heads",
        type=Path,
        default=Path("results/ensembles/validation_selected_heads.csv"),
    )
    parser.add_argument(
        "--ceiling-matrix",
        type=Path,
        default=Path("results/normalized_headroom/ceiling_matrix.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/prediction_improvement"),
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        choices=DATASETS,
        help="Exclude datasets from all plots and summaries.",
    )
    parser.add_argument(
        "--replace-trizod-with-updated",
        action="store_true",
        help="Use trizod_updated instead of trizod in all plots and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    global METRIC_COLUMNS
    args = parse_args()
    METRIC_COLUMNS = resolve_metric_columns(
        args.exclude_datasets, args.replace_trizod_with_updated
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_matrix = read_baseline_matrix(args.baseline_matrix)
    udon_matrix = read_udon_matrix(args.udon_matrix)
    ensemble_matrix = read_numeric_matrix(args.ensemble_matrix)
    if not args.ensemble_summary.exists():
        raise FileNotFoundError(
            f"Missing {args.ensemble_summary}. Run the ensemble step first."
        )
    ensemble_summary = pd.read_csv(args.ensemble_summary, index_col="strategy")
    ceiling_matrix = read_numeric_matrix(args.ceiling_matrix)

    best_baseline = best_series(baseline_matrix)
    best_udon = best_series(udon_matrix)
    non_oracle_ensemble = best_ensemble_matrix(ensemble_matrix)
    if "best_individual_test_oracle" in ensemble_matrix.index:
        best_udon = ensemble_matrix.loc["best_individual_test_oracle"].combine_first(best_udon)
    best_ensemble = best_series(non_oracle_ensemble)
    best_ceiling = best_series(ceiling_matrix)
    best_heads = read_best_individual_heads(args.best_individual_heads)
    combinations = read_combination_lookup(
        args.ensemble_weights,
        args.ensemble_subset_choices,
        args.validation_selected_heads,
    )
    strategy_summary = best_strategy_summary(
        non_oracle_ensemble, best_udon, best_heads, combinations
    )

    delta = non_oracle_ensemble.subtract(best_udon, axis="columns")
    delta.to_csv(args.output_dir / "ensemble_delta_vs_best_individual.csv", float_format="%.6f")
    strategy_summary.to_csv(
        args.output_dir / "best_ensemble_strategy_by_metric.csv",
        index=False,
        float_format="%.6f",
    )
    write_improvement_summary(
        best_baseline,
        best_udon,
        best_ensemble,
        best_ceiling,
        args.output_dir / "improvement_summary.csv",
    )
    plot_delta_heatmap(delta, args.output_dir / "ensemble_delta_heatmap.png")
    plot_best_barplot(best_udon, best_ensemble, args.output_dir / "best_individual_vs_ensemble.png")
    plot_metric_delta_bar(
        best_ensemble - best_udon,
        args.output_dir / "best_ensemble_delta_by_metric.png",
    )
    plot_best_strategy_by_metric(
        strategy_summary,
        args.output_dir / "best_ensemble_strategy_by_metric.png",
    )
    plot_best_combination_by_metric(
        strategy_summary,
        args.output_dir / "best_ensemble_combination_by_metric.png",
    )
    plot_disprot_focus_decision(
        delta,
        strategy_summary,
        args.output_dir / "disprot_focus_ensemble_decision.png",
    )
    plot_strategy_summary(
        ensemble_summary,
        args.output_dir / "ensemble_strategy_mean_delta.png",
    )
    plot_headroom(
        best_baseline,
        best_udon,
        best_ensemble,
        best_ceiling,
        args.output_dir / "baseline_udon_ensemble_ceiling.png",
    )
    print(f"Wrote prediction-improvement figures to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
