#!/usr/bin/env python3
"""Plot compact DisPredict3 comparison diagnostics from the summary CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


GROUP_COLORS = {
    "external": "#2a9d8f",
    "UdonPred": "#4c78a8",
}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


def sorted_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(
        ["group", "mean_per_protein_residue_spearman"],
        ascending=[True, False],
    ).reset_index(drop=True)


def plot_metric_barh(
    ax: plt.Axes,
    rows: pd.DataFrame,
    metric: str,
    error: str | None,
    title: str,
    xlabel: str,
    invert_x: bool = False,
) -> None:
    y = np.arange(len(rows))
    colors = [GROUP_COLORS.get(group, "#777777") for group in rows["group"]]
    xerr = rows[error].to_numpy() if error else None
    metric_values = rows[metric].to_numpy()
    ax.barh(
        y,
        metric_values,
        xerr=xerr,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["predictor"])
    ax.invert_yaxis()
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)

    finite_values = metric_values[np.isfinite(metric_values)]
    if len(finite_values):
        span = max(float(np.max(finite_values) - np.min(finite_values)), 0.05)
        pad = span * 0.04
        right = float(np.max(finite_values + (xerr if xerr is not None else 0))) + span * 0.16
        left = min(0.0, float(np.min(finite_values)) - span * 0.05)
        ax.set_xlim(left, right)

        for row_index, row in rows.iterrows():
            value = row[metric]
            if not np.isfinite(value):
                continue
            if error:
                label = f"{value:.3f} +/- {row[error]:.3f}"
                label_x = value + row[error] + pad
            else:
                label = f"{value:.3f}"
                label_x = value + pad
            ax.text(
                label_x,
                row_index,
                label,
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
            )


def plot_score_ranges(ax: plt.Axes, score_rows: pd.DataFrame) -> None:
    rows = score_rows.sort_values("protein_mean_score_mean", ascending=False).reset_index(drop=True)
    y = np.arange(len(rows))
    low = rows["protein_mean_score_mean"] - rows["score_p05"]
    high = rows["score_p95"] - rows["protein_mean_score_mean"]
    xerr = np.vstack([np.maximum(low, 0), np.maximum(high, 0)])
    ax.errorbar(
        rows["protein_mean_score_mean"],
        y,
        xerr=xerr,
        fmt="o",
        color="#333333",
        ecolor="#b5b5b5",
        elinewidth=2,
        capsize=2,
        markersize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["predictor"])
    ax.invert_yaxis()
    ax.set_title("Score scale and spread", loc="left", fontweight="bold")
    ax.set_xlabel("mean protein score with residue p05-p95")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def plot_scatter(ax: plt.Axes, rows: pd.DataFrame) -> None:
    for group, group_rows in rows.groupby("group", sort=False):
        ax.scatter(
            group_rows["mean_per_protein_residue_spearman"],
            group_rows["protein_mean_score_spearman"],
            s=70,
            color=GROUP_COLORS.get(group, "#777777"),
            edgecolor="white",
            linewidth=0.8,
            label=group,
            alpha=0.95,
        )
        for _, row in group_rows.iterrows():
            ax.annotate(
                row["predictor"],
                (row["mean_per_protein_residue_spearman"], row["protein_mean_score_spearman"]),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("Residue pattern vs protein ranking", loc="left", fontweight="bold")
    ax.set_xlabel("mean per-protein residue Spearman")
    ax.set_ylabel("protein mean-score Spearman")
    ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")


def add_zscore_mae_reference_lines(ax: plt.Axes) -> None:
    """Add interpretable anchors for MAE between two standardized score vectors."""
    references = [
        2 * np.sqrt((1 - 0.75) / np.pi),
        2 * np.sqrt((1 - 0.50) / np.pi),
    ]
    for value in references:
        ax.axvline(value, color="#555555", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.set_xlim(ax.get_xlim()[0], max(ax.get_xlim()[1], 0.85))
    reference_handle = Line2D(
        [0],
        [0],
        color="#555555",
        linestyle=":",
        linewidth=1.0,
        label="z-score MAE anchors: rho 0.75 ~= 0.56, rho 0.50 ~= 0.80, independent ~= 1.13",
    )
    ax.legend(
        handles=[reference_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        fontsize=8,
    )


def make_summary_panel(pair_rows: pd.DataFrame, score_rows: pd.DataFrame, output_path: Path) -> None:
    rows = sorted_pair_rows(pair_rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    plot_metric_barh(
        axes[0, 0],
        rows,
        "mean_per_protein_residue_spearman",
        "mean_per_protein_residue_spearman_se",
        "Residue-level agreement to DisPredict3",
        "mean Spearman +/- SE over proteins",
    )
    plot_metric_barh(
        axes[0, 1],
        rows,
        "protein_mean_score_spearman",
        None,
        "Protein-ranking agreement",
        "Spearman over protein mean scores",
    )
    plot_metric_barh(
        axes[1, 0],
        rows.sort_values("mean_per_protein_zscore_mae", ascending=True).reset_index(drop=True),
        "mean_per_protein_zscore_mae",
        "mean_per_protein_zscore_mae_se",
        "Scaled score disagreement",
        "mean z-score MAE +/- SE; lower is closer",
    )
    add_zscore_mae_reference_lines(axes[1, 0])
    plot_scatter(axes[1, 1], rows)

    fig.suptitle("DisPredict3 compared with UdonPred and CAID-style predictors", fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    score_output = output_path.with_name("dispredict3_score_scale_ranges.png")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_score_ranges(ax, score_rows)
    fig.savefig(score_output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/compare_predictors_with_all_caid_predictors_wo_pdbflex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/compare_predictors_with_all_caid_predictors_wo_pdbflex/"
            "dispredict3_diagnostic_panel.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_required_csv(args.input_dir / "dispredict3_vs_predictors_proteinwise_se.csv")
    score_rows = read_required_csv(args.input_dir / "dispredict3_predictor_score_distribution.csv")
    make_summary_panel(pair_rows, score_rows, args.output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_name('dispredict3_score_scale_ranges.png')}")


if __name__ == "__main__":
    main()
