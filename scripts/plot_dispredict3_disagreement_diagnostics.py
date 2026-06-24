#!/usr/bin/env python3
"""Plot DisPredict3 disagreement diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTOR_COLORS = {
    "DisoFLAG": "#2a9d8f",
    "DisorderUnetLM": "#1f9e89",
    "PUNCH2_light": "#56b4a5",
    "ADOPT": "#76b7b2",
    "disprot": "#4c78a8",
    "plddt": "#6f8fbd",
    "trizod": "#5279ad",
    "chezod": "#7a9cc6",
    "softdis": "#3f6f9f",
    "atlas": "#5d85b5",
}

DIRECTION_COLORS = {
    "DisPredict3 higher": "#c44e52",
    "balanced": "#777777",
}


def predictor_color(name: str) -> str:
    return PREDICTOR_COLORS.get(name, "#777777")


def read_csv(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_segment_overlap(rows: pd.DataFrame, output_dir: Path) -> None:
    rows = rows.sort_values("fraction_of_dispredict3_high_shared", ascending=False)
    predictors = rows["predictor"].tolist()
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    axes[0].barh(
        y,
        rows["fraction_of_dispredict3_high_shared"],
        color=[predictor_color(name) for name in predictors],
        edgecolor="white",
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(predictors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("fraction of DisPredict3 high-score residues also high in predictor")
    axes[0].set_title("Shared DisPredict3 high-score residues", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#d9d9d9", linewidth=0.7)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    for index, value in enumerate(rows["fraction_of_dispredict3_high_shared"]):
        axes[0].text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9)

    width = 0.35
    x = np.arange(len(rows))
    axes[1].bar(
        x - width / 2,
        rows["residue_jaccard"],
        width,
        color="#4c78a8",
        label="Residue Jaccard",
    )
    axes[1].bar(
        x + width / 2,
        rows["mean_per_protein_jaccard"],
        width,
        color="#2a9d8f",
        label="Mean protein Jaccard",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(predictors, rotation=35, ha="right")
    axes[1].set_ylim(0, max(0.4, float(rows[["residue_jaccard", "mean_per_protein_jaccard"]].max().max()) * 1.25))
    axes[1].set_ylabel("Jaccard of top-score residue sets")
    axes[1].set_title("High-score residue overlap", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)

    save(fig, output_dir / "diagnostic_segment_overlap.png")


def plot_protein_disagreement(rows: pd.DataFrame, output_dir: Path) -> None:
    predictors = (
        rows.groupby("predictor")["zscore_mae"]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    data = [rows.loc[rows["predictor"] == name, "zscore_mae"].dropna().to_numpy() for name in predictors]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    box = axes[0].boxplot(
        data,
        tick_labels=predictors,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for patch, name in zip(box["boxes"], predictors):
        patch.set_facecolor(predictor_color(name))
        patch.set_alpha(0.9)
        patch.set_edgecolor("white")
    axes[0].set_ylabel("per-protein z-score MAE")
    axes[0].set_title("Which predictors disagree most per protein?", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)

    summary = (
        rows.groupby("predictor")[
            [
                "dispredict3_z_high_other_z_low_fraction",
                "dispredict3_z_low_other_z_high_fraction",
            ]
        ]
        .mean()
        .loc[predictors]
    )
    x = np.arange(len(summary))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        summary["dispredict3_z_high_other_z_low_fraction"],
        width,
        color="#c44e52",
        label="DisPredict3 high, other low",
    )
    axes[1].bar(
        x + width / 2,
        summary["dispredict3_z_low_other_z_high_fraction"],
        width,
        color="#4c78a8",
        label="DisPredict3 low, other high",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary.index, rotation=35, ha="right")
    axes[1].set_ylabel("mean fraction of residues per protein")
    axes[1].set_title("Direction of disagreement", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)

    save(fig, output_dir / "diagnostic_protein_disagreement.png")


def plot_top_regions(rows: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    predictors = sorted(rows["predictor"].unique())
    ncols = 2
    nrows = int(np.ceil(len(predictors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 3.2 * nrows)), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)

    for ax, predictor in zip(axes_array, predictors):
        subset = (
            rows.loc[rows["predictor"] == predictor]
            .sort_values("mean_abs_zscore_difference", ascending=True)
            .tail(top_n)
            .copy()
        )
        labels = [
            f"{protein}:{int(start)}-{int(end)}"
            for protein, start, end in zip(subset["protein_id"], subset["start_residue"], subset["end_residue"])
        ]
        colors = [
            DIRECTION_COLORS.get(direction, "#4c78a8")
            for direction in subset["direction"]
        ]
        y = np.arange(len(subset))
        ax.barh(y, subset["mean_abs_zscore_difference"], color=colors, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(predictor, loc="left", fontweight="bold")
        ax.set_xlabel("mean abs z-score difference in window")
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes_array[len(predictors) :]:
        ax.axis("off")

    fig.suptitle("Top local disagreement windows", fontweight="bold")
    save(fig, output_dir / "diagnostic_top_regions.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/dispredict3_disagreement_diagnostics_key_predictors"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --input-dir.",
    )
    parser.add_argument("--top-regions-per-predictor", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir

    segment_rows = read_csv(args.input_dir, "high_score_segment_overlap.csv")
    protein_rows = read_csv(args.input_dir, "per_protein_disagreement.csv")
    region_rows = read_csv(args.input_dir, "top_disagreement_regions.csv")

    plot_segment_overlap(segment_rows, output_dir)
    plot_protein_disagreement(protein_rows, output_dir)
    plot_top_regions(region_rows, output_dir, top_n=args.top_regions_per_predictor)


if __name__ == "__main__":
    main()
