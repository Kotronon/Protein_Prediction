#!/usr/bin/env python3
"""Plot DisProt subset ensemble candidates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteinprediction-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREDICTOR_COLORS = {
    "chezod": "#4c78a8",
    "softdis": "#59a14f",
    "plddt": "#f28e2b",
    "disprot": "#e15759",
    "trizod_updated": "#b07aa1",
    "trizod": "#b07aa1",
}


def candidate_label(row: pd.Series) -> str:
    return f"{row['strategy']} {int(row['n_heads'])}x | {str(row['heads']).replace('+', ' + ')}"


def predictor_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {
        "strategy",
        "n_heads",
        "heads",
        "validation_AP",
        "test_AP",
        "test_AUROC",
        "delta_AP_vs_disprot_head",
        "delta_AUROC_vs_disprot_head",
    }
    return [column for column in frame.columns if column not in metadata]


def plot_delta_bar(frame: pd.DataFrame, output: Path) -> None:
    plot = frame.sort_values("delta_AP_vs_disprot_head", ascending=True).copy()
    values = plot["delta_AP_vs_disprot_head"].to_numpy(dtype=float)
    colors = [
        "#2ca25f" if value > 1e-6 else "#de2d26" if value < -1e-6 else "#969696"
        for value in values
    ]
    labels = [candidate_label(row) for _, row in plot.iterrows()]

    fig_height = max(6, 0.36 * len(plot) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(plot))
    ax.barh(y, values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("DisProt AP delta vs single DisProt head")
    ax.set_title("DisProt subset candidates: only near-DisProt-only convex mixes improve", loc="left")
    for i, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = 0.00045 if value >= 0 else -0.00045
        ax.text(value + offset, i, f"{value:+.4f}", va="center", ha=ha, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_weight_stack(frame: pd.DataFrame, output: Path, top_n: int) -> None:
    columns = predictor_columns(frame)
    plot = frame.sort_values("test_AP", ascending=False).head(top_n).copy()
    labels = [candidate_label(row) for _, row in plot.iterrows()]
    y = np.arange(len(plot))

    fig_height = max(4.8, 0.48 * len(plot) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    left = np.zeros(len(plot))
    for column in columns:
        values = plot[column].fillna(0).to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            label=column,
            color=PREDICTOR_COLORS.get(column),
        )
        left += values
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("ensemble weight")
    ax.set_title(f"Top {len(plot)} DisProt candidates by AP: learned weights", loc="left")
    ax.legend(frameon=False, ncol=min(len(columns), 4), loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ap_vs_complexity(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for strategy, marker, color in [
        ("convex", "o", "#2ca25f"),
        ("mean", "s", "#de2d26"),
    ]:
        subset = frame[frame["strategy"] == strategy]
        ax.scatter(
            subset["n_heads"],
            subset["delta_AP_vs_disprot_head"],
            marker=marker,
            s=70,
            color=color,
            label=strategy,
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(sorted(frame["n_heads"].unique()))
    ax.set_xlabel("number of predictors in ensemble")
    ax.set_ylabel("DisProt AP delta vs single DisProt head")
    ax.set_title("More predictors do not imply better DisProt performance", loc="left")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "results/figures/prediction_improvement_trizod_updated_without_atlas_pdbflex_trizod_updated_focus_disprot/disprot_subset_candidates.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.candidates.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.candidates)

    plot_delta_bar(frame, output_dir / "disprot_subset_delta_ap_bar.png")
    plot_weight_stack(frame, output_dir / "disprot_subset_top_weights.png", args.top_n)
    plot_ap_vs_complexity(frame, output_dir / "disprot_subset_ap_vs_complexity.png")
    print(f"Wrote DisProt subset candidate plots to {output_dir}")


if __name__ == "__main__":
    main()
