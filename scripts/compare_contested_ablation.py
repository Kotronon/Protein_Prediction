#!/usr/bin/env python3
"""Compare two contested-region runs, e.g. with and without PDBFlex."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_windows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path / "contested_windows.csv")
    table["region"] = (
        table["protein_id"].astype(str)
        + ":"
        + table["start"].astype(str)
        + "-"
        + table["end"].astype(str)
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("results/contested_regions/plddt"))
    parser.add_argument("--ablation-dir", type=Path, default=Path("results/contested_regions/plddt_no_pdbflex"))
    parser.add_argument("--baseline-label", default="with_pdbflex")
    parser.add_argument("--ablation-label", default="without_pdbflex")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contested_regions/pdbflex_ablation_comparison"))
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_windows(args.baseline_dir)
    ablation = load_windows(args.ablation_dir)

    metrics = [
        "priority_score",
        "mean_disagreement",
        "boundary_uncertainty",
        "nmr_suitability",
        "concept_family_spread_z",
        "possible_more_score",
        "tmh_like_score",
    ]
    summary = pd.DataFrame(
        {
            args.baseline_label: base[metrics].median(),
            args.ablation_label: ablation[metrics].median(),
        }
    )
    summary[f"delta_{args.ablation_label}_minus_{args.baseline_label}"] = (
        summary[args.ablation_label] - summary[args.baseline_label]
    )
    summary.to_csv(args.output_dir / "score_summary.csv")

    base_top = base.nlargest(args.top_n, "priority_score").copy()
    ablation_top = ablation.nlargest(args.top_n, "priority_score").copy()
    overlap = sorted(set(base_top["region"]) & set(ablation_top["region"]))
    overlap_summary = pd.DataFrame(
        [
            {
                "top_n": args.top_n,
                "overlap_n": len(overlap),
                "union_n": len(set(base_top["region"]) | set(ablation_top["region"])),
                "jaccard": len(overlap) / len(set(base_top["region"]) | set(ablation_top["region"])),
            }
        ]
    )
    overlap_summary.to_csv(args.output_dir / "top_region_overlap_summary.csv", index=False)
    pd.DataFrame({"shared_top_region": overlap}).to_csv(args.output_dir / "shared_top_regions.csv", index=False)

    base_rank = base.sort_values("priority_score", ascending=False).reset_index(drop=True)
    base_rank[f"rank_{args.baseline_label}"] = np.arange(1, len(base_rank) + 1)
    ablation_rank = ablation.sort_values("priority_score", ascending=False).reset_index(drop=True)
    ablation_rank[f"rank_{args.ablation_label}"] = np.arange(1, len(ablation_rank) + 1)
    merged = base_rank[
        [
            "region",
            f"rank_{args.baseline_label}",
            "priority_score",
            "mean_disagreement",
            "disagreement_type",
            "top_predictor",
            "bottom_predictor",
        ]
    ].merge(
        ablation_rank[
            [
                "region",
                f"rank_{args.ablation_label}",
                "priority_score",
                "mean_disagreement",
                "disagreement_type",
                "top_predictor",
                "bottom_predictor",
            ]
        ],
        on="region",
        suffixes=(f"_{args.baseline_label}", f"_{args.ablation_label}"),
    )
    merged[f"rank_delta_{args.ablation_label}_minus_{args.baseline_label}"] = (
        merged[f"rank_{args.ablation_label}"] - merged[f"rank_{args.baseline_label}"]
    )
    merged.to_csv(args.output_dir / "region_rank_comparison.csv", index=False)
    merged.sort_values(f"rank_delta_{args.ablation_label}_minus_{args.baseline_label}", ascending=False).head(50).to_csv(
        args.output_dir / "largest_rank_drops_without_pdbflex.csv", index=False
    )
    merged.sort_values(f"rank_delta_{args.ablation_label}_minus_{args.baseline_label}", ascending=True).head(50).to_csv(
        args.output_dir / "largest_rank_gains_without_pdbflex.csv", index=False
    )

    type_counts = pd.concat(
        [
            base["disagreement_type"].value_counts().rename(args.baseline_label),
            ablation["disagreement_type"].value_counts().rename(args.ablation_label),
        ],
        axis=1,
    ).fillna(0).astype(int)
    type_counts["delta_without_minus_with"] = type_counts[args.ablation_label] - type_counts[args.baseline_label]
    type_counts.to_csv(args.output_dir / "disagreement_type_count_comparison.csv")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_summary = summary.loc[["priority_score", "mean_disagreement", "concept_family_spread_z", "possible_more_score"]]
    x = np.arange(len(plot_summary))
    width = 0.36
    ax.bar(x - width / 2, plot_summary[args.baseline_label], width=width, label=args.baseline_label, color="#4C78A8")
    ax.bar(x + width / 2, plot_summary[args.ablation_label], width=width, label=args.ablation_label, color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_summary.index, rotation=20, ha="right")
    ax.set_ylabel("Median value")
    ax.set_title("Removing PDBFlex lowers disagreement and concept-family spread")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "score_summary_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(
        merged[f"rank_{args.baseline_label}"],
        merged[f"rank_{args.ablation_label}"],
        s=10,
        alpha=0.45,
        color="#4C78A8",
    )
    limit = max(len(base), len(ablation))
    ax.plot([1, limit], [1, limit], color="black", lw=1, alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Rank {args.baseline_label}")
    ax.set_ylabel(f"Rank {args.ablation_label}")
    ax.set_title("Rank stability after removing PDBFlex")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rank_stability_scatter.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    type_counts_sorted = type_counts.sort_values(args.baseline_label)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(type_counts_sorted))
    ax.barh(y - 0.18, type_counts_sorted[args.baseline_label], height=0.34, label=args.baseline_label, color="#4C78A8")
    ax.barh(y + 0.18, type_counts_sorted[args.ablation_label], height=0.34, label=args.ablation_label, color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(type_counts_sorted.index)
    ax.set_xlabel("Number of windows")
    ax.set_title("Disagreement classes with and without PDBFlex")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "disagreement_type_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Wrote comparison outputs to {args.output_dir}")
    print(overlap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
