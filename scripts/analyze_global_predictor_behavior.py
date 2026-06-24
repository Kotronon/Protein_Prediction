#!/usr/bin/env python3
"""Analyze predictor clusters, annotation agreement, and protein-length effects.

This complements ``compare_predictors.py``. It deliberately treats pooled
residue correlation and protein-level correlation as different summaries and
uses within-predictor ranks when scores from incompatible scales are combined.
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from compare_predictors import load_predictions, maybe_negate_scores, parse_predictor_argument


UDONPRED_MODELS = {"trizod", "chezod", "softdis", "atlas", "plddt", "disprot"}
ALL_UDONPRED_MODELS = UDONPRED_MODELS | {"pdbflex"}
CAID_STYLE = {"DisPredict3", "DisoFLAG", "DisorderUnetLM", "PUNCH2_light"}
CLASSICAL = {"SETH", "IUPred3", "ADOPT", "metapredict"}
DEFAULT_NEGATED = {"chezod", "plddt", "adopt"}
DEFAULT_PREDICTORS = {
    "trizod": Path("results/human_proteome/UdonPred/trizod"),
    "chezod": Path("results/human_proteome/UdonPred/chezod"),
    "softdis": Path("results/human_proteome/UdonPred/softdis"),
    "atlas": Path("results/human_proteome/UdonPred/atlas"),
    "plddt": Path("results/human_proteome/UdonPred/plddt"),
    "disprot": Path("results/human_proteome/UdonPred/disprot"),
    "SETH": Path("results/human_proteome/SETH/seth_human_proteome.caid"),
    "IUPred3": Path("results/human_proteome/IUPred3"),
    "ADOPT": Path("results/human_proteome/ADOPT"),
    "metapredict": Path("results/human_proteome/metapredict/metapredict_human_proteome.caid"),
    "PUNCH2_light": Path("results/human_proteome/PUNCH2_light/disorder"),
    "DisoFLAG": Path("results/human_proteome/DisoFLAG/caid"),
    "DisorderUnetLM": Path("results/human_proteome/DisorderUnetLM/disorder"),
    "DisPredict3": Path("results/human_proteome/Dispredict3_native/caid"),
}
GROUP_COLORS = {
    "UdonPred": "#4c78a8",
    "CAID-style external": "#e45756",
    "classical external": "#54a24b",
    "other": "#9d9d9d",
}
LENGTH_BINS = [0, 200, 500, 1000, np.inf]
LENGTH_LABELS = ["<200", "200-499", "500-999", ">=1000"]


def predictor_group(name: str) -> str:
    if name in ALL_UDONPRED_MODELS:
        return "UdonPred"
    if name in CAID_STYLE:
        return "CAID-style external"
    if name in CLASSICAL:
        return "classical external"
    return "other"


def pairwise_matrix(pairwise: pd.DataFrame, value_column: str) -> pd.DataFrame:
    names = sorted(set(pairwise["predictor_a"]) | set(pairwise["predictor_b"]))
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for name in names:
        matrix.loc[name, name] = 1.0
    for row in pairwise.itertuples(index=False):
        value = getattr(row, value_column)
        matrix.loc[row.predictor_a, row.predictor_b] = value
        matrix.loc[row.predictor_b, row.predictor_a] = value
    return matrix


def cluster_tree(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.isna().any().any():
        missing = matrix.index[matrix.isna().any(axis=1)].tolist()
        raise ValueError(f"Cannot cluster incomplete agreement matrix; missing values for {missing}")
    distance = np.clip(1.0 - matrix.to_numpy(dtype=float), 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=True), method="average", optimal_ordering=True)


def cluster_order(matrix: pd.DataFrame) -> list[str]:
    tree = cluster_tree(matrix)
    return matrix.index[leaves_list(tree)].tolist()


def plot_cluster(matrix: pd.DataFrame, metric_label: str, output: Path) -> list[str]:
    tree = cluster_tree(matrix)
    order = matrix.index[leaves_list(tree)].tolist()
    row_colors = pd.Series(
        [GROUP_COLORS[predictor_group(name)] for name in matrix.index], index=matrix.index
    )
    grid = sns.clustermap(
        matrix,
        row_linkage=tree,
        col_linkage=tree,
        row_colors=row_colors,
        cmap="viridis",
        vmin=float(np.nanmin(matrix.to_numpy())),
        vmax=1.0,
        annot=True,
        fmt=".2f",
        figsize=(10, 9),
        cbar_kws={"label": "Spearman correlation"},
    )
    grid.fig.suptitle(f"Hierarchical predictor clustering: {metric_label}", y=1.02)
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="x", rotation=45)
    grid.ax_heatmap.legend(
        handles=[Patch(facecolor=color, label=group) for group, color in GROUP_COLORS.items()
                 if group != "other"],
        title="Predictor group",
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        frameon=False,
    )
    grid.fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)
    return order


def group_agreement_rows(pairwise: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for metric in ("residue_spearman", "protein_spearman"):
        work = pairwise[["predictor_a", "predictor_b", metric]].copy()
        work["group_a"] = work["predictor_a"].map(predictor_group)
        work["group_b"] = work["predictor_b"].map(predictor_group)
        work["group_pair"] = work.apply(
            lambda row: " vs ".join(sorted((row["group_a"], row["group_b"]))), axis=1
        )
        for group_pair, values in work.groupby("group_pair")[metric]:
            rows.append(
                {
                    "metric": metric,
                    "group_pair": group_pair,
                    "n_predictor_pairs": int(values.notna().sum()),
                    "mean_spearman": float(values.mean()),
                    "min_spearman": float(values.min()),
                    "max_spearman": float(values.max()),
                }
            )
    return rows


def record_summary(records, direction: str) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    means: dict[str, float] = {}
    lengths: dict[str, int] = {}
    finite_min = math.inf
    finite_max = -math.inf
    nonfinite = 0
    residues = 0
    for protein_id, record in records.items():
        finite = record.scores[np.isfinite(record.scores)]
        nonfinite += int(len(record.scores) - len(finite))
        residues += len(record.scores)
        lengths[protein_id] = len(record.scores)
        if len(finite):
            means[protein_id] = float(np.mean(finite))
            finite_min = min(finite_min, float(np.min(finite)))
            finite_max = max(finite_max, float(np.max(finite)))
    stats = {
        "proteins": len(records),
        "residues": residues,
        "nonfinite_scores": nonfinite,
        "oriented_score_min": finite_min if math.isfinite(finite_min) else math.nan,
        "oriented_score_max": finite_max if math.isfinite(finite_max) else math.nan,
        "score_direction_after_transform": "higher = more disorder",
        "score_transform": direction,
        "absolute_0_5_threshold_used": False,
        "cross_predictor_threshold_policy": "within-predictor percentile/quantile",
    }
    return pd.Series(means, dtype=float), pd.Series(lengths, dtype="int64"), stats


def qc_against_reference(reference, records, name: str, stats: dict[str, object]) -> dict[str, object]:
    reference_ids = set(reference)
    predictor_ids = set(records)
    common = sorted(reference_ids & predictor_ids)
    length_mismatches = 0
    overlap_sequence_mismatches = 0
    exact_sequence_mismatches = 0
    for protein_id in common:
        left = reference[protein_id]
        right = records[protein_id]
        length = min(len(left.scores), len(right.scores))
        length_mismatches += len(left.scores) != len(right.scores)
        overlap_sequence_mismatches += left.sequence[:length] != right.sequence[:length]
        exact_sequence_mismatches += left.sequence != right.sequence
    return {
        "predictor": name,
        **stats,
        "reference_predictor": "trizod",
        "common_proteins_with_reference": len(common),
        "only_reference": len(reference_ids - predictor_ids),
        "only_predictor": len(predictor_ids - reference_ids),
        "length_mismatched_common_proteins": length_mismatches,
        "overlap_sequence_mismatched_proteins": overlap_sequence_mismatches,
        "exact_sequence_mismatched_proteins": exact_sequence_mismatches,
    }


def load_protein_summaries(
    paths: dict[str, Path], negated: set[str]
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, object]]]:
    reference = load_predictions(paths["trizod"])
    mean_columns: dict[str, pd.Series] = {}
    lengths: pd.Series | None = None
    qc_rows = []
    for name, path in paths.items():
        raw = reference if name == "trizod" else load_predictions(path)
        direction = "multiply by -1" if name.lower() in negated else "none"
        records = maybe_negate_scores(raw, name, negated)
        means, predictor_lengths, stats = record_summary(records, direction)
        mean_columns[name] = means
        if name == "trizod":
            lengths = predictor_lengths
        qc_rows.append(qc_against_reference(reference, records, name, stats))
    assert lengths is not None
    return pd.DataFrame(mean_columns), lengths.rename("protein_length"), qc_rows


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic)


def length_effect_tables(
    means: pd.DataFrame, lengths: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = means.join(lengths, how="inner")
    bins = pd.cut(
        common["protein_length"], LENGTH_BINS, labels=LENGTH_LABELS, right=False
    )
    predictor_rows = []
    for predictor in means.columns:
        values = common[[predictor, "protein_length"]].dropna()
        predictor_rows.append(
            {
                "predictor": predictor,
                "group": predictor_group(predictor),
                "n_proteins": len(values),
                "length_vs_mean_score_spearman": safe_spearman(
                    values["protein_length"], values[predictor]
                ),
            }
        )

    percentile_means = means.rank(pct=True, axis=0, method="average")
    consensus = percentile_means.mean(axis=1, skipna=True).rename("mean_disorder_percentile")
    variance = percentile_means.std(axis=1, skipna=True, ddof=0).rename("predictor_disagreement")
    protein_table = pd.concat([lengths, consensus, variance], axis=1).dropna()
    protein_table["length_bin"] = pd.cut(
        protein_table["protein_length"], LENGTH_BINS, labels=LENGTH_LABELS, right=False
    )

    bin_rows = []
    for label in LENGTH_LABELS:
        subset = protein_table[protein_table["length_bin"] == label]
        bin_rows.append(
            {
                "length_bin": label,
                "n_proteins": len(subset),
                "median_length": float(subset["protein_length"].median()),
                "mean_disorder_percentile": float(subset["mean_disorder_percentile"].mean()),
                "mean_predictor_disagreement": float(subset["predictor_disagreement"].mean()),
            }
        )

    agreement_rows = []
    for left, right in combinations(means.columns, 2):
        pair = common[[left, right]].copy()
        pair["length_bin"] = bins
        for label in LENGTH_LABELS:
            subset = pair[pair["length_bin"] == label][[left, right]].dropna()
            agreement_rows.append(
                {
                    "predictor_a": left,
                    "predictor_b": right,
                    "length_bin": label,
                    "n_common_proteins": len(subset),
                    "protein_mean_score_spearman": safe_spearman(subset[left], subset[right]),
                }
            )
    return (
        pd.DataFrame(predictor_rows),
        pd.DataFrame(bin_rows),
        pd.DataFrame(agreement_rows),
        protein_table,
    )


def binned_predictor_scores(means: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    percentiles = means.rank(pct=True, axis=0, method="average").join(lengths, how="inner")
    percentiles["length_bin"] = pd.cut(
        percentiles["protein_length"], LENGTH_BINS, labels=LENGTH_LABELS, right=False
    )
    rows = []
    for predictor in means.columns:
        for label in LENGTH_LABELS:
            values = percentiles.loc[percentiles["length_bin"] == label, predictor].dropna()
            rows.append(
                {
                    "predictor": predictor,
                    "group": predictor_group(predictor),
                    "length_bin": label,
                    "n_proteins": len(values),
                    "mean_protein_score_percentile": float(values.mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_binned_predictor_scores(rows: pd.DataFrame, output: Path) -> None:
    groups = ["UdonPred", "CAID-style external", "classical external"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, group in zip(axes, groups):
        subset = rows[rows["group"] == group]
        sns.lineplot(
            data=subset,
            x="length_bin",
            y="mean_protein_score_percentile",
            hue="predictor",
            marker="o",
            ax=ax,
        )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_title(group, loc="left")
        ax.set_xlabel("Protein length bin")
        ax.set_ylabel("Mean within-predictor score percentile" if ax is axes[0] else "")
        ax.legend(title="", fontsize=8)
    fig.suptitle("Protein length vs mean disorder-score rank", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_length_effects(
    predictor_effects: pd.DataFrame,
    bin_summary: pd.DataFrame,
    protein_table: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ordered = predictor_effects.sort_values("length_vs_mean_score_spearman")
    axes[0].barh(
        ordered["predictor"],
        ordered["length_vs_mean_score_spearman"],
        color=[GROUP_COLORS[group] for group in ordered["group"]],
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Spearman: length vs mean score")
    axes[0].set_title("Predictor-specific length effects", loc="left")

    sample = protein_table.sample(min(10000, len(protein_table)), random_state=13)
    axes[1].scatter(
        sample["protein_length"], sample["predictor_disagreement"], s=5, alpha=0.18
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Protein length (log scale)")
    axes[1].set_ylabel("SD of predictor percentiles")
    axes[1].set_title("Length and predictor disagreement", loc="left")

    sns.barplot(
        data=bin_summary,
        x="length_bin",
        y="mean_predictor_disagreement",
        color="#4c78a8",
        ax=axes[2],
    )
    for index, row in bin_summary.iterrows():
        axes[2].text(index, row["mean_predictor_disagreement"], f"n={row['n_proteins']}",
                     ha="center", va="bottom", fontsize=8)
    axes[2].set_xlabel("Protein length bin")
    axes[2].set_ylabel("Mean SD of predictor percentiles")
    axes[2].set_title("Disagreement by length class", loc="left")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def annotation_comparison(
    pairwise: pd.DataFrame, annotation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictor_lookup = {}
    for row in pairwise.itertuples(index=False):
        predictor_lookup[frozenset((row.predictor_a, row.predictor_b))] = row.residue_spearman

    rows = []
    pairwise_names = set(pairwise["predictor_a"]) | set(pairwise["predictor_b"])
    models = sorted(ALL_UDONPRED_MODELS & pairwise_names)
    exact = annotation[annotation["comparison_level"] == "exact"].copy()
    for left, right in combinations(models, 2):
        metric = "auroc" if "disprot" in (left, right) else "spearman"
        candidates = exact[
            (exact["metric"] == metric)
            & exact.apply(
                lambda row: {row["dataset_a"], row["dataset_b"]} == {left, right}, axis=1
            )
        ]
        annotation_row = candidates.iloc[0] if len(candidates) else None
        rows.append(
            {
                "dataset_a": left,
                "dataset_b": right,
                "predictor_residue_spearman": predictor_lookup.get(frozenset((left, right)), math.nan),
                "annotation_metric": metric,
                "annotation_agreement": annotation_row["value"] if annotation_row is not None else math.nan,
                "n_annotation_overlap_proteins": (
                    annotation_row["n_proteins_overlap"] if annotation_row is not None else 0
                ),
                "n_annotation_residues": (
                    annotation_row["n_residues_compared"] if annotation_row is not None else 0
                ),
                "interpretation_scope": "descriptive; annotation metrics and overlap sizes differ",
            }
        )
    result = pd.DataFrame(rows)
    comparable = result.dropna(subset=["predictor_residue_spearman", "annotation_agreement"])
    continuous = comparable[comparable["annotation_metric"] == "spearman"]
    summary = {
        "n_primary_pairs": float(len(comparable)),
        "all_primary_pair_spearman": safe_spearman(
            comparable["annotation_agreement"], comparable["predictor_residue_spearman"]
        ),
        "n_continuous_pairs": float(len(continuous)),
        "continuous_pair_spearman": safe_spearman(
            continuous["annotation_agreement"], continuous["predictor_residue_spearman"]
        ),
    }
    return result, summary


def plot_annotation_comparison(rows: pd.DataFrame, output: Path) -> None:
    usable = rows.dropna(subset=["annotation_agreement", "predictor_residue_spearman"]).copy()
    sizes = 25 + 35 * np.log10(usable["n_annotation_residues"].clip(lower=1))
    colors = usable["annotation_metric"].map({"spearman": "#4c78a8", "auroc": "#e45756"})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        usable["annotation_agreement"], usable["predictor_residue_spearman"],
        s=sizes, c=colors, alpha=0.75, edgecolor="white",
    )
    for row in usable.itertuples(index=False):
        ax.annotate(
            f"{row.dataset_a}-{row.dataset_b}\n(n={int(row.n_annotation_overlap_proteins)})",
            (row.annotation_agreement, row.predictor_residue_spearman),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )
    ax.set_xlabel("Experimental annotation agreement (pair-specific primary metric)")
    ax.set_ylabel("Human-proteome predictor Spearman")
    ax.set_title("Training annotations vs model behavior (descriptive)", loc="left")
    ax.text(
        0.01, -0.17,
        "Point size reflects compared annotation residues. DisProt pairs use AUROC; continuous pairs use Spearman.",
        transform=ax.transAxes, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary: dict[str, float], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["statistic", "value"])
        writer.writerows(summary.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairwise-csv", type=Path,
        default=Path("results/compare_predictors_with_all_predictors_wo_pdbflex/pairwise_agreement.csv"),
    )
    parser.add_argument(
        "--annotation-csv", type=Path,
        default=Path("results/annotation_ceiling/annotation_ceiling_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/compare_predictors_with_all_predictors_wo_pdbflex/global_behavior"),
    )
    parser.add_argument(
        "--predictor", action="append", type=parse_predictor_argument,
        help="Override defaults with the complete repeated NAME=PATH predictor set.",
    )
    parser.add_argument("--negate", nargs="*", default=sorted(DEFAULT_NEGATED))
    parser.add_argument(
        "--include-pdbflex",
        action="store_true",
        help="Add the UdonPred PDBflex model to the default predictor set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = dict(args.predictor) if args.predictor else dict(DEFAULT_PREDICTORS)
    if args.include_pdbflex and not args.predictor:
        paths["pdbflex"] = Path("results/human_proteome/UdonPred/pdbflex")
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing predictor paths: " + ", ".join(missing))
    if "trizod" not in paths:
        raise ValueError("A trizod predictor is required as the sequence/length QC reference")

    pairwise = pd.read_csv(args.pairwise_csv)
    cluster_rows = []
    for column, label, filename in (
        ("residue_spearman", "pooled residue level", "predictor_clusters_residue.png"),
        ("protein_spearman", "protein mean-score level", "predictor_clusters_protein.png"),
    ):
        matrix = pairwise_matrix(pairwise, column)
        order = plot_cluster(matrix, label, args.output_dir / filename)
        cluster_rows.extend(
            {"metric": column, "cluster_position": index + 1, "predictor": name,
             "group": predictor_group(name)}
            for index, name in enumerate(order)
        )
    pd.DataFrame(cluster_rows).to_csv(args.output_dir / "predictor_cluster_order.csv", index=False)
    pd.DataFrame(group_agreement_rows(pairwise)).to_csv(
        args.output_dir / "predictor_group_agreement.csv", index=False
    )

    means, lengths, qc_rows = load_protein_summaries(
        paths, {name.lower() for name in args.negate}
    )
    pd.DataFrame(qc_rows).to_csv(args.output_dir / "predictor_qc.csv", index=False)
    effects, bin_summary, binned_agreement, protein_table = length_effect_tables(means, lengths)
    binned_scores = binned_predictor_scores(means, lengths)
    effects.to_csv(args.output_dir / "length_effects_by_predictor.csv", index=False)
    bin_summary.to_csv(args.output_dir / "length_bin_summary.csv", index=False)
    binned_agreement.to_csv(args.output_dir / "length_binned_pairwise_agreement.csv", index=False)
    binned_scores.to_csv(args.output_dir / "length_binned_predictor_scores.csv", index=False)
    protein_table.to_csv(args.output_dir / "protein_length_disagreement.csv")
    plot_length_effects(effects, bin_summary, protein_table, args.output_dir / "protein_length_effects.png")
    plot_binned_predictor_scores(
        binned_scores, args.output_dir / "protein_length_vs_mean_scores.png"
    )

    annotation = pd.read_csv(args.annotation_csv)
    annotation_rows, annotation_summary = annotation_comparison(pairwise, annotation)
    annotation_rows.to_csv(args.output_dir / "annotation_vs_predictor_agreement.csv", index=False)
    write_summary(annotation_summary, args.output_dir / "annotation_vs_predictor_summary.csv")
    plot_annotation_comparison(
        annotation_rows, args.output_dir / "annotation_vs_predictor_agreement.png"
    )
    print(f"Wrote global predictor behavior analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
