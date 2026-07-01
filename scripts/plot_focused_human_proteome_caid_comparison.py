#!/usr/bin/env python3
"""Focused Human Proteome comparison of UdonPred and selected CAID predictors.

The script renders residue/protein Spearman heatmaps and top-10% protein-overlap
heatmaps in two annotation styles:

* value only
* value plus standard error in each cell

It loads the Human Proteome CAID outputs directly from results/human_proteome.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

from cmcrameri import cm as cmcrameri_cm
import matplotlib.pyplot as plt

from compare_predictors import compute_pairwise_agreements, load_predictions, maybe_negate_scores


UDONPRED_MODELS = ["trizod", "chezod", "softdis", "atlas", "plddt", "disprot"]
EXTERNAL_MODELS = ["PUNCH2_light", "DisoFLAG", "DisorderUnetLM", "DisPredict3"]
PREDICTOR_ORDER = UDONPRED_MODELS + EXTERNAL_MODELS
NEGATED_PREDICTORS = {"chezod", "plddt"}
EXTERNAL_PATHS = {
    "PUNCH2_light": Path("PUNCH2_light/disorder"),
    "DisoFLAG": Path("DisoFLAG/caid"),
    "DisorderUnetLM": Path("DisorderUnetLM/disorder"),
    "DisPredict3": Path("Dispredict3_native/caid"),
}
DISPLAY_LABELS = {
    "PUNCH2_light": "PUNCH2-Light",
    "DisPredict3": "DisPredict3.0",
}
TITLE_FONTSIZE = 23
SUPTITLE_FONTSIZE = 28
TICK_FONTSIZE = 17
COLORBAR_LABEL_FONTSIZE = 19
COLORBAR_TICK_FONTSIZE = 16
CELL_FONTSIZE = 25
CELL_VALUE_FONTSIZE_WITH_SE = 25
CELL_SE_FONTSIZE = 16


def display_labels(labels: pd.Index) -> list[str]:
    return [DISPLAY_LABELS.get(str(label), str(label)) for label in labels]


BATLOW_CMAP = cmcrameri_cm.batlow


def correlation_se(rho: float, n: float) -> float:
    """Approximate Spearman SE with a Fisher-z delta-method estimate."""
    if not math.isfinite(rho) or not math.isfinite(n) or n <= 3:
        return math.nan
    rho = min(max(rho, -0.999999), 0.999999)
    return float((1.0 - rho * rho) / math.sqrt(n - 3.0))


def binomial_percent_se(percent: float, n: float) -> float:
    if not math.isfinite(percent) or not math.isfinite(n) or n <= 0:
        return math.nan
    p = min(max(percent / 100.0, 0.0), 1.0)
    return float(100.0 * math.sqrt(p * (1.0 - p) / n))


def pairwise_matrix(pairwise: pd.DataFrame, value_column: str, n_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.DataFrame(np.nan, index=PREDICTOR_ORDER, columns=PREDICTOR_ORDER)
    errors = pd.DataFrame(np.nan, index=PREDICTOR_ORDER, columns=PREDICTOR_ORDER)

    for predictor in PREDICTOR_ORDER:
        values.loc[predictor, predictor] = 1.0
        errors.loc[predictor, predictor] = 0.0

    for row in pairwise.itertuples(index=False):
        left = row.predictor_a
        right = row.predictor_b
        if left not in values.index or right not in values.columns:
            continue
        value = float(getattr(row, value_column))
        n = float(getattr(row, n_column))
        se = correlation_se(value, n)
        values.loc[left, right] = value
        values.loc[right, left] = value
        errors.loc[left, right] = se
        errors.loc[right, left] = se

    return values, errors


def top10_matrices(top10_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.DataFrame(np.nan, index=PREDICTOR_ORDER, columns=PREDICTOR_ORDER)
    errors = pd.DataFrame(np.nan, index=PREDICTOR_ORDER, columns=PREDICTOR_ORDER)

    for predictor in PREDICTOR_ORDER:
        values.loc[predictor, predictor] = 100.0
        errors.loc[predictor, predictor] = 0.0

    for row in top10_pairs.itertuples(index=False):
        left = row.predictor_a
        right = row.predictor_b
        if left not in values.index or right not in values.columns:
            continue
        value = float(row.top_10_percent_overlap)
        n = float(row.proteins_per_top_10_percent)
        se = binomial_percent_se(value, n)
        values.loc[left, right] = value
        values.loc[right, left] = value
        errors.loc[left, right] = se
        errors.loc[right, left] = se

    return values, errors


def load_focused_predictions(human_proteome_root: Path) -> dict[str, dict]:
    predictions = {}
    udonpred_root = human_proteome_root / "UdonPred"
    for name in UDONPRED_MODELS:
        records = load_predictions(udonpred_root / name)
        predictions[name] = maybe_negate_scores(records, name, NEGATED_PREDICTORS)

    for name, relative_path in EXTERNAL_PATHS.items():
        records = load_predictions(human_proteome_root / relative_path)
        predictions[name] = maybe_negate_scores(records, name, NEGATED_PREDICTORS)

    return predictions


def compute_top10_pairs(predictions: dict[str, dict]) -> pd.DataFrame:
    protein_mean_series = {}
    for predictor, predictor_records in predictions.items():
        means = {}
        for protein_id, record in predictor_records.items():
            finite_scores = record.scores[np.isfinite(record.scores)]
            if len(finite_scores):
                means[protein_id] = float(np.mean(finite_scores))
        protein_mean_series[predictor] = pd.Series(means, dtype=float)

    all_protein_means = pd.DataFrame(protein_mean_series)
    rows = []
    for index, predictor_a in enumerate(PREDICTOR_ORDER):
        for predictor_b in PREDICTOR_ORDER[index + 1 :]:
            pair = all_protein_means[[predictor_a, predictor_b]].dropna()
            top_n = int(np.ceil(0.10 * len(pair)))
            top_a = set(pair[predictor_a].nlargest(top_n).index)
            top_b = set(pair[predictor_b].nlargest(top_n).index)
            shared = len(top_a & top_b)
            rows.append(
                {
                    "predictor_a": predictor_a,
                    "predictor_b": predictor_b,
                    "common_proteins": len(pair),
                    "proteins_per_top_10_percent": top_n,
                    "shared_top_10_percent_proteins": shared,
                    "top_10_percent_overlap": 100.0 * shared / top_n if top_n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def annotation(values: pd.DataFrame, errors: pd.DataFrame, value_fmt: str, error_fmt: str) -> pd.DataFrame:
    annot = pd.DataFrame("", index=values.index, columns=values.columns)
    for row in values.index:
        for col in values.columns:
            value = values.loc[row, col]
            error = errors.loc[row, col]
            if not math.isfinite(value):
                continue
            if math.isfinite(error):
                annot.loc[row, col] = f"{value:{value_fmt}}\nSE {error:{error_fmt}}"
            else:
                annot.loc[row, col] = f"{value:{value_fmt}}"
    return annot


def draw_cell_label(ax: plt.Axes, x: int, y: int, label: str, color: str, *, with_se: bool) -> None:
    if with_se and "\n" in label:
        value_label, se_label = label.split("\n", 1)
        ax.text(
            x,
            y - 0.12,
            value_label,
            ha="center",
            va="center",
            fontsize=CELL_VALUE_FONTSIZE_WITH_SE,
            color=color,
        )
        ax.text(
            x,
            y + 0.18,
            se_label,
            ha="center",
            va="center",
            fontsize=CELL_SE_FONTSIZE,
            color=color,
        )
        return

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=CELL_FONTSIZE,
        color=color,
    )


def draw_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    fmt: str = ".2f",
    annot: pd.DataFrame | bool = True,
    figsize: tuple[float, float] = (17.0, 14.0),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix.values.astype(float), cmap=BATLOW_CMAP, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(display_labels(matrix.columns))
    ax.set_yticklabels(display_labels(matrix.index))
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    if isinstance(annot, pd.DataFrame):
        labels = annot
        with_se = True
    elif annot:
        labels = matrix.map(lambda value: f"{value:{fmt}}" if math.isfinite(value) else "")
        with_se = False
    else:
        labels = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
        with_se = False

    threshold = vmin + 0.55 * (vmax - vmin)
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            label = labels.loc[row, col]
            if not label:
                continue
            color = "white" if float(matrix.loc[row, col]) < threshold else "black"
            draw_cell_label(ax, j, i, label, color, with_se=with_se)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_spearman_pair(
    residue_values: pd.DataFrame,
    residue_errors: pd.DataFrame,
    protein_values: pd.DataFrame,
    protein_errors: pd.DataFrame,
    path: Path,
    *,
    with_se: bool,
) -> None:
    annot_residue: pd.DataFrame | bool = True
    annot_protein: pd.DataFrame | bool = True
    fmt = ".2f"
    if with_se:
        annot_residue = annotation(residue_values, residue_errors, ".2f", ".4f")
        annot_protein = annotation(protein_values, protein_errors, ".2f", ".4f")
        fmt = ""

    vmin = min(float(np.nanmin(residue_values.values)), float(np.nanmin(protein_values.values)))

    fig, axes = plt.subplots(1, 2, figsize=(34.0, 15.0))
    for ax, matrix, annot, title in [
        (axes[0], residue_values, annot_residue, "Residue-level Spearman"),
        (axes[1], protein_values, annot_protein, "Protein-level Spearman"),
    ]:
        image = ax.imshow(matrix.values.astype(float), cmap=BATLOW_CMAP, vmin=vmin, vmax=1.0)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spearman rho", fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(display_labels(matrix.columns))
        ax.set_yticklabels(display_labels(matrix.index))
        ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)

        if isinstance(annot, pd.DataFrame):
            labels = annot
            with_se_labels = True
        else:
            labels = matrix.map(lambda value: f"{value:{fmt}}" if math.isfinite(value) else "")
            with_se_labels = False
        threshold = vmin + 0.55 * (1.0 - vmin)
        for i, row in enumerate(matrix.index):
            for j, col in enumerate(matrix.columns):
                label = labels.loc[row, col]
                if not label:
                    continue
                color = "white" if float(matrix.loc[row, col]) < threshold else "black"
                draw_cell_label(ax, j, i, label, color, with_se=with_se_labels)

        ax.set_title(title, fontsize=TITLE_FONTSIZE, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)

    fig.suptitle(
        "Human Proteome: UdonPred vs PUNCH2-Light, DisoFLAG, DisorderUnetLM, DisPredict3.0",
        fontsize=SUPTITLE_FONTSIZE,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output_dir: Path,
    residue_values: pd.DataFrame,
    residue_errors: pd.DataFrame,
    protein_values: pd.DataFrame,
    protein_errors: pd.DataFrame,
    top10_values: pd.DataFrame,
    top10_errors: pd.DataFrame,
    pairwise: pd.DataFrame,
    top10_pairs: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    residue_values.to_csv(output_dir / "residue_spearman_matrix.csv")
    residue_errors.to_csv(output_dir / "residue_spearman_se_matrix.csv")
    protein_values.to_csv(output_dir / "protein_spearman_matrix.csv")
    protein_errors.to_csv(output_dir / "protein_spearman_se_matrix.csv")
    top10_values.to_csv(output_dir / "top10_protein_overlap_matrix.csv")
    top10_errors.to_csv(output_dir / "top10_protein_overlap_se_matrix.csv")

    focused_pairwise = pairwise[
        pairwise["predictor_a"].isin(PREDICTOR_ORDER)
        & pairwise["predictor_b"].isin(PREDICTOR_ORDER)
    ].copy()
    focused_pairwise["residue_spearman_se"] = focused_pairwise.apply(
        lambda row: correlation_se(float(row["residue_spearman"]), float(row["matched_residues"])),
        axis=1,
    )
    focused_pairwise["protein_spearman_se"] = focused_pairwise.apply(
        lambda row: correlation_se(float(row["protein_spearman"]), float(row["common_proteins"])),
        axis=1,
    )
    focused_pairwise.to_csv(output_dir / "focused_pairwise_spearman_with_se.csv", index=False)

    focused_top10 = top10_pairs[
        top10_pairs["predictor_a"].isin(PREDICTOR_ORDER)
        & top10_pairs["predictor_b"].isin(PREDICTOR_ORDER)
    ].copy()
    focused_top10["top_10_percent_overlap_se"] = focused_top10.apply(
        lambda row: binomial_percent_se(
            float(row["top_10_percent_overlap"]),
            float(row["proteins_per_top_10_percent"]),
        ),
        axis=1,
    )
    focused_top10.to_csv(output_dir / "focused_top10_protein_overlap_with_se.csv", index=False)

    draw_spearman_pair(
        residue_values,
        residue_errors,
        protein_values,
        protein_errors,
        output_dir / "spearman_heatmaps_without_se.png",
        with_se=False,
    )
    draw_spearman_pair(
        residue_values,
        residue_errors,
        protein_values,
        protein_errors,
        output_dir / "spearman_heatmaps_with_se.png",
        with_se=True,
    )
    draw_heatmap(
        top10_values,
        output_dir / "top10_protein_overlap_without_se.png",
        title="Human Proteome: top-10% most disorder-prone protein overlap",
        cbar_label="Shared top-10% proteins (%)",
        vmin=0,
        vmax=100,
        fmt=".0f",
        annot=True,
    )
    draw_heatmap(
        top10_values,
        output_dir / "top10_protein_overlap_with_se.png",
        title="Human Proteome: top-10% most disorder-prone protein overlap",
        cbar_label="Shared top-10% proteins (%)",
        vmin=0,
        vmax=100,
        annot=annotation(top10_values, top10_errors, ".0f", ".1f"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-proteome-root",
        type=Path,
        default=Path("results/human_proteome"),
        help="Root directory containing Human Proteome predictor CAID outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_proteome_udonpred_caid_focus"),
        help="Directory for focused heatmaps and matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_focused_predictions(args.human_proteome_root)
    pairwise = pd.DataFrame(compute_pairwise_agreements(predictions))
    top10_pairs = compute_top10_pairs(predictions)

    residue_values, residue_errors = pairwise_matrix(pairwise, "residue_spearman", "matched_residues")
    protein_values, protein_errors = pairwise_matrix(pairwise, "protein_spearman", "common_proteins")
    top10_values, top10_errors = top10_matrices(top10_pairs)

    write_outputs(
        args.output_dir,
        residue_values,
        residue_errors,
        protein_values,
        protein_errors,
        top10_values,
        top10_errors,
        pairwise,
        top10_pairs,
    )

    print(f"Wrote focused Human Proteome comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
