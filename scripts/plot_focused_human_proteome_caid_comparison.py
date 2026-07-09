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
from matplotlib.patches import Patch, Rectangle

from compare_predictors import compute_pairwise_agreements, load_predictions, maybe_negate_scores


UDONPRED_MODELS = ["trizod", "chezod", "softdis", "atlas", "plddt", "disprot"]
EXTERNAL_MODELS = ["PUNCH2_light", "DisoFLAG", "DisorderUnetLM", "DisPredict3"]
CAID_TEXT_WHITE_MODELS = {"DisoFLAG", "DisorderUnetLM", "DisPredict3"}
PREDICTOR_ORDER = UDONPRED_MODELS + EXTERNAL_MODELS
NEGATED_PREDICTORS = {"chezod", "plddt"}
EXTERNAL_PATHS = {
    "PUNCH2_light": Path("PUNCH2_light/disorder"),
    "DisoFLAG": Path("DisoFLAG/caid"),
    "DisorderUnetLM": Path("DisorderUnetLM/disorder"),
    "DisPredict3": Path("Dispredict3_native/caid"),
}
DISPLAY_LABELS = {
    "trizod": "TriZOD",
    "chezod": "CheZOD",
    "softdis": "SoftDis",
    "atlas": "ATLAS",
    "plddt": "plDDt",
    "disprot": "DisProt",
    "PUNCH2_light": "PUNCH2-Light",
    "DisPredict3": "DisPredict3.0",
}
TITLE_FONTSIZE = 30
SUPTITLE_FONTSIZE = 36
TICK_FONTSIZE = 24
COLORBAR_LABEL_FONTSIZE = 24
COLORBAR_TICK_FONTSIZE = 22
CELL_FONTSIZE = 28
CELL_VALUE_FONTSIZE_WITH_SE = 28
CELL_SE_FONTSIZE = 21
PRESENTATION_CELL_FONTSIZE = 21
CLUSTER_LEGEND_FONTSIZE = 24
UDON_GROUP_LABEL_FONTSIZE = 22
TOP10_COLOR_MIN = 20
TOP10_COLOR_MAX = 100
UDON_CLUSTER_COLOR = "#005f73"
PUNCH2_BRIDGE_COLOR = "#ee9b00"
CAID_BLOCK_COLOR = "#ae2012"
CLUSTER_HALO_COLOR = "white"
UDON_GROUP_LABEL = "UdonPred"


def display_labels(labels: pd.Index) -> list[str]:
    return [DISPLAY_LABELS.get(str(label), str(label)) for label in labels]


BATLOW_CMAP = cmcrameri_cm.batlow


def cell_text_color(row: str, col: str, value: float, threshold: float) -> str:
    if row in CAID_TEXT_WHITE_MODELS or col in CAID_TEXT_WHITE_MODELS:
        return "white"
    return "white" if value < threshold else "black"


def correlation_se(rho: float, n: float) -> float:
    """Approximate Spearman SE with a Fisher-z delta-method estimate."""
    if not math.isfinite(rho) or not math.isfinite(n) or n <= 3:
        return math.nan
    rho = min(max(rho, -0.999999), 0.999999)
    return float((1.0 - rho * rho) / math.sqrt(n - 3.0))


def binomial_percent_se(percent: float, n: float, population_n: float | None = None) -> float:
    if not math.isfinite(percent) or not math.isfinite(n) or n <= 0:
        return math.nan
    p = min(max(percent / 100.0, 0.0), 1.0)
    se = 100.0 * math.sqrt(p * (1.0 - p) / n)
    if population_n is not None and math.isfinite(population_n) and population_n > 1 and n < population_n:
        se *= math.sqrt((population_n - n) / (population_n - 1.0))
    return float(se)


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
        population_n = float(row.common_proteins)
        se = binomial_percent_se(value, n, population_n)
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
                annot.loc[row, col] = f"{value:{value_fmt}}\n\u00b1 {error:{error_fmt}}"
            else:
                annot.loc[row, col] = f"{value:{value_fmt}}"
    return annot


def draw_cell_label(
    ax: plt.Axes,
    x: int,
    y: int,
    label: str,
    color: str,
    *,
    with_se: bool,
) -> None:
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


def add_udonpred_axis_brackets(ax: plt.Axes) -> None:
    group_size = len(UDONPRED_MODELS)
    x_start = -0.5
    x_end = group_size - 0.5
    y_start = -0.5
    y_end = group_size - 0.5
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()

    bracket_kwargs = {
        "color": UDON_CLUSTER_COLOR,
        "linewidth": 3.2,
        "solid_capstyle": "butt",
        "clip_on": False,
        "zorder": 10,
    }
    ax.plot(
        [x_start, x_start, x_end, x_end],
        [-0.55, -0.78, -0.78, -0.55],
        **bracket_kwargs,
    )
    ax.plot(
        [-1.50, -1.85, -1.85, -1.50],
        [y_start, y_start, y_end, y_end],
        **bracket_kwargs,
    )
    ax.text(
        -2.18,
        (y_start + y_end) / 2,
        UDON_GROUP_LABEL,
        ha="center",
        va="center",
        rotation=90,
        fontsize=UDON_GROUP_LABEL_FONTSIZE,
        weight="bold",
        color=UDON_CLUSTER_COLOR,
        clip_on=False,
    )

    ax.set_xlim(old_xlim)
    ax.set_ylim(old_ylim)


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
    figsize: tuple[float, float] = (19.5, 16.0),
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
            value = float(matrix.loc[row, col])
            color = cell_text_color(str(row), str(col), value, threshold)
            draw_cell_label(ax, j, i, label, color, with_se=with_se)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, weight="bold", pad=72)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    add_udonpred_axis_brackets(ax)
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

    fig, axes = plt.subplots(1, 2, figsize=(38.0, 17.0))
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
                value = float(matrix.loc[row, col])
                color = cell_text_color(str(row), str(col), value, threshold)
                draw_cell_label(ax, j, i, label, color, with_se=with_se_labels)

        ax.set_title(title, fontsize=TITLE_FONTSIZE, weight="bold", pad=62)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
        add_udonpred_axis_brackets(ax)

    fig.suptitle(
        "Human Proteome: UdonPred vs PUNCH2-Light, DisoFLAG, DisorderUnetLM, DisPredict3.0",
        fontsize=SUPTITLE_FONTSIZE,
        weight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_presentation_cluster_heatmap(protein_values: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(27.0, 17.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 0.38], wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[0, 2])
    legend_ax.axis("off")

    vmin = float(np.nanmin(protein_values.values))
    image = ax.imshow(protein_values.values.astype(float), cmap=BATLOW_CMAP, vmin=vmin, vmax=1.0)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Protein-level Spearman rho", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    ax.set_xticks(np.arange(len(protein_values.columns)))
    ax.set_yticks(np.arange(len(protein_values.index)))
    ax.set_xticklabels(display_labels(protein_values.columns))
    ax.set_yticklabels(display_labels(protein_values.index))
    ax.set_xticks(np.arange(-0.5, len(protein_values.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(protein_values.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = vmin + 0.55 * (1.0 - vmin)
    for i, row in enumerate(protein_values.index):
        for j, col in enumerate(protein_values.columns):
            value = float(protein_values.loc[row, col])
            color = cell_text_color(str(row), str(col), value, threshold)
            draw_cell_label(ax, j, i, f"{value:.2f}", color, with_se=False)

    add_presentation_overlays(ax, linewidth=4.5)

    legend_handles = [
        Patch(facecolor="none", edgecolor=UDON_CLUSTER_COLOR, linewidth=4, label="UdonPred cluster"),
        Patch(
            facecolor=PUNCH2_BRIDGE_COLOR,
            edgecolor=PUNCH2_BRIDGE_COLOR,
            alpha=0.22,
            linestyle="dashed",
            label="PUNCH2-Light bridge",
        ),
        Patch(
            facecolor="none",
            edgecolor=CAID_BLOCK_COLOR,
            linewidth=4,
            linestyle="dashdot",
            label="CAID-style block",
        ),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=CLUSTER_LEGEND_FONTSIZE,
        borderaxespad=0,
        handlelength=1.6,
        labelspacing=1.3,
    )

    ax.set_title(
        "Protein-Level Agreement: Three Main Patterns",
        fontsize=TITLE_FONTSIZE + 2,
        weight="bold",
        pad=64,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
    ax.set_xlim(-0.5, len(protein_values.columns) - 0.5)
    ax.set_ylim(len(protein_values.index) - 0.5, -0.5)
    add_udonpred_axis_brackets(ax)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.90)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def add_haloed_rectangle(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    edgecolor: str,
    linewidth: float,
    linestyle: str = "solid",
) -> None:
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            fill=False,
            edgecolor=CLUSTER_HALO_COLOR,
            linewidth=linewidth + 3.0,
            linestyle=linestyle,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=6,
        )
    )


def add_presentation_overlays(ax: plt.Axes, *, linewidth: float = 4.0) -> None:
    ax.axhspan(5.5, 6.5, color=PUNCH2_BRIDGE_COLOR, alpha=0.14, zorder=2)
    ax.axvspan(5.5, 6.5, color=PUNCH2_BRIDGE_COLOR, alpha=0.14, zorder=2)
    add_haloed_rectangle(
        ax,
        (-0.5, -0.5),
        6,
        6,
        edgecolor=UDON_CLUSTER_COLOR,
        linewidth=linewidth,
        linestyle="solid",
    )
    add_haloed_rectangle(
        ax,
        (-0.5, 5.5),
        10,
        1,
        edgecolor=PUNCH2_BRIDGE_COLOR,
        linewidth=linewidth - 0.5,
        linestyle="dashed",
    )
    add_haloed_rectangle(
        ax,
        (5.5, -0.5),
        1,
        10,
        edgecolor=PUNCH2_BRIDGE_COLOR,
        linewidth=linewidth - 0.5,
        linestyle="dashed",
    )
    add_haloed_rectangle(
        ax,
        (6.5, 6.5),
        3,
        3,
        edgecolor=CAID_BLOCK_COLOR,
        linewidth=linewidth,
        linestyle="dashdot",
    )


def draw_presentation_dual_cluster_heatmaps(
    residue_values: pd.DataFrame,
    protein_values: pd.DataFrame,
    path: Path,
) -> None:
    matrices = [
        ("Residue-level Spearman", residue_values),
        ("Protein-level Spearman", protein_values),
    ]
    vmin = min(float(np.nanmin(residue_values.values)), float(np.nanmin(protein_values.values)))
    fig = plt.figure(figsize=(42.0, 18.5))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.045, 0.35], wspace=0.18)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    legend_ax = fig.add_subplot(grid[0, 3])
    legend_ax.axis("off")

    image = None
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix.values.astype(float), cmap=BATLOW_CMAP, vmin=vmin, vmax=1.0)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(display_labels(matrix.columns))
        ax.set_yticklabels(display_labels(matrix.index))
        ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        threshold = vmin + 0.55 * (1.0 - vmin)
        for i, row in enumerate(matrix.index):
            for j, col in enumerate(matrix.columns):
                value = float(matrix.loc[row, col])
                color = cell_text_color(str(row), str(col), value, threshold)
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=PRESENTATION_CELL_FONTSIZE,
                    color=color,
                )

        add_presentation_overlays(ax, linewidth=4.0)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, weight="bold", pad=54)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(-0.5, len(matrix.columns) - 0.5)
        ax.set_ylim(len(matrix.index) - 0.5, -0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=TICK_FONTSIZE)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONTSIZE)
        add_udonpred_axis_brackets(ax)

    assert image is not None
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Spearman rho", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    legend_handles = [
        Patch(facecolor="none", edgecolor=UDON_CLUSTER_COLOR, linewidth=4, label="UdonPred cluster"),
        Patch(
            facecolor=PUNCH2_BRIDGE_COLOR,
            edgecolor=PUNCH2_BRIDGE_COLOR,
            alpha=0.22,
            linestyle="dashed",
            label="PUNCH2-Light bridge",
        ),
        Patch(
            facecolor="none",
            edgecolor=CAID_BLOCK_COLOR,
            linewidth=4,
            linestyle="dashdot",
            label="CAID-style block",
        ),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=CLUSTER_LEGEND_FONTSIZE,
        borderaxespad=0,
        handlelength=1.6,
        labelspacing=1.3,
    )
    fig.suptitle(
        "Agreement Structure at Residue and Protein Level",
        fontsize=SUPTITLE_FONTSIZE,
        weight="bold",
        y=0.965,
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.18, top=0.88)
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
            float(row["common_proteins"]),
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
        vmin=TOP10_COLOR_MIN,
        vmax=TOP10_COLOR_MAX,
        fmt=".0f",
        annot=True,
    )
    draw_heatmap(
        top10_values,
        output_dir / "top10_protein_overlap_with_se.png",
        title="Human Proteome: top-10% most disorder-prone protein overlap",
        cbar_label="Shared top-10% proteins (%)",
        vmin=TOP10_COLOR_MIN,
        vmax=TOP10_COLOR_MAX,
        annot=annotation(top10_values, top10_errors, ".0f", ".1f"),
    )
    draw_presentation_cluster_heatmap(
        protein_values,
        output_dir / "presentation_cluster_annotated_protein_spearman.png",
    )
    draw_presentation_dual_cluster_heatmaps(
        residue_values,
        protein_values,
        output_dir / "presentation_cluster_annotated_spearman_heatmaps.png",
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
