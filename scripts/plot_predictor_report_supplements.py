#!/usr/bin/env python3
"""Create supplementary figures for the predictor comparison report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_predictors import load_predictions, maybe_negate_scores  # noqa: E402


OUTPUT_DIR = Path("docs/images/predictor_comparison")
RESULT_DIR = Path("results/compare_predictors_with_all_predictors_wo_pdbflex")

PREDICTOR_PATHS = {
    "trizod": Path("results/human_proteome/UdonPred/trizod"),
    "chezod": Path("results/human_proteome/UdonPred/chezod"),
    "softdis": Path("results/human_proteome/UdonPred/softdis"),
    "atlas": Path("results/human_proteome/UdonPred/atlas"),
    "plddt": Path("results/human_proteome/UdonPred/plddt"),
    "disprot": Path("results/human_proteome/UdonPred/disprot"),
    "SETH": Path("results/human_proteome/SETH/seth_human_proteome.caid"),
    "ADOPT": Path("results/human_proteome/ADOPT"),
    "metapredict": Path("results/human_proteome/metapredict/metapredict_human_proteome.caid"),
    "PUNCH2_light": Path("results/human_proteome/PUNCH2_light/disorder"),
    "DisPredict3": Path("results/human_proteome/Dispredict3_native/caid"),
    "DisoFLAG": Path("results/human_proteome/DisoFLAG/caid"),
    "DisorderUnetLM": Path("results/human_proteome/DisorderUnetLM/disorder"),
    "IUPred3": Path("results/human_proteome/IUPred3"),
}

NEGATED = {"chezod", "plddt", "adopt"}

CONSENSUS = [
    "trizod",
    "chezod",
    "softdis",
    "atlas",
    "plddt",
    "disprot",
    "SETH",
    "ADOPT",
    "metapredict",
    "PUNCH2_light",
]
CAID_SUBGROUP = ["DisPredict3", "DisoFLAG", "DisorderUnetLM"]
PROFILE_PREDICTORS = [
    "trizod",
    "plddt",
    "disprot",
    "SETH",
    "PUNCH2_light",
    "DisPredict3",
    "DisoFLAG",
    "DisorderUnetLM",
    "IUPred3",
]
CASE_PROTEINS = {
    "Q3KQU3": "Consensus-high / DisPredict3-low",
    "P29762": "DisPredict3-high / consensus-low",
}


def protein_spearman(pairwise: pd.DataFrame, left: str, right: str) -> float:
    mask = (
        ((pairwise["predictor_a"] == left) & (pairwise["predictor_b"] == right))
        | ((pairwise["predictor_a"] == right) & (pairwise["predictor_b"] == left))
    )
    return float(pairwise.loc[mask, "protein_spearman"].iloc[0])


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    box = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="top", fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.03, body, ha="center", va="center", fontsize=10.5, linespacing=1.35)


def plot_takehome_map() -> None:
    group = pd.read_csv(RESULT_DIR / "global_behavior/predictor_group_agreement.csv")
    pairwise = pd.read_csv(RESULT_DIR / "pairwise_agreement.csv")

    def group_value(name: str) -> float:
        rows = group[(group["metric"] == "protein_spearman") & (group["group_pair"] == name)]
        return float(rows["mean_spearman"].iloc[0])

    udon_internal = group_value("UdonPred vs UdonPred")
    udon_classical = group_value("UdonPred vs classical external")
    caid_internal = group_value("CAID-style external vs CAID-style external")
    caid_udon = group_value("CAID-style external vs UdonPred")
    d3_disoflag = protein_spearman(pairwise, "DisPredict3", "DisoFLAG")
    d3_unet = protein_spearman(pairwise, "DisPredict3", "DisorderUnetLM")
    d3_iupred = protein_spearman(pairwise, "DisPredict3", "IUPred3")

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    draw_box(
        ax,
        (0.4, 3.15),
        (4.1, 2.1),
        "Broad consensus / NMR-UdonPred core",
        "TriZOD, CheZOD, SoftDis, ATLAS,\npLDDT, DisProt, SETH, ADOPT,\nmetapredict, PUNCH2-light\n\nhigh shared disorder burden signal",
        "#d8e8f6",
    )
    draw_box(
        ax,
        (5.55, 3.15),
        (3.9, 2.1),
        "CAID-style subgroup",
        "DisPredict3, DisoFLAG,\nDisorderUnetLM\n\nbinary/segmentation-like behavior;\nconservative high-score regions",
        "#d8efe7",
    )
    draw_box(
        ax,
        (3.1, 0.55),
        (3.8, 1.45),
        "IUPred3",
        "energy-based folding tendency\ncomplementary rather than central",
        "#f6e2cc",
    )

    arrow = dict(arrowstyle="<->", color="#333333", linewidth=1.6, shrinkA=8, shrinkB=8)
    ax.annotate("", xy=(5.55, 4.2), xytext=(4.5, 4.2), arrowprops=arrow)
    ax.text(
        5.02,
        4.55,
        f"between groups\nmean protein rho = {caid_udon:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.annotate("", xy=(4.2, 2.0), xytext=(3.7, 3.15), arrowprops=arrow)
    ax.text(
        3.1,
        2.62,
        f"UdonPred vs classical\nmean protein rho = {udon_classical:.2f}",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.annotate("", xy=(6.15, 2.0), xytext=(7.2, 3.15), arrowprops=arrow)
    ax.text(
        7.32,
        2.55,
        f"DisPredict3 vs IUPred3\nprotein rho = {d3_iupred:.2f}",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.text(2.45, 5.45, f"within UdonPred mean protein rho = {udon_internal:.2f}", ha="center", fontsize=11)
    ax.text(7.5, 5.45, f"within CAID-style mean protein rho = {caid_internal:.2f}", ha="center", fontsize=11)
    ax.text(
        7.5,
        3.0,
        f"DisPredict3 with DisoFLAG/UnetLM:\nprotein rho = {d3_disoflag:.2f} / {d3_unet:.2f}",
        ha="center",
        va="top",
        fontsize=10,
    )

    ax.text(
        5.0,
        0.1,
        "Interpretation: agreement is structured. The panel contains a shared disorder signal, a CAID-style branch, and an energy-based complementary view.",
        ha="center",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictor_group_takehome.png", dpi=220)
    plt.close(fig)


def percentile_transform(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    finite_reference = np.sort(reference[np.isfinite(reference)])
    if len(finite_reference) == 0:
        return np.full_like(values, np.nan, dtype=float)
    return np.searchsorted(finite_reference, values, side="right") / len(finite_reference)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def load_profile_predictors() -> dict[str, dict]:
    records = {}
    for name in PROFILE_PREDICTORS:
        loaded = load_predictions(PREDICTOR_PATHS[name])
        records[name] = maybe_negate_scores(loaded, name, NEGATED)
    return records


def plot_case_profiles() -> None:
    records = load_profile_predictors()

    residue_references = {
        name: np.concatenate([record.scores[np.isfinite(record.scores)] for record in predictor.values()])
        for name, predictor in records.items()
    }
    protein_means = {}
    protein_mean_percentiles = {}
    for name, predictor in records.items():
        ids = list(predictor)
        means = np.asarray([np.nanmean(predictor[protein_id].scores) for protein_id in ids], dtype=float)
        order = np.argsort(means)
        ranks = np.empty(len(means), dtype=float)
        ranks[order] = np.arange(len(means)) / max(len(means) - 1, 1)
        protein_means[name] = dict(zip(ids, means))
        protein_mean_percentiles[name] = dict(zip(ids, ranks))

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.8), gridspec_kw={"height_ratios": [1.45, 1.0]})
    colors = {
        "Consensus mean": "#376795",
        "CAID subgroup mean": "#2a9d8f",
        "DisPredict3": "#d9534f",
        "IUPred3": "#9c6b30",
    }

    for column, (protein_id, label) in enumerate(CASE_PROTEINS.items()):
        profile_ax = axes[0, column]
        bar_ax = axes[1, column]
        lengths = [len(records[name][protein_id].scores) for name in PROFILE_PREDICTORS if protein_id in records[name]]
        length = min(lengths)
        x = np.arange(1, length + 1)

        percentile_profiles = {}
        for name in PROFILE_PREDICTORS:
            scores = records[name][protein_id].scores[:length]
            percentile_profiles[name] = percentile_transform(scores, residue_references[name])

        consensus_profile = np.nanmean([percentile_profiles[name] for name in ["trizod", "plddt", "disprot", "SETH", "PUNCH2_light"]], axis=0)
        caid_profile = np.nanmean([percentile_profiles[name] for name in CAID_SUBGROUP], axis=0)
        plot_profiles = {
            "Consensus mean": consensus_profile,
            "CAID subgroup mean": caid_profile,
            "DisPredict3": percentile_profiles["DisPredict3"],
            "IUPred3": percentile_profiles["IUPred3"],
        }
        window = 31 if length > 250 else 11
        for name, values in plot_profiles.items():
            profile_ax.plot(x, moving_average(values, window), label=name, color=colors[name], linewidth=2.0)
        profile_ax.set_title(f"{protein_id}: {label}", fontsize=13, weight="bold")
        profile_ax.set_xlabel("Residue position")
        profile_ax.set_ylabel("Residue score percentile")
        profile_ax.set_ylim(0, 1.02)
        profile_ax.grid(alpha=0.25)
        if column == 0:
            profile_ax.legend(loc="lower right", fontsize=9)

        loaded_consensus = [name for name in CONSENSUS if name in protein_mean_percentiles]
        group_percentiles = {
            "Consensus\nmean": float(np.mean([protein_mean_percentiles[name][protein_id] for name in loaded_consensus if protein_id in protein_mean_percentiles[name]])),
            "CAID\nmean": float(np.mean([protein_mean_percentiles[name][protein_id] for name in CAID_SUBGROUP if protein_id in protein_mean_percentiles[name]])),
            "DisPredict3": float(protein_mean_percentiles["DisPredict3"][protein_id]),
            "IUPred3": float(protein_mean_percentiles["IUPred3"][protein_id]),
        }
        bar_colors = [colors["Consensus mean"], colors["CAID subgroup mean"], colors["DisPredict3"], colors["IUPred3"]]
        bar_ax.bar(list(group_percentiles), list(group_percentiles.values()), color=bar_colors)
        bar_ax.set_ylim(0, 1.02)
        bar_ax.set_ylabel("Protein mean-score percentile")
        bar_ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(group_percentiles.values()):
            bar_ax.text(index, value + 0.03, f"{value:.2f}", ha="center", fontsize=9)

    fig.suptitle(
        "Concrete protein examples: group-level ranking disagreement and local score profiles",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "example_protein_profiles.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_takehome_map()
    plot_case_profiles()


if __name__ == "__main__":
    main()
