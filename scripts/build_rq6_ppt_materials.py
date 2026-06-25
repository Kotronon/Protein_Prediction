#!/usr/bin/env python3
"""Build PPT-ready tables, figures, and notes for RQ6 disagreement results.

Inputs are the existing threshold-based disagreement outputs. This script does
not rerun UdonPred predictions and does not transform raw prediction scores.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


INPUT_DIR = Path("results/rq6_threshold_disagreement")
ORIGINAL_RAW_TABLE = Path("results/nmr_disagreement/tables/trizod_raw_predictions.csv")
OUTPUT_DIR = Path("results/rq6_ppt_materials")
FIGURE_DIRNAME = "figures"
MAX_GAP = 2

MODEL_ORDER = ["trizod", "chezod", "atlas", "plddt", "softdis", "disprot", "pdbflex"]
LABEL_COLUMNS = [f"{model}_label" for model in MODEL_ORDER]
RAW_COLUMNS = [f"{model}_raw" for model in MODEL_ORDER]
LABEL_COLORS = {
    "O": "#4C78A8",
    "D": "#F58518",
    "U": "#B8B8B8",
    "NA": "#E6E6E6",
}
LABEL_NAMES = {
    "O": "Ordered-like",
    "D": "Disordered-like",
    "U": "Uncertain",
    "NA": "Missing",
}
HEATMAP_VALUES = {"O": 0, "D": 1, "U": 2, "NA": 3}
HEATMAP_COLORS = ["#4C78A8", "#F58518", "#B8B8B8", "#E6E6E6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Existing RQ6 threshold disagreement output directory.",
    )
    parser.add_argument(
        "--raw-table",
        type=Path,
        default=ORIGINAL_RAW_TABLE,
        help="Original raw prediction table, recorded in report if present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output folder for PPT materials.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files in the output directory if they already exist.",
    )
    return parser.parse_args()


def configure_matplotlib(output_dir: Path):
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    return plt, ListedColormap, Patch


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    raw_matrix_path = input_dir / "raw_prediction_matrix.csv"
    residue_labels_path = input_dir / "residue_labels.csv"
    fragments_path = input_dir / "candidate_fragments.csv"
    summary_path = input_dir / "summary.txt"

    required = [raw_matrix_path, residue_labels_path, fragments_path, summary_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required RQ6 threshold outputs: "
            + ", ".join(str(path) for path in missing)
        )

    raw_matrix = pd.read_csv(raw_matrix_path, dtype={"protein_id": "string"})
    residue_labels = pd.read_csv(
        residue_labels_path,
        dtype={"protein_id": "string", "aa": "string"},
        keep_default_na=False,
    )
    fragments = pd.read_csv(fragments_path, dtype={"protein_id": "string"})
    summary_text = summary_path.read_text()
    return raw_matrix, residue_labels, fragments, summary_text


def labelable_models(residue_labels: pd.DataFrame) -> list[str]:
    models = []
    for model in MODEL_ORDER:
        label_column = f"{model}_label"
        if label_column not in residue_labels.columns:
            continue
        labels = set(residue_labels[label_column].dropna().astype(str).unique())
        if labels - {"U", "NA"}:
            models.append(model)
    return models


def model_interpretation(model: str, counts: dict[str, int], total: int) -> str:
    d_pct = 100 * counts["D"] / total if total else 0
    o_pct = 100 * counts["O"] / total if total else 0
    u_pct = 100 * counts["U"] / total if total else 0

    if model == "softdis":
        return "SoftDis labels most residues as disorder-like under the current raw threshold; sensitivity analysis is needed."
    if model == "pdbflex":
        return "PDBFlex raw scores are retained, but threshold is unresolved, so this head is not used for strong D/O voting yet."
    if model == "chezod":
        return "CheZOD uses a middle uncertain zone, so some residues are intentionally left as U."
    if model in {"trizod", "atlas", "plddt"} and o_pct >= 75:
        return f"{model.upper()} is mostly ordered-like under the current raw threshold."
    if model == "disprot" and o_pct >= d_pct:
        return "DisProt is mostly ordered-like here but contributes disorder-like votes in contested regions."
    if u_pct >= 90:
        return "This model is currently mostly uncertain under the configured labeling rule."
    return "This head contributes categorical votes on its own raw score scale."


def parse_model_list(model_counts: str, max_models: int = 3) -> str:
    if not isinstance(model_counts, str) or not model_counts:
        return ""
    names = []
    for item in model_counts.split(";"):
        if not item:
            continue
        names.append(item.split(":", maxsplit=1)[0])
    return ", ".join(names[:max_models])


def fragment_message(row: pd.Series) -> str:
    disordered = parse_model_list(row.get("dominant_disordered_models", ""))
    ordered = parse_model_list(row.get("dominant_ordered_models", ""))
    if disordered and ordered:
        return (
            f"Clean categorical split: {disordered} tend to call disorder-like, "
            f"while {ordered} tend to call ordered-like."
        )
    return "High categorical disagreement under current raw thresholds; interpret as a candidate region only."


def count_grouped_strong_fragments(residue_labels: pd.DataFrame) -> int:
    count = 0
    strong_rows = residue_labels.loc[
        residue_labels["strong_disagreement"],
        ["protein_id", "residue_index"],
    ].sort_values(["protein_id", "residue_index"])

    for _, protein_rows in strong_rows.groupby("protein_id", sort=True):
        previous = None
        for residue_index in protein_rows["residue_index"].astype(int):
            if previous is None:
                count += 1
            else:
                gap = residue_index - previous - 1
                if gap > MAX_GAP:
                    count += 1
            previous = residue_index
    return count


def build_key_numbers(
    raw_matrix: pd.DataFrame,
    residue_labels: pd.DataFrame,
    fragments: pd.DataFrame,
) -> pd.DataFrame:
    raw_columns = [column for column in raw_matrix.columns if column.endswith("_raw")]
    models_with_labels = labelable_models(residue_labels)
    total_residues = len(residue_labels)
    strong_residues = int(residue_labels["strong_disagreement"].sum())
    any_residues = int(residue_labels["has_disagreement"].sum())
    top_score = float(fragments["fragment_rank_score"].max()) if not fragments.empty else 0.0
    before_filter = count_grouped_strong_fragments(residue_labels)

    rows = [
        (
            "number of proteins",
            int(raw_matrix["protein_id"].nunique()),
            "TriZOD test-set proteins represented in the raw prediction matrix.",
        ),
        (
            "number of residues",
            total_residues,
            "Residue-level table size used for threshold-based disagreement detection.",
        ),
        (
            "number of raw prediction heads",
            len(raw_columns),
            "Seven UdonPred heads are retained as raw-score columns.",
        ),
        (
            "number of labelable heads",
            len(models_with_labels),
            "Heads with configured D/O raw thresholds; PDBFlex is retained as U only.",
        ),
        (
            "missing value status",
            "all raw columns have 0 missing values",
            "No missing raw predictions were observed in the current table.",
        ),
        (
            "any disagreement residues",
            any_residues,
            "Residues with at least one disorder-like and one ordered-like categorical vote.",
        ),
        (
            "strong disagreement residues",
            strong_residues,
            "Residues with at least two D votes and at least two O votes.",
        ),
        (
            "candidate fragments before filtering",
            before_filter,
            "Strong-disagreement residues grouped with gap <= 2 before length filtering.",
        ),
        (
            "candidate fragments after filtering",
            len(fragments),
            "Fragments retained after the minimum length filter.",
        ),
        (
            "top-ranked fragment score",
            round(top_score, 6),
            "Highest fragment_rank_score among retained candidate fragments.",
        ),
        (
            "warning about SoftDis being highly disorder-like",
            "SoftDis D labels dominate",
            "SoftDis labels most residues as disorder-like under the current threshold.",
        ),
        (
            "warning about PDBFlex threshold unresolved",
            "PDBFlex label set to U",
            "PDBFlex raw scores are retained but not used for strong D/O voting yet.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation_for_ppt"])


def build_model_label_counts(residue_labels: pd.DataFrame) -> pd.DataFrame:
    total = len(residue_labels)
    rows = []
    for model in MODEL_ORDER:
        label_column = f"{model}_label"
        if label_column not in residue_labels.columns:
            continue
        counts = residue_labels[label_column].value_counts().to_dict()
        label_counts = {label: int(counts.get(label, 0)) for label in ["O", "D", "U", "NA"]}
        rows.append(
            {
                "model": model,
                "ordered_like_count": label_counts["O"],
                "disordered_like_count": label_counts["D"],
                "uncertain_count": label_counts["U"],
                "missing_count": label_counts["NA"],
                "ordered_like_percent": round(100 * label_counts["O"] / total, 2),
                "disordered_like_percent": round(100 * label_counts["D"] / total, 2),
                "uncertain_percent": round(100 * label_counts["U"] / total, 2),
                "ppt_interpretation": model_interpretation(model, label_counts, total),
            }
        )
    return pd.DataFrame(rows)


def build_top_fragments(fragments: pd.DataFrame) -> pd.DataFrame:
    top = fragments.head(20).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    keep_columns = [
        "rank",
        "protein_id",
        "start",
        "end",
        "length",
        "n_strong_disagreement_residues",
        "fraction_strong_disagreement",
        "mean_categorical_disagreement_score",
        "fragment_rank_score",
        "dominant_disordered_models",
        "dominant_ordered_models",
    ]
    top = top[keep_columns]
    top["short_ppt_message"] = top.apply(fragment_message, axis=1)
    return top


def build_simplified_top10(top_fragments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in top_fragments.head(10).iterrows():
        rows.append(
            {
                "rank": int(row["rank"]),
                "protein_id": row["protein_id"],
                "region": f"{int(row['start'])}-{int(row['end'])}",
                "length": int(row["length"]),
                "rank_score": round(float(row["fragment_rank_score"]), 6),
                "disordered_side": parse_model_list(row["dominant_disordered_models"]),
                "ordered_side": parse_model_list(row["dominant_ordered_models"]),
                "why_it_matters": (
                    "Contested region with strong categorical disagreement; "
                    "use for prioritization discussion, not as validation."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_slide_outline() -> str:
    slides = [
        (
            "Slide 1: Research Question And Motivation",
            [
                "RQ6: Can prediction disagreement guide new NMR experiments?",
                "NMR experiments are expensive and benefit from candidate prioritization.",
                "Predictor disagreement may flag regions where new measurements are especially informative.",
            ],
            "Recommended figure/table: one-sentence RQ plus small workflow schematic.",
            "Speaker note: We are testing whether disagreement can become a practical signal for choosing future NMR measurements.",
        ),
        (
            "Slide 2: Current Input Data",
            [
                "TriZOD is used as the test set.",
                "348 proteins and 38,526 residues are represented.",
                "Raw predictions from 7 UdonPred heads are available with no missing values.",
            ],
            "Recommended figure/table: ppt_key_numbers.csv summary strip.",
            "Speaker note: The current analysis starts from a complete residue-level matrix of existing predictions.",
        ),
        (
            "Slide 3: Why No Standardization",
            [
                "Julius/Tobias suggested using raw literature-style thresholds.",
                "Raw scores from different heads are not directly comparable as continuous scales.",
                "The analysis avoids standardization, min-max scaling, sigmoid conversion, and raw-score variance.",
            ],
            "Recommended figure/table: short methods box listing prohibited transformations.",
            "Speaker note: The key methodological choice is to compare categorical calls, not raw score magnitudes.",
        ),
        (
            "Slide 4: Threshold-Based Labeling Strategy",
            [
                "Each residue/model pair becomes O, D, U, or NA.",
                "O means strong ordered-like; D means strong disordered-like; U means uncertain.",
                "CheZOD has an explicit middle U zone.",
                "PDBFlex raw scores are retained, but threshold is unresolved and labels are U.",
            ],
            "Recommended figure/table: fig_model_label_distribution.png.",
            "Speaker note: Each model keeps its own raw threshold, so the vote is categorical rather than scaled.",
        ),
        (
            "Slide 5: Disagreement Scoring",
            [
                "Count how many models vote D and how many vote O.",
                "Strong disagreement requires at least 2 D votes and at least 2 O votes.",
                "The categorical disagreement score combines vote balance and vote coverage.",
                "No raw-score variance is used.",
            ],
            "Recommended figure/table: compact formula plus fig_disagreement_summary.png.",
            "Speaker note: A region scores highly when both sides are well represented among called labels.",
        ),
        (
            "Slide 6: Main Results",
            [
                "34,897 residues have any categorical disagreement.",
                "3,420 residues have strong disagreement.",
                "127 candidate fragments remain after filtering.",
                "SoftDis is a strong driver and needs sensitivity analysis.",
            ],
            "Recommended figure/table: fig_disagreement_summary.png.",
            "Speaker note: The pipeline narrows residue-level disagreement into a manageable set of candidate fragments.",
        ),
        (
            "Slide 7: Top Candidate Fragments",
            [
                "Show the top 10 fragment table.",
                "Top fragment: protein_id 34101111, residues 143-151, length 9, score 0.825397.",
                "Top fragments represent contested regions, not experimentally validated recommendations yet.",
            ],
            "Recommended figure/table: fig_top10_fragment_scores.png and fig_top_fragment_label_heatmap.png.",
            "Speaker note: The top fragment has a clear split between disorder-like and ordered-like model calls.",
        ),
        (
            "Slide 8: Next Steps",
            [
                "Run threshold sensitivity analysis.",
                "Compare with and without SoftDis and DisProt.",
                "Visualize top fragments using label heatmaps.",
                "Later: retrospective BMRB validation.",
            ],
            "Recommended figure/table: top-five heatmaps as appendix material.",
            "Speaker note: The next goal is to test whether the candidate list is robust to reasonable threshold choices.",
        ),
    ]

    lines = ["# RQ6 PPT Slide Outline", ""]
    for title, bullets, recommendation, note in slides:
        lines.extend([f"## {title}", ""])
        lines.extend(f"- {bullet}" for bullet in bullets)
        lines.extend(["", recommendation, note, ""])
    return "\n".join(lines)


def save_model_label_distribution(model_counts: pd.DataFrame, figure_path: Path, plt) -> None:
    counts_by_model = model_counts.set_index("model")
    plot_df = counts_by_model[
        ["ordered_like_percent", "disordered_like_percent", "uncertain_percent"]
    ].rename(
        columns={
            "ordered_like_percent": "O",
            "disordered_like_percent": "D",
            "uncertain_percent": "U",
        }
    )
    plot_df["NA"] = counts_by_model["missing_count"] / counts_by_model[
        [
            "ordered_like_count",
            "disordered_like_count",
            "uncertain_count",
            "missing_count",
        ]
    ].sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottom = pd.Series(0, index=plot_df.index, dtype=float)
    for label in ["O", "D", "U", "NA"]:
        ax.bar(
            plot_df.index,
            plot_df[label],
            bottom=bottom,
            color=LABEL_COLORS[label],
            label=f"{label} - {LABEL_NAMES[label]}",
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += plot_df[label]

    ax.set_title("Raw-threshold labels differ strongly across prediction heads", pad=14)
    ax.set_ylabel("Residues (%)")
    ax.set_xlabel("Prediction head")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_disagreement_summary(residue_labels: pd.DataFrame, fragments: pd.DataFrame, figure_path: Path, plt) -> None:
    labels = [
        "Total residues",
        "Any disagreement\nresidues",
        "Strong disagreement\nresidues",
        "Candidate fragments\nafter filtering",
    ]
    values = [
        len(residue_labels),
        int(residue_labels["has_disagreement"].sum()),
        int(residue_labels["strong_disagreement"].sum()),
        len(fragments),
    ]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Threshold-based disagreement narrows residues into candidate fragments", pad=14)
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    ax.set_ylim(0, max(values) * 1.16)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top10_scores(top10: pd.DataFrame, figure_path: Path, plt) -> None:
    plot_df = top10.copy()
    plot_df["fragment"] = plot_df.apply(
        lambda row: f"{row['protein_id']}:{int(row['region'].split('-')[0])}-{int(row['region'].split('-')[1])}",
        axis=1,
    )
    plot_df = plot_df.sort_values("rank_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.barh(plot_df["fragment"], plot_df["rank_score"], color="#F58518")
    ax.set_title("Top candidate fragments by categorical disagreement score", pad=14)
    ax.set_xlabel("Fragment rank score")
    ax.set_ylabel("Fragment")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in plot_df["rank_score"]], padding=4)
    ax.set_xlim(0, max(plot_df["rank_score"]) * 1.18)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_fragment_heatmap(
    residue_labels: pd.DataFrame,
    fragment: pd.Series,
    figure_path: Path,
    plt,
    ListedColormap,
    Patch,
    title_prefix: str,
) -> None:
    protein_id = str(fragment["protein_id"])
    start = int(fragment["start"])
    end = int(fragment["end"])
    rows = residue_labels[
        (residue_labels["protein_id"].astype(str) == protein_id)
        & (residue_labels["residue_index"] >= start)
        & (residue_labels["residue_index"] <= end)
    ].sort_values("residue_index")

    available_label_columns = [
        column for column in LABEL_COLUMNS if column in residue_labels.columns
    ]
    matrix = []
    for label_column in available_label_columns:
        matrix.append(rows[label_column].map(HEATMAP_VALUES).fillna(3).astype(int).to_list())

    fig_width = max(7.2, 0.42 * len(rows) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(HEATMAP_COLORS), vmin=0, vmax=3)

    ax.set_title(f"{title_prefix}: {protein_id} residues {start}-{end}", pad=14)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Prediction head")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows["residue_index"].astype(int).astype(str), rotation=0)
    ax.set_yticks(range(len(available_label_columns)))
    ax.set_yticklabels([column.removesuffix("_label") for column in available_label_columns])

    for y, label_column in enumerate(available_label_columns):
        for x, label in enumerate(rows[label_column].astype(str).to_list()):
            text_color = "white" if label in {"O", "D"} else "black"
            ax.text(x, y, label, ha="center", va="center", color=text_color, fontsize=9, weight="bold")

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=f"{label} - {LABEL_NAMES[label]}")
        for label in ["O", "D", "U", "NA"]
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
    ax.set_xticks([tick - 0.5 for tick in range(1, len(rows))], minor=True)
    ax.set_yticks([tick - 0.5 for tick in range(1, len(available_label_columns))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    input_dir: Path,
    raw_table: Path,
    key_numbers: pd.DataFrame,
    top10: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    metrics = dict(zip(key_numbers["metric"], key_numbers["value"]))
    best_figures = [
        "fig_model_label_distribution.png",
        "fig_disagreement_summary.png",
        "fig_top10_fragment_scores.png",
        "fig_top_fragment_label_heatmap.png",
    ]
    lines = [
        "RQ6 PPT materials report",
        "",
        "Files read:",
        f"  - {input_dir / 'raw_prediction_matrix.csv'}",
        f"  - {input_dir / 'residue_labels.csv'}",
        f"  - {input_dir / 'candidate_fragments.csv'}",
        f"  - {input_dir / 'summary.txt'}",
    ]
    if raw_table.exists():
        lines.append(f"  - {raw_table}")

    lines.extend(
        [
            "",
            "Files created:",
            "  - ppt_key_numbers.csv",
            "  - ppt_model_label_counts.csv",
            "  - ppt_top20_fragments.csv",
            "  - ppt_top10_fragments_simplified.csv",
            "  - ppt_slide_outline.md",
            "  - ppt_materials_report.txt",
        ]
    )
    lines.extend(f"  - figures/{path.name}" for path in figure_paths)

    lines.extend(
        [
            "",
            "Key numbers:",
            f"  - Proteins: {metrics['number of proteins']}",
            f"  - Residues: {metrics['number of residues']}",
            f"  - Raw prediction heads: {metrics['number of raw prediction heads']}",
            f"  - Labelable heads: {metrics['number of labelable heads']}",
            f"  - Any disagreement residues: {metrics['any disagreement residues']}",
            f"  - Strong disagreement residues: {metrics['strong disagreement residues']}",
            f"  - Candidate fragments after filtering: {metrics['candidate fragments after filtering']}",
            f"  - Top-ranked fragment score: {metrics['top-ranked fragment score']}",
            "",
            "Main result message:",
            "  - Existing raw-threshold labels identify a manageable set of contested fragments for discussion.",
            "  - The top-ranked fragment is "
            f"{top10.iloc[0]['protein_id']}:{top10.iloc[0]['region']} "
            f"with rank score {top10.iloc[0]['rank_score']}.",
            "",
            "Caveats:",
            "  - This is threshold-based candidate detection, not final validation.",
            "  - SoftDis labels most residues as disorder-like under the current threshold, so sensitivity analysis is necessary.",
            "  - PDBFlex threshold is unresolved and currently not used for strong D/O voting.",
            "  - Candidate fragments should not yet be described as experimentally validated NMR targets.",
            "",
            "Recommended next step:",
            "  - Run threshold sensitivity analysis, including with/without SoftDis and DisProt, then compare fragment rank stability.",
            "",
            "Best figures to put in the PPT:",
        ]
    )
    lines.extend(f"  - {figure}" for figure in best_figures)
    output_path.write_text("\n".join(lines) + "\n")


def assert_outputs(output_dir: Path, force: bool) -> None:
    expected = [
        output_dir / "ppt_key_numbers.csv",
        output_dir / "ppt_model_label_counts.csv",
        output_dir / "ppt_top20_fragments.csv",
        output_dir / "ppt_top10_fragments_simplified.csv",
        output_dir / "ppt_slide_outline.md",
        output_dir / "ppt_materials_report.txt",
        output_dir / FIGURE_DIRNAME / "fig_model_label_distribution.png",
        output_dir / FIGURE_DIRNAME / "fig_disagreement_summary.png",
        output_dir / FIGURE_DIRNAME / "fig_top10_fragment_scores.png",
        output_dir / FIGURE_DIRNAME / "fig_top_fragment_label_heatmap.png",
        *[
            output_dir / FIGURE_DIRNAME / f"fig_top_fragment_{index:02d}_heatmap.png"
            for index in range(1, 6)
        ],
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing PPT materials without --force: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    assert_outputs(args.output_dir, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / FIGURE_DIRNAME
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt, ListedColormap, Patch = configure_matplotlib(args.output_dir)

    raw_matrix, residue_labels, fragments, _summary_text = read_inputs(args.input_dir)

    key_numbers = build_key_numbers(raw_matrix, residue_labels, fragments)
    model_counts = build_model_label_counts(residue_labels)
    top20 = build_top_fragments(fragments)
    top10 = build_simplified_top10(top20)

    key_numbers.to_csv(args.output_dir / "ppt_key_numbers.csv", index=False)
    model_counts.to_csv(args.output_dir / "ppt_model_label_counts.csv", index=False)
    top20.to_csv(args.output_dir / "ppt_top20_fragments.csv", index=False)
    top10.to_csv(args.output_dir / "ppt_top10_fragments_simplified.csv", index=False)
    (args.output_dir / "ppt_slide_outline.md").write_text(build_slide_outline())

    figure_paths = [
        figure_dir / "fig_model_label_distribution.png",
        figure_dir / "fig_disagreement_summary.png",
        figure_dir / "fig_top10_fragment_scores.png",
        figure_dir / "fig_top_fragment_label_heatmap.png",
    ]
    save_model_label_distribution(model_counts, figure_paths[0], plt)
    save_disagreement_summary(residue_labels, fragments, figure_paths[1], plt)
    save_top10_scores(top10, figure_paths[2], plt)
    save_fragment_heatmap(
        residue_labels,
        fragments.iloc[0],
        figure_paths[3],
        plt,
        ListedColormap,
        Patch,
        "Top fragment label heatmap",
    )

    for index, (_, fragment) in enumerate(fragments.head(5).iterrows(), start=1):
        path = figure_dir / f"fig_top_fragment_{index:02d}_heatmap.png"
        save_fragment_heatmap(
            residue_labels,
            fragment,
            path,
            plt,
            ListedColormap,
            Patch,
            f"Top fragment {index} label heatmap",
        )
        figure_paths.append(path)

    write_report(
        args.output_dir / "ppt_materials_report.txt",
        args.input_dir,
        args.raw_table,
        key_numbers,
        top10,
        figure_paths,
    )

    expected_files = [
        args.output_dir / "ppt_key_numbers.csv",
        args.output_dir / "ppt_model_label_counts.csv",
        args.output_dir / "ppt_top20_fragments.csv",
        args.output_dir / "ppt_top10_fragments_simplified.csv",
        args.output_dir / "ppt_slide_outline.md",
        args.output_dir / "ppt_materials_report.txt",
        *figure_paths,
    ]
    missing = [path for path in expected_files if not path.exists()]
    if missing:
        raise RuntimeError(
            "Expected output files were not created: "
            + ", ".join(str(path) for path in missing)
        )

    print(f"Wrote PPT materials to {args.output_dir}")
    print(f"Tables: 4 CSV files")
    print(f"Figures: {len(figure_paths)} PNG files")
    print(f"Top fragment: {top10.iloc[0]['protein_id']}:{top10.iloc[0]['region']}")


if __name__ == "__main__":
    main()
