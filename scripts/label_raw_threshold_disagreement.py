#!/usr/bin/env python3
"""Detect threshold-based prediction disagreement on TriZOD residues.

This script intentionally keeps each UdonPred head on its own raw score scale.
It labels residues with head-specific raw thresholds, then computes categorical
disagreement summaries without standardization, min-max scaling, sigmoid
conversion, pLDDT inversion, or raw-score variance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLUMNS = ["protein_id", "residue_index", "aa"]
DEFAULT_TEST_DATASET = "trizod"
DEFAULT_HEADS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
DEFAULT_PREDICTIONS_DIR = Path("results/udonpred_matrix/predictions")
DEFAULT_RAW_TABLE = Path("results/nmr_disagreement/tables/trizod_raw_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("results/rq6_threshold_disagreement")

MAX_GAP = 2
MIN_FRAGMENT_LENGTH = 8

THRESHOLD_CONFIG: dict[str, dict[str, Any]] = {
    "trizod": {
        "raw_column": "trizod_raw",
        "label_column": "trizod_label",
        "disordered": {">=": 0.4},
        "ordered": {"<": 0.4},
    },
    "chezod": {
        "raw_column": "chezod_raw",
        "label_column": "chezod_label",
        "disordered": {"<=": 3.0},
        "ordered": {">=": 8.0},
        "uncertain": "U",
    },
    "atlas": {
        "raw_column": "atlas_raw",
        "label_column": "atlas_label",
        "disordered": {">=": 2.0},
        "ordered": {"<": 2.0},
    },
    "plddt": {
        "raw_column": "plddt_raw",
        "label_column": "plddt_label",
        "disordered": {"<=": 68.8},
        "ordered": {">": 68.8},
    },
    "softdis": {
        "raw_column": "softdis_raw",
        "label_column": "softdis_label",
        "disordered": {">=": 0.025},
        "ordered": {"<": 0.025},
    },
    "disprot": {
        "raw_column": "disprot_raw",
        "label_column": "disprot_label",
        "disordered": {">=": 0.5},
        "ordered": {"<": 0.5},
    },
    "pdbflex": {
        "raw_column": "pdbflex_raw",
        "label_column": "pdbflex_label",
        "unresolved_threshold": True,
        "available_label": "U",
        "note": "No raw threshold configured; non-missing raw scores are labeled U.",
    },
}


@dataclass
class PipelineReport:
    input_files: list[str] = field(default_factory=list)
    caid_cache_counts: dict[str, int] = field(default_factory=dict)
    raw_columns_detected: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    candidate_fragments_before_filtering: int = 0
    candidate_fragments_after_filtering: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-table",
        type=Path,
        default=None,
        help=(
            "Existing residue-level raw prediction table. Defaults to "
            "results/nmr_disagreement/tables/trizod_raw_predictions.csv if present."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory containing cached <head>_<test_dataset> .caid folders.",
    )
    parser.add_argument(
        "--test-dataset",
        default=DEFAULT_TEST_DATASET,
        help="Test dataset suffix used for CAID cache directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pipeline outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs in --output-dir if they already exist.",
    )
    return parser.parse_args()


def compare_score(score: float, rule: dict[str, float]) -> bool:
    if len(rule) != 1:
        raise ValueError(f"Expected one comparison operator per rule, got: {rule}")
    operator, threshold = next(iter(rule.items()))
    if operator == ">=":
        return score >= threshold
    if operator == ">":
        return score > threshold
    if operator == "<=":
        return score <= threshold
    if operator == "<":
        return score < threshold
    raise ValueError(f"Unsupported threshold operator: {operator}")


def label_score(score: Any, config: dict[str, Any]) -> str:
    if pd.isna(score):
        return "NA"

    if config.get("unresolved_threshold"):
        return config.get("available_label", "U")

    score = float(score)
    if compare_score(score, config["disordered"]):
        return "D"
    if compare_score(score, config["ordered"]):
        return "O"
    return config.get("uncertain", "U")


def parse_caid_file(path: Path, model: str) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty CAID file: {path}")

    header = lines[0].strip()
    if not header.startswith(">"):
        raise ValueError(f"Expected CAID header starting with '>': {path}")
    protein_id = header.lstrip(">")

    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Bad CAID line {line_number} in {path}: {line!r}")
        rows.append(
            {
                "protein_id": protein_id,
                "residue_index": int(parts[0]),
                "aa": parts[1],
                f"{model}_raw": float(parts[2]),
            }
        )
    return pd.DataFrame(rows, columns=KEY_COLUMNS + [f"{model}_raw"])


def collect_caid_head(head_dir: Path, head: str) -> pd.DataFrame:
    caid_paths = sorted(head_dir.glob("*.caid"))
    if not caid_paths:
        raise ValueError(f"No .caid files found in {head_dir}")

    frame = pd.concat(
        [parse_caid_file(path, head) for path in caid_paths],
        ignore_index=True,
    )
    duplicated = frame.duplicated(KEY_COLUMNS)
    if duplicated.any():
        example = frame.loc[duplicated, KEY_COLUMNS].head(1).to_dict("records")[0]
        raise ValueError(f"Duplicate residue key for {head}: {example}")
    return frame


def build_raw_matrix_from_caid(
    predictions_dir: Path,
    test_dataset: str,
    report: PipelineReport,
) -> pd.DataFrame:
    head_tables = []
    missing_heads = []

    for head in DEFAULT_HEADS:
        head_dir = predictions_dir / f"{head}_{test_dataset}"
        if not head_dir.exists():
            missing_heads.append(head)
            continue
        report.input_files.append(str(head_dir))
        head_tables.append(collect_caid_head(head_dir, head))

    if missing_heads:
        report.warnings.append(
            "Missing CAID cache directories for heads: " + ", ".join(missing_heads)
        )
    if not head_tables:
        raise FileNotFoundError(
            f"No usable CAID cache directories found in {predictions_dir}"
        )

    merged = head_tables[0]
    for table in head_tables[1:]:
        merged = merged.merge(table, on=KEY_COLUMNS, how="outer", validate="one_to_one")

    return merged


def detect_caid_cache_counts(predictions_dir: Path, test_dataset: str) -> dict[str, int]:
    counts = {}
    for head in DEFAULT_HEADS:
        head_dir = predictions_dir / f"{head}_{test_dataset}"
        if head_dir.exists():
            counts[head] = len(list(head_dir.glob("*.caid")))
    return counts


def load_raw_matrix(args: argparse.Namespace, report: PipelineReport) -> pd.DataFrame:
    if args.raw_table is not None:
        raw_table = args.raw_table
    elif DEFAULT_RAW_TABLE.exists():
        raw_table = DEFAULT_RAW_TABLE
    else:
        raw_table = None

    if raw_table is not None:
        if not raw_table.exists():
            raise FileNotFoundError(raw_table)
        report.input_files.append(str(raw_table))
        return pd.read_csv(raw_table, dtype={"protein_id": "string", "aa": "string"})

    return build_raw_matrix_from_caid(args.predictions_dir, args.test_dataset, report)


def validate_raw_matrix(raw_matrix: pd.DataFrame, report: PipelineReport) -> pd.DataFrame:
    missing_key_columns = [column for column in KEY_COLUMNS if column not in raw_matrix.columns]
    if missing_key_columns:
        raise ValueError(f"Raw matrix is missing key columns: {missing_key_columns}")

    raw_matrix = raw_matrix.copy()
    raw_matrix["protein_id"] = raw_matrix["protein_id"].astype("string")
    raw_matrix["aa"] = raw_matrix["aa"].astype("string")
    raw_matrix["residue_index"] = pd.to_numeric(
        raw_matrix["residue_index"], errors="raise"
    ).astype("int64")

    duplicate_key_count = int(raw_matrix.duplicated(KEY_COLUMNS).sum())
    if duplicate_key_count:
        report.warnings.append(
            f"Duplicate protein_id + residue_index + aa rows: {duplicate_key_count}"
        )

    residue_aa_counts = raw_matrix.groupby(
        ["protein_id", "residue_index"], dropna=False
    )["aa"].nunique()
    aa_mismatch_count = int((residue_aa_counts > 1).sum())
    if aa_mismatch_count:
        report.warnings.append(
            f"AA mismatches at identical protein_id + residue_index: {aa_mismatch_count}"
        )

    empty_protein_count = int(raw_matrix["protein_id"].fillna("").str.len().eq(0).sum())
    if empty_protein_count:
        report.warnings.append(f"Rows with empty protein_id: {empty_protein_count}")

    empty_aa_count = int(raw_matrix["aa"].fillna("").str.len().eq(0).sum())
    if empty_aa_count:
        report.warnings.append(f"Rows with empty aa: {empty_aa_count}")

    non_positive_residue_count = int((raw_matrix["residue_index"] <= 0).sum())
    if non_positive_residue_count:
        report.warnings.append(
            f"Rows with non-positive residue_index: {non_positive_residue_count}"
        )

    raw_columns = sorted(column for column in raw_matrix.columns if column.endswith("_raw"))
    report.raw_columns_detected = raw_columns
    if not raw_columns:
        raise ValueError("No raw score columns ending in '_raw' were detected.")

    report.missing_values = {
        column: int(raw_matrix[column].isna().sum()) for column in raw_columns
    }
    missing_columns = {
        column: count for column, count in report.missing_values.items() if count > 0
    }
    if missing_columns:
        report.warnings.append(f"Missing raw prediction values: {missing_columns}")

    expected_rows_by_protein = raw_matrix.groupby("protein_id", dropna=False).size()
    inconsistent_models = []
    for column in raw_columns:
        observed = raw_matrix.groupby("protein_id", dropna=False)[column].apply(
            lambda values: int(values.notna().sum())
        )
        mismatched = observed[observed != expected_rows_by_protein]
        if not mismatched.empty:
            inconsistent_models.append(f"{column}: {len(mismatched)} proteins")
    if inconsistent_models:
        report.warnings.append(
            "Model output residue counts differ from merged protein lengths: "
            + "; ".join(inconsistent_models)
        )

    configured_raw_columns = {
        config["raw_column"] for config in THRESHOLD_CONFIG.values()
    }
    unconfigured_raw_columns = [
        column for column in raw_columns if column not in configured_raw_columns
    ]
    if unconfigured_raw_columns:
        report.warnings.append(
            "Raw columns retained without threshold labeling config: "
            + ", ".join(unconfigured_raw_columns)
        )

    missing_configured_columns = [
        config["raw_column"]
        for config in THRESHOLD_CONFIG.values()
        if config["raw_column"] not in raw_matrix.columns and config["raw_column"] != "pdbflex_raw"
    ]
    if missing_configured_columns:
        report.warnings.append(
            "Configured raw columns not found in inputs: "
            + ", ".join(missing_configured_columns)
        )

    return raw_matrix.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_raw_prediction_matrix(raw_matrix: pd.DataFrame) -> pd.DataFrame:
    raw_columns = [column for column in raw_matrix.columns if column.endswith("_raw")]
    preferred_order = KEY_COLUMNS + [
        config["raw_column"]
        for head, config in THRESHOLD_CONFIG.items()
        if config["raw_column"] in raw_columns
    ]
    extra_raw_columns = [
        column for column in raw_columns if column not in set(preferred_order)
    ]
    return raw_matrix[preferred_order + extra_raw_columns].copy()


def label_residues(raw_prediction_matrix: pd.DataFrame, report: PipelineReport) -> pd.DataFrame:
    labeled = raw_prediction_matrix.copy()
    label_columns = []

    for head, config in THRESHOLD_CONFIG.items():
        raw_column = config["raw_column"]
        if raw_column not in labeled.columns:
            continue
        label_column = config["label_column"]
        labeled[label_column] = labeled[raw_column].apply(lambda score: label_score(score, config))
        label_columns.append(label_column)

        counts = labeled[label_column].value_counts(dropna=False).to_dict()
        report.label_counts[head] = {
            label: int(counts.get(label, 0)) for label in ["O", "D", "U", "NA"]
        }

        if config.get("unresolved_threshold"):
            report.warnings.append(f"{head}: {config.get('note', 'threshold unresolved')}")

    if not label_columns:
        raise ValueError("No label columns could be generated from detected raw columns.")

    report.label_columns = label_columns
    label_frame = labeled[label_columns]
    labeled["n_disordered_like"] = (label_frame == "D").sum(axis=1)
    labeled["n_ordered_like"] = (label_frame == "O").sum(axis=1)
    labeled["n_uncertain"] = (label_frame == "U").sum(axis=1)
    labeled["n_missing"] = (label_frame == "NA").sum(axis=1)
    labeled["n_available"] = len(label_columns) - labeled["n_missing"]
    labeled["n_called"] = labeled["n_disordered_like"] + labeled["n_ordered_like"]
    labeled["has_disagreement"] = (
        (labeled["n_disordered_like"] >= 1) & (labeled["n_ordered_like"] >= 1)
    )
    labeled["strong_disagreement"] = (
        (labeled["n_disordered_like"] >= 2) & (labeled["n_ordered_like"] >= 2)
    )

    balance = pd.Series(0.0, index=labeled.index)
    called_mask = labeled["n_called"] > 0
    balance.loc[called_mask] = (
        2
        * labeled.loc[called_mask, ["n_disordered_like", "n_ordered_like"]].min(axis=1)
        / labeled.loc[called_mask, "n_called"]
    )

    coverage = pd.Series(0.0, index=labeled.index)
    available_mask = labeled["n_available"] > 0
    coverage.loc[available_mask] = (
        labeled.loc[available_mask, "n_called"]
        / labeled.loc[available_mask, "n_available"]
    )

    labeled["categorical_disagreement_score"] = balance * coverage
    return labeled


def summarize_dominant_models(fragment_rows: pd.DataFrame, label_columns: list[str], label: str) -> str:
    counts = {}
    for label_column in label_columns:
        head = label_column.removesuffix("_label")
        count = int((fragment_rows[label_column] == label).sum())
        if count > 0:
            counts[head] = count
    return ";".join(f"{head}:{count}" for head, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def iter_fragment_ranges(residue_labels: pd.DataFrame) -> list[tuple[str, int, int]]:
    ranges = []
    strong_rows = residue_labels.loc[
        residue_labels["strong_disagreement"],
        ["protein_id", "residue_index"],
    ].sort_values(["protein_id", "residue_index"])

    for protein_id, protein_rows in strong_rows.groupby("protein_id", sort=True):
        start = None
        previous = None
        for residue_index in protein_rows["residue_index"].astype(int):
            if start is None:
                start = residue_index
                previous = residue_index
                continue

            gap = residue_index - previous - 1
            if gap <= MAX_GAP:
                previous = residue_index
                continue

            ranges.append((str(protein_id), int(start), int(previous)))
            start = residue_index
            previous = residue_index

        if start is not None and previous is not None:
            ranges.append((str(protein_id), int(start), int(previous)))

    return ranges


def build_candidate_fragments(
    residue_labels: pd.DataFrame,
    label_columns: list[str],
    report: PipelineReport,
) -> pd.DataFrame:
    rows = []
    for protein_id, start, end in iter_fragment_ranges(residue_labels):
        fragment_rows = residue_labels[
            (residue_labels["protein_id"] == protein_id)
            & (residue_labels["residue_index"] >= start)
            & (residue_labels["residue_index"] <= end)
        ]
        if fragment_rows.empty:
            continue

        length = end - start + 1
        n_strong = int(fragment_rows["strong_disagreement"].sum())
        fraction_strong = n_strong / length if length else 0.0
        mean_score = float(fragment_rows["categorical_disagreement_score"].mean())
        max_score = float(fragment_rows["categorical_disagreement_score"].max())
        rank_score = mean_score * fraction_strong

        rows.append(
            {
                "protein_id": protein_id,
                "start": start,
                "end": end,
                "length": length,
                "n_strong_disagreement_residues": n_strong,
                "fraction_strong_disagreement": fraction_strong,
                "mean_categorical_disagreement_score": mean_score,
                "max_categorical_disagreement_score": max_score,
                "mean_n_disordered_like": float(fragment_rows["n_disordered_like"].mean()),
                "mean_n_ordered_like": float(fragment_rows["n_ordered_like"].mean()),
                "mean_n_called": float(fragment_rows["n_called"].mean()),
                "dominant_disordered_models": summarize_dominant_models(
                    fragment_rows, label_columns, "D"
                ),
                "dominant_ordered_models": summarize_dominant_models(
                    fragment_rows, label_columns, "O"
                ),
                "fragment_rank_score": rank_score,
            }
        )

    fragments = pd.DataFrame(rows)
    report.candidate_fragments_before_filtering = int(len(fragments))
    if fragments.empty:
        report.candidate_fragments_after_filtering = 0
        return fragments

    filtered = fragments.loc[fragments["length"] >= MIN_FRAGMENT_LENGTH].copy()
    report.candidate_fragments_after_filtering = int(len(filtered))
    return filtered.sort_values(
        ["fragment_rank_score", "mean_categorical_disagreement_score", "length"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def format_config() -> str:
    lines = [
        "Threshold config:",
        f"  MAX_GAP = {MAX_GAP}",
        f"  MIN_FRAGMENT_LENGTH = {MIN_FRAGMENT_LENGTH}",
    ]
    for head, config in THRESHOLD_CONFIG.items():
        if config.get("unresolved_threshold"):
            lines.append(f"  {head}: unresolved threshold; available raw scores -> U")
            continue
        lines.append(
            f"  {head}: D if {config['disordered']}; O if {config['ordered']}"
        )
    return "\n".join(lines)


def write_summary(
    output_path: Path,
    raw_prediction_matrix: pd.DataFrame,
    residue_labels: pd.DataFrame,
    candidate_fragments: pd.DataFrame,
    report: PipelineReport,
) -> None:
    lines = [
        "RQ6 threshold-based disagreement summary",
        "",
        "Important: no score standardization, min-max scaling, sigmoid conversion,",
        "pLDDT inversion, cross-model raw-score scaling, or raw-score variance was used.",
        "",
        format_config(),
        "",
        "Input files used:",
    ]
    lines.extend(f"  - {path}" for path in report.input_files)
    if not report.input_files:
        lines.append("  - None recorded")

    lines.extend(["", "TriZOD CAID cache detected:"])
    if report.caid_cache_counts:
        lines.extend(
            f"  - {head}: {count} .caid files"
            for head, count in sorted(report.caid_cache_counts.items())
        )
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            f"Number of proteins: {raw_prediction_matrix['protein_id'].nunique()}",
            f"Number of residues: {len(raw_prediction_matrix)}",
            "Raw score columns detected: " + ", ".join(report.raw_columns_detected),
            "",
            "Missing values per model:",
        ]
    )
    lines.extend(
        f"  - {column}: {count}" for column, count in sorted(report.missing_values.items())
    )

    lines.extend(["", "Label counts per model:"])
    for head, counts in report.label_counts.items():
        lines.append(
            f"  - {head}: O={counts['O']} D={counts['D']} U={counts['U']} NA={counts['NA']}"
        )

    lines.extend(
        [
            "",
            f"Residues with any disagreement: {int(residue_labels['has_disagreement'].sum())}",
            f"Residues with strong disagreement: {int(residue_labels['strong_disagreement'].sum())}",
            (
                "Candidate fragments before filtering: "
                f"{report.candidate_fragments_before_filtering}"
            ),
            (
                "Candidate fragments after filtering: "
                f"{report.candidate_fragments_after_filtering}"
            ),
            "",
            "Top 20 candidate fragments ranked by fragment_rank_score:",
        ]
    )
    if candidate_fragments.empty:
        lines.append("  - none")
    else:
        top_columns = [
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
        lines.append(candidate_fragments.head(20)[top_columns].to_string(index=False))

    lines.extend(["", "Warnings:"])
    if report.warnings:
        lines.extend(f"  - {warning}" for warning in report.warnings)
    else:
        lines.append("  - none")

    output_path.write_text("\n".join(lines) + "\n")


def assert_output_paths(output_dir: Path, force: bool) -> dict[str, Path]:
    paths = {
        "raw_prediction_matrix": output_dir / "raw_prediction_matrix.csv",
        "residue_labels": output_dir / "residue_labels.csv",
        "candidate_fragments": output_dir / "candidate_fragments.csv",
        "summary": output_dir / "summary.txt",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --force: "
            + ", ".join(str(path) for path in existing)
        )
    return paths


def main() -> None:
    args = parse_args()
    report = PipelineReport()
    report.caid_cache_counts = detect_caid_cache_counts(
        args.predictions_dir, args.test_dataset
    )
    output_paths = assert_output_paths(args.output_dir, args.force)

    raw_matrix = load_raw_matrix(args, report)
    raw_matrix = validate_raw_matrix(raw_matrix, report)
    raw_prediction_matrix = build_raw_prediction_matrix(raw_matrix)
    residue_labels = label_residues(raw_prediction_matrix, report)
    candidate_fragments = build_candidate_fragments(
        residue_labels, report.label_columns, report
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_prediction_matrix.to_csv(output_paths["raw_prediction_matrix"], index=False)
    residue_labels.to_csv(output_paths["residue_labels"], index=False)
    candidate_fragments.to_csv(output_paths["candidate_fragments"], index=False)
    write_summary(
        output_paths["summary"],
        raw_prediction_matrix,
        residue_labels,
        candidate_fragments,
        report,
    )

    print(f"Wrote {output_paths['raw_prediction_matrix']}")
    print(f"Wrote {output_paths['residue_labels']}")
    print(f"Wrote {output_paths['candidate_fragments']}")
    print(f"Wrote {output_paths['summary']}")
    print(
        "Residues: "
        f"{len(raw_prediction_matrix)} | Proteins: {raw_prediction_matrix['protein_id'].nunique()} | "
        f"Strong disagreement residues: {int(residue_labels['strong_disagreement'].sum())} | "
        f"Fragments after filtering: {len(candidate_fragments)}"
    )


if __name__ == "__main__":
    main()
