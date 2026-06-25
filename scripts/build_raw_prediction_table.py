#!/usr/bin/env python3
"""Build residue-level raw prediction tables from cached UdonPred CAID files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["protein_id", "residue_index", "aa"]
CAID_COLUMNS = ["protein_id", "residue_index", "aa", "score", "model"]
DEFAULT_HEADS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
DEFAULT_PREDICTIONS_DIR = Path("results/udonpred_matrix/predictions")
DEFAULT_OUTPUT_DIR = Path("results/nmr_disagreement/tables")


def parse_caid_file(path: str | Path, model: str) -> pd.DataFrame:
    """Parse one UdonPred .caid file into a residue-level long table.

    UdonPred writes one protein per .caid file. The first line is the protein
    header, and each following line contains residue index, amino acid, and raw
    prediction score separated by tabs. Some lines end with an extra trailing
    tab, so only the first three fields are meaningful here.
    """
    path = Path(path)

    if path.is_dir():
        raise IsADirectoryError(
            f"parse_caid_file expects one .caid file, but got a directory: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(path)

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
                "score": float(parts[2]),
                "model": model,
            }
        )

    return pd.DataFrame(rows, columns=CAID_COLUMNS)


def collect_predictions_for_head(head_dir: str | Path, head_name: str) -> pd.DataFrame:
    """Parse all .caid files for one model/head into one long table."""
    head_dir = Path(head_dir)

    if not head_dir.exists():
        raise FileNotFoundError(head_dir)
    if not head_dir.is_dir():
        raise NotADirectoryError(head_dir)

    caid_paths = sorted(head_dir.glob("*.caid"))
    if not caid_paths:
        raise ValueError(f"No .caid files found in {head_dir}")

    frames = [parse_caid_file(path, head_name) for path in caid_paths]
    return pd.concat(frames, ignore_index=True)


def prepare_head_predictions_for_merge(
    head_frame: pd.DataFrame, head_name: str
) -> pd.DataFrame:
    """Convert one head's long table to a merge-ready raw-score table."""
    required_columns = KEY_COLUMNS + ["score"]
    missing_columns = [
        column for column in required_columns if column not in head_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    output = head_frame[required_columns].copy()
    output = output.rename(columns={"score": f"{head_name}_raw"})

    duplicate_mask = output.duplicated(KEY_COLUMNS)
    if duplicate_mask.any():
        example = output.loc[duplicate_mask, KEY_COLUMNS].head(1).to_dict("records")[0]
        raise ValueError(f"Duplicate residue key for {head_name}: {example}")

    return output


def merge_head_tables(head_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge prepared per-head raw-score tables on residue identity."""
    if not head_tables:
        raise ValueError("No head tables were provided for merging.")

    merged = head_tables[0].copy()
    for table in head_tables[1:]:
        missing_keys = [column for column in KEY_COLUMNS if column not in table.columns]
        if missing_keys:
            raise ValueError(f"Head table is missing key columns: {missing_keys}")
        merged = merged.merge(table, on=KEY_COLUMNS, how="outer", validate="one_to_one")

    return merged


def build_raw_prediction_table(
    predictions_dir: str | Path,
    test_dataset: str,
    heads: list[str] | None = None,
) -> pd.DataFrame:
    """Build a residue-level raw wide table for one fixed test dataset."""
    predictions_dir = Path(predictions_dir)
    heads = DEFAULT_HEADS if heads is None else heads

    prepared_tables = []
    for head in heads:
        head_dir = predictions_dir / f"{head}_{test_dataset}"
        head_frame = collect_predictions_for_head(head_dir, head)
        prepared_tables.append(prepare_head_predictions_for_merge(head_frame, head))

    return merge_head_tables(prepared_tables)


def validate_merged_table(
    table: pd.DataFrame, heads: list[str] | None = None
) -> dict[str, int]:
    """Validate a merged raw prediction table and return basic counts."""
    heads = DEFAULT_HEADS if heads is None else heads
    expected_columns = KEY_COLUMNS + [f"{head}_raw" for head in heads]

    missing_columns = [
        column for column in expected_columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    duplicate_count = int(table.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        example = table.loc[table.duplicated(KEY_COLUMNS), KEY_COLUMNS].head(1)
        raise ValueError(
            f"Found {duplicate_count} duplicate residue keys; "
            f"first example: {example.to_dict('records')[0]}"
        )

    empty_protein_count = int(table["protein_id"].astype(str).str.len().eq(0).sum())
    if empty_protein_count:
        raise ValueError(f"Found {empty_protein_count} rows with empty protein_id")

    non_positive_residue_count = int((table["residue_index"] <= 0).sum())
    if non_positive_residue_count:
        raise ValueError(
            f"Found {non_positive_residue_count} rows with non-positive residue_index"
        )

    empty_aa_count = int(table["aa"].astype(str).str.len().eq(0).sum())
    if empty_aa_count:
        raise ValueError(f"Found {empty_aa_count} rows with empty aa")

    raw_columns = [f"{head}_raw" for head in heads]
    raw_nan_counts = table[raw_columns].isna().sum()
    if raw_nan_counts.any():
        missing = {
            column: int(count)
            for column, count in raw_nan_counts.items()
            if int(count) > 0
        }
        raise ValueError(f"Missing raw prediction values: {missing}")

    return {
        "rows": int(len(table)),
        "proteins": int(table["protein_id"].nunique()),
        "raw_columns": len(raw_columns),
    }


def default_output_path(test_dataset: str) -> Path:
    """Return the default output path for one test dataset."""
    return DEFAULT_OUTPUT_DIR / f"{test_dataset}_raw_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory containing cached <head>_<test_dataset> .caid folders.",
    )
    parser.add_argument(
        "--test-dataset",
        choices=DEFAULT_HEADS,
        default="trizod",
        help="Fixed test dataset to merge across all heads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to results/nmr_disagreement/tables/<test_dataset>_raw_predictions.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.test_dataset)

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to overwrite."
        )

    table = build_raw_prediction_table(args.predictions_dir, args.test_dataset)
    summary = validate_merged_table(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    print(f"Wrote {output_path}")
    print(
        f"Rows: {summary['rows']} | "
        f"Proteins: {summary['proteins']} | "
        f"Raw columns: {summary['raw_columns']}"
    )


if __name__ == "__main__":
    main()
