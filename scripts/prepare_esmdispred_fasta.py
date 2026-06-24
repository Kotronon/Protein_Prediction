#!/usr/bin/env python3
"""Prepare a FASTA file for ESMDisPred/DisPredict3.0.

ESMDisPred's bundled DisPredict3.0/flDPnn tools are sensitive to FASTA
headers because some helper programs use headers as temporary file names.
This script writes simple UniProt-accession headers and drops records that
ESMDisPred would ignore anyway: non-canonical amino acids and very long
sequences.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from protein_prediction_io import (
    extract_uniprot_accession,
    read_fasta,
    write_wrapped_sequence,
)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def prepare_fasta(
    input_fasta: Path,
    output_fasta: Path,
    report_csv: Path,
    min_length: int,
    max_length: int,
    limit: int | None,
) -> None:
    records = read_fasta(input_fasta, uppercase=True)
    seen_ids: set[str] = set()
    kept = 0
    skipped_short = 0
    skipped_length = 0
    skipped_invalid = 0

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_fasta.open("w", encoding="utf-8", newline="\n") as fasta_handle, report_csv.open(
        "w", encoding="utf-8", newline=""
    ) as report_handle:
        writer = csv.DictWriter(
            report_handle,
            fieldnames=[
                "status",
                "prepared_id",
                "length",
                "invalid_characters",
                "original_header",
            ],
        )
        writer.writeheader()

        for record in records:
            prepared_id = extract_uniprot_accession(record.header)
            invalid_characters = "".join(sorted(set(record.sequence) - CANONICAL_AA))
            status = "kept"

            if len(record.sequence) < min_length:
                status = "skipped_short"
                skipped_short += 1
            elif len(record.sequence) > max_length:
                status = "skipped_length"
                skipped_length += 1
            elif invalid_characters:
                status = "skipped_invalid"
                skipped_invalid += 1
            elif prepared_id in seen_ids:
                status = "skipped_duplicate_id"
            elif limit is not None and kept >= limit:
                status = "skipped_limit"

            writer.writerow(
                {
                    "status": status,
                    "prepared_id": prepared_id,
                    "length": len(record.sequence),
                    "invalid_characters": invalid_characters,
                    "original_header": record.header,
                }
            )

            if status != "kept":
                continue

            seen_ids.add(prepared_id)
            kept += 1
            fasta_handle.write(f">{prepared_id}\n")
            write_wrapped_sequence(fasta_handle, record.sequence)

    print(
        f"Wrote {kept} records to {output_fasta}; "
        f"skipped {skipped_short} short, {skipped_length} length, "
        f"{skipped_invalid} invalid/non-canonical. "
        f"Report: {report_csv}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fasta", type=Path)
    parser.add_argument("output_fasta", type=Path)
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Path for the preparation report CSV. Defaults to <output>.report.csv.",
    )
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Drop very short proteins; flDPnn/PSIPRED can fail to produce .ss2 files for them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Keep only the first N valid records, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_csv = args.report_csv or args.output_fasta.with_suffix(
        f"{args.output_fasta.suffix}.report.csv"
    )
    prepare_fasta(
        args.input_fasta,
        args.output_fasta,
        report_csv,
        args.min_length,
        args.max_length,
        args.limit,
    )


if __name__ == "__main__":
    main()
