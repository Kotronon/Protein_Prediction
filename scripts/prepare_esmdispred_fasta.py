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
from dataclasses import dataclass
from pathlib import Path


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = "".join(sequence_parts).upper()
        if not sequence:
            raise ValueError(f"{path}: empty sequence for {header}")
        records.append(FastaRecord(header=header, sequence=sequence))
        header = None
        sequence_parts = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence before first FASTA header")
                sequence_parts.append(line)

    flush()
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def uniprot_accession(header: str) -> str:
    first_token = header.split()[0]
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def write_wrapped_sequence(handle, sequence: str, width: int = 80) -> None:
    for start in range(0, len(sequence), width):
        handle.write(f"{sequence[start:start + width]}\n")


def prepare_fasta(
    input_fasta: Path,
    output_fasta: Path,
    report_csv: Path,
    min_length: int,
    max_length: int,
    limit: int | None,
) -> None:
    records = read_fasta(input_fasta)
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
            prepared_id = uniprot_accession(record.header)
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
