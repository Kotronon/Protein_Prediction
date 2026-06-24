#!/usr/bin/env python3
"""Convert metapredict FASTA CSV output to CAID-like residue-score format."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from protein_prediction_io import read_fasta_dict


@dataclass(frozen=True)
class MetapredictRecord:
    sequence: str | None
    scores: list[float]


def read_metapredict_csv(path: Path) -> dict[str, MetapredictRecord]:
    predictions: dict[str, MetapredictRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            header = row[0].strip()
            if not header:
                raise ValueError(f"{path}:{line_number}: missing sequence header")
            if header in predictions:
                raise ValueError(f"{path}:{line_number}: duplicate sequence header {header!r}")

            score_start = 1
            sequence: str | None = None
            if len(row) > 1:
                try:
                    float(row[1].strip())
                except ValueError:
                    sequence = row[1].strip()
                    score_start = 2

            try:
                scores = [float(value.strip()) for value in row[score_start:] if value.strip()]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric disorder score") from exc
            predictions[header] = MetapredictRecord(sequence=sequence, scores=scores)

    if not predictions:
        raise ValueError(f"No metapredict rows found in {path}")
    return predictions


def build_header_lookup(fasta_records: dict[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for header in fasta_records:
        lookup[header] = header
        lookup[header.replace(",", " ")] = header
    return lookup


def convert(csv_path: Path, fasta_path: Path, output_path: Path) -> None:
    fasta_records = read_fasta_dict(fasta_path)
    predictions = read_metapredict_csv(csv_path)
    header_lookup = build_header_lookup(fasta_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for predicted_header, prediction in predictions.items():
            fasta_header = header_lookup.get(predicted_header)
            if fasta_header is None:
                raise ValueError(f"{csv_path}: header not found in FASTA: {predicted_header!r}")

            fasta_sequence = fasta_records[fasta_header]
            sequence = prediction.sequence or fasta_sequence
            if len(sequence) != len(fasta_sequence):
                raise ValueError(
                    f"{predicted_header!r}: CSV sequence length {len(sequence)} != "
                    f"FASTA length {len(fasta_sequence)}"
                )
            if len(sequence) != len(prediction.scores):
                raise ValueError(
                    f"{predicted_header!r}: FASTA length {len(sequence)} != "
                    f"score count {len(prediction.scores)}"
                )

            handle.write(f">{fasta_header}\n")
            for index, (residue, score) in enumerate(zip(sequence, prediction.scores), start=1):
                handle.write(f"{index} {residue} {score:.6g}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metapredict_csv", type=Path)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("output_caid", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.metapredict_csv, args.fasta, args.output_caid)


if __name__ == "__main__":
    main()
