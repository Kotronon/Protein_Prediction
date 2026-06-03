#!/usr/bin/env python3
"""Split a FASTA file into record or residue-budget chunks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


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
        records.append(FastaRecord(header=header, sequence="".join(sequence_parts)))
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


def write_record(handle, record: FastaRecord, width: int = 80) -> None:
    handle.write(f">{record.header}\n")
    for start in range(0, len(record.sequence), width):
        handle.write(f"{record.sequence[start:start + width]}\n")


def build_chunks(
    records: list[FastaRecord],
    records_per_chunk: int,
    max_residues_per_chunk: int | None,
) -> list[list[FastaRecord]]:
    if records_per_chunk < 1:
        raise ValueError("--records-per-chunk must be >= 1")
    if max_residues_per_chunk is not None and max_residues_per_chunk < 1:
        raise ValueError("--max-residues-per-chunk must be >= 1")

    chunks: list[list[FastaRecord]] = []
    current: list[FastaRecord] = []
    current_residues = 0

    for record in records:
        record_len = len(record.sequence)
        would_exceed_records = len(current) >= records_per_chunk
        would_exceed_residues = (
            max_residues_per_chunk is not None
            and current
            and current_residues + record_len > max_residues_per_chunk
        )
        if would_exceed_records or would_exceed_residues:
            chunks.append(current)
            current = []
            current_residues = 0

        current.append(record)
        current_residues += record_len

    if current:
        chunks.append(current)
    return chunks


def split_fasta(
    input_fasta: Path,
    output_dir: Path,
    records_per_chunk: int,
    prefix: str,
    max_residues_per_chunk: int | None,
) -> None:
    records = read_fasta(input_fasta)
    chunks = build_chunks(records, records_per_chunk, max_residues_per_chunk)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    rows: list[dict[str, object]] = []
    total_chunks = len(chunks)
    for chunk_index, chunk_records in enumerate(chunks, start=1):
        chunk_path = output_dir / f"{prefix}_{chunk_index:04d}.fasta"
        residues = sum(len(record.sequence) for record in chunk_records)
        with chunk_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in chunk_records:
                write_record(handle, record)
        rows.append(
            {
                "chunk": chunk_index,
                "path": str(chunk_path),
                "records": len(chunk_records),
                "residues": residues,
                "first_id": chunk_records[0].header.split()[0],
                "last_id": chunk_records[-1].header.split()[0],
            }
        )
        print(
            f"[{chunk_index}/{total_chunks}] wrote {chunk_path} "
            f"({len(chunk_records)} records, {residues} residues)"
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["chunk", "path", "records", "residues", "first_id", "last_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} chunks for {len(records)} records. Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fasta", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--records-per-chunk", type=int, default=500)
    parser.add_argument(
        "--max-residues-per-chunk",
        type=int,
        default=None,
        help="Optional residue budget per chunk. Chunks split when adding a record would exceed it.",
    )
    parser.add_argument("--prefix", default="chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_fasta(
        args.input_fasta,
        args.output_dir,
        args.records_per_chunk,
        args.prefix,
        args.max_residues_per_chunk,
    )


if __name__ == "__main__":
    main()
