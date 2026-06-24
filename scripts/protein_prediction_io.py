"""Shared file-format helpers for project analysis scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str

    @property
    def first_token(self) -> str:
        return self.header.split()[0]

    @property
    def uniprot_accession(self) -> str:
        return extract_uniprot_accession(self.header)


def extract_uniprot_accession(header: str) -> str:
    first_token = header.lstrip(">").strip().split()[0]
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def safe_filename(value: str, fallback: str = "sequence") -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or fallback


def read_fasta(path: Path, *, uppercase: bool = False) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush(line_number: int | None = None) -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = "".join(sequence_parts)
        if uppercase:
            sequence = sequence.upper()
        if not sequence:
            location = f"{path}:{line_number}" if line_number else str(path)
            raise ValueError(f"{location}: empty sequence for {header}")
        records.append(FastaRecord(header=header, sequence=sequence))
        header = None
        sequence_parts = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush(line_number)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(
                        f"{path}:{line_number}: sequence before first FASTA header"
                    )
                sequence_parts.append(line)

    flush()
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def read_fasta_dict(path: Path, *, uppercase: bool = False) -> dict[str, str]:
    records: dict[str, str] = {}
    for record in read_fasta(path, uppercase=uppercase):
        if record.header in records:
            raise ValueError(f"{path}: duplicate FASTA header {record.header!r}")
        records[record.header] = record.sequence
    return records


def write_wrapped_sequence(handle: TextIO, sequence: str, width: int = 80) -> None:
    for start in range(0, len(sequence), width):
        handle.write(f"{sequence[start:start + width]}\n")


def write_fasta_record(handle: TextIO, record: FastaRecord, width: int = 80) -> None:
    handle.write(f">{record.header}\n")
    write_wrapped_sequence(handle, record.sequence, width)


def write_caid_scores(
    path: Path,
    header: str,
    sequence: str,
    scores: Iterable[float],
    *,
    precision: int = 6,
) -> None:
    score_values = list(scores)
    if len(score_values) != len(sequence):
        raise ValueError(
            f"{path}: score count {len(score_values)} does not match "
            f"sequence length {len(sequence)}"
        )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f">{header.lstrip('>')}\n")
        for position, (residue, score) in enumerate(
            zip(sequence, score_values), start=1
        ):
            handle.write(f"{position}\t{residue}\t{score:.{precision}f}\n")
