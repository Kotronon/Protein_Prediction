#!/usr/bin/env python3
"""Run IUPred3 on a multi-record FASTA file and write CAID-like outputs."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IUPRED3_DIR = REPO_ROOT / "iupred3"
sys.path.insert(0, str(IUPRED3_DIR))

import iupred3_lib  # noqa: E402


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str


def extract_uniprot_accession(header: str) -> str:
    first_token = header.lstrip(">").strip().split()[0]
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush_record() -> None:
        nonlocal header, chunks
        if header is None:
            return
        sequence = "".join(chunks)
        if not sequence:
            raise ValueError(f"{path}: empty sequence for {header}")
        records.append(FastaRecord(header=header, sequence=sequence))
        header = None
        chunks = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"{path}:{line_number}: empty FASTA header")
            else:
                if header is None:
                    raise ValueError(f"{path}:{line_number}: sequence before first header")
                chunks.append(line)

    flush_record()
    if not records:
        raise ValueError(f"{path}: no FASTA records found")
    return records


def write_caid(
    path: Path,
    record: FastaRecord,
    iupred_scores: list[float],
    anchor_scores: list[float] | None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{record.header}\n")
        if anchor_scores is None:
            for pos, residue in enumerate(record.sequence, start=1):
                handle.write(f"{pos}\t{residue}\t{iupred_scores[pos - 1]:.4f}\n")
        else:
            for pos, residue in enumerate(record.sequence, start=1):
                handle.write(
                    f"{pos}\t{residue}\t{iupred_scores[pos - 1]:.4f}\t"
                    f"{anchor_scores[pos - 1]:.4f}\n"
                )


def effective_smoothing(sequence: str, requested_smoothing: str) -> str:
    if requested_smoothing == "medium" and len(sequence) < 19:
        return "no"
    return requested_smoothing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IUPred3 on each protein in a FASTA file"
    )
    parser.add_argument("fasta", type=Path, help="Multi-record FASTA input")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-protein *.caid outputs",
    )
    parser.add_argument(
        "--iupred-type",
        choices=["long", "short", "glob"],
        default="long",
        help="IUPred3 analysis type",
    )
    parser.add_argument(
        "--smoothing",
        choices=["no", "medium", "strong"],
        default="medium",
        help="IUPred3 smoothing type",
    )
    parser.add_argument("--anchor", action="store_true", help="Add ANCHOR2 scores")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute records whose output file already exists",
    )
    args = parser.parse_args()

    records = read_fasta(args.fasta)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(f"IUPred3: loaded {total} FASTA records", flush=True)

    for index, record in enumerate(records, start=1):
        protein_id = extract_uniprot_accession(record.header)
        output_path = args.output_dir / f"{safe_filename(protein_id)}.caid"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] {protein_id} skipped (exists)", flush=True)
            continue

        smoothing = effective_smoothing(record.sequence, args.smoothing)
        if smoothing != args.smoothing:
            print(
                f"[{index}/{total}] {protein_id} uses smoothing={smoothing} "
                f"because length {len(record.sequence)} is shorter than 19",
                flush=True,
            )

        iupred_result = iupred3_lib.iupred(
            record.sequence,
            args.iupred_type,
            smoothing=smoothing,
        )
        anchor_scores = iupred3_lib.anchor2(record.sequence) if args.anchor else None
        write_caid(output_path, record, iupred_result[0], anchor_scores)
        print(f"[{index}/{total}] {protein_id} written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
