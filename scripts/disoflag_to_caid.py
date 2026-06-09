#!/usr/bin/env python3
"""Convert DisoFLAG propensity output to per-protein CAID files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SCORE_TYPES = {
    "idr": 0,
    "protein_binding": 1,
    "dna_binding": 2,
    "rna_binding": 3,
    "ion_binding": 4,
    "lipid_binding": 5,
    "linker": 6,
}


def extract_protein_id(header: str) -> str:
    first_token = header.lstrip(">").strip().split()[0]
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "protein"


def parse_scores(line: str, path: Path, line_number: int) -> list[float]:
    try:
        return [float(value) for value in line.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid comma-separated score row"
        ) from exc


def convert(input_path: Path, output_dir: Path, score_type: str) -> int:
    lines = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    score_index = SCORE_TYPES[score_type]
    offset = 0
    written = 0
    seen_ids: set[str] = set()

    while offset < len(lines):
        header = lines[offset]
        if not header.startswith(">"):
            raise ValueError(
                f"{input_path}:{offset + 1}: expected FASTA header, got {header!r}"
            )
        if offset + 8 >= len(lines):
            raise ValueError(
                f"{input_path}:{offset + 1}: incomplete DisoFLAG record"
            )

        sequence = lines[offset + 1]
        score_lines = lines[offset + 2 : offset + 9]
        scores = parse_scores(
            score_lines[score_index],
            input_path,
            offset + 3 + score_index,
        )

        protein_id = extract_protein_id(header)
        if protein_id in seen_ids:
            raise ValueError(f"{input_path}: duplicate protein ID {protein_id!r}")
        seen_ids.add(protein_id)

        if len(scores) != len(sequence):
            raise ValueError(
                f"{input_path}: {protein_id} has {len(sequence)} residues but "
                f"{len(scores)} {score_type} scores"
            )

        output_path = output_dir / f"{safe_filename(protein_id)}.caid"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(f"{header}\n")
            for position, (residue, score) in enumerate(
                zip(sequence, scores), start=1
            ):
                handle.write(f"{position}\t{residue}\t{score:.6f}\n")

        written += 1
        offset += 9

    print(
        f"Wrote {written} {score_type} CAID files to {output_dir}",
        flush=True,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DisoFLAG propensity output to per-protein CAID files."
    )
    parser.add_argument("input", type=Path, help="DisoFLAG propensity TXT file")
    parser.add_argument("output_dir", type=Path, help="Directory for *.caid files")
    parser.add_argument(
        "--score-type",
        choices=sorted(SCORE_TYPES),
        default="idr",
        help="DisoFLAG score row to convert (default: idr)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    convert(args.input, args.output_dir, args.score_type)


if __name__ == "__main__":
    main()
