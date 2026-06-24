#!/usr/bin/env python3
"""Run ESMDisPred Docker inference over FASTA chunks sequentially.

The runner reuses ESMDisPred's host-side caches and skips chunks whose final
ESMDisPred-DNN CAID files already exist in the shared output directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FastaRecord:
    header: str
    sequence: str
    source_chunk: str

    @property
    def protein_id(self) -> str:
        return self.header.split()[0]


def read_fasta_records(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        records.append(
            FastaRecord(
                header=header,
                sequence="".join(sequence_parts),
                source_chunk=path.name,
            )
        )
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


def read_fasta_ids(path: Path) -> list[str]:
    return [record.protein_id for record in read_fasta_records(path)]


def caid_exists(protein_id: str, output_dir: Path) -> bool:
    caid_dir = output_dir / "disorder" / "ESMDisPred-DNN"
    return (caid_dir / f"{protein_id}.caid").is_file()


def missing_outputs(chunk_fasta: Path, output_dir: Path) -> list[str]:
    return [
        protein_id
        for protein_id in read_fasta_ids(chunk_fasta)
        if not caid_exists(protein_id, output_dir)
    ]


def write_fasta_records(output_fasta: Path, records: list[FastaRecord]) -> None:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.header}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(f"{record.sequence[start:start + 80]}\n")


def write_filtered_fasta(input_fasta: Path, output_fasta: Path, keep_ids: set[str]) -> None:
    """Write only FASTA records whose first header token is in keep_ids."""
    written_ids: set[str] = set()
    keep_record = False

    with input_fasta.open(encoding="utf-8") as source, output_fasta.open(
        "w",
        encoding="utf-8",
    ) as target:
        for raw_line in source:
            if raw_line.startswith(">"):
                protein_id = raw_line[1:].split()[0]
                keep_record = protein_id in keep_ids
                if keep_record:
                    written_ids.add(protein_id)
                    target.write(raw_line)
                continue

            if keep_record:
                target.write(raw_line)

    missing_ids = keep_ids - written_ids
    if missing_ids:
        preview = ", ".join(sorted(missing_ids)[:5])
        raise ValueError(
            f"{input_fasta}: could not find {len(missing_ids)} requested protein(s): {preview}"
        )


def build_record_batches(
    records: list[FastaRecord],
    records_per_run: int,
    max_residues_per_run: int | None,
) -> list[list[FastaRecord]]:
    if records_per_run < 1:
        raise ValueError("--missing-records-per-run must be >= 1")
    if max_residues_per_run is not None and max_residues_per_run < 1:
        raise ValueError("--missing-residues-per-run must be >= 1")

    batches: list[list[FastaRecord]] = []
    current: list[FastaRecord] = []
    current_residues = 0

    for record in records:
        record_len = len(record.sequence)
        would_exceed_records = len(current) >= records_per_run
        would_exceed_residues = (
            max_residues_per_run is not None
            and current
            and current_residues + record_len > max_residues_per_run
        )
        if would_exceed_records or would_exceed_residues:
            batches.append(current)
            current = []
            current_residues = 0

        current.append(record)
        current_residues += record_len

    if current:
        batches.append(current)
    return batches


def run_command(
    command: list[str],
    esmdispred_dir: Path,
    env: dict[str, str],
    dry_run: bool,
) -> int | None:
    print("  " + " ".join(command), flush=True)
    if dry_run:
        return None
    try:
        subprocess.run(command, cwd=esmdispred_dir, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chunks_dir",
        type=Path,
        help="Directory containing chunk FASTA files, e.g. HumanProteome/esmdispred_chunks.",
    )
    parser.add_argument(
        "--esmdispred-dir",
        type=Path,
        default=Path("ESMDisPred"),
        help="Path to the ESMDisPred checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_proteome/ESMDisPred"),
        help="Shared ESMDisPred output directory.",
    )
    parser.add_argument("--pattern", default="*.fasta", help="Chunk filename glob.")
    parser.add_argument("--model", default="4", help="ESMDisPred model option.")
    parser.add_argument("--docker-cpus", default="6")
    parser.add_argument("--fldpnn-nproc", default="2")
    parser.add_argument(
        "--start-at",
        default=None,
        help="Optional chunk filename to start from, e.g. esmdispred_0012.fasta.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running Docker.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later chunks when one chunk fails.",
    )
    parser.add_argument(
        "--failed-chunks-csv",
        type=Path,
        default=None,
        help="CSV report for failed chunks. Defaults to <chunks-dir>/failed_chunks.csv.",
    )
    parser.add_argument(
        "--batch-missing",
        action="store_true",
        help="Collect all missing proteins first and run them in fewer combined rest batches.",
    )
    parser.add_argument(
        "--missing-records-per-run",
        type=int,
        default=10,
        help="Maximum records per combined missing batch when --batch-missing is used.",
    )
    parser.add_argument(
        "--missing-residues-per-run",
        type=int,
        default=10000,
        help="Maximum residues per combined missing batch when --batch-missing is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_dir = args.chunks_dir.resolve()
    esmdispred_dir = args.esmdispred_dir.resolve()
    output_dir = args.output_dir.resolve()
    docker_script = esmdispred_dir / "run_ESMDisPred_Docker.sh"

    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    if not docker_script.is_file():
        raise FileNotFoundError(f"Docker wrapper not found: {docker_script}")

    chunks = sorted(chunks_dir.glob(args.pattern))
    if not chunks:
        raise ValueError(f"No chunk files matching {args.pattern!r} in {chunks_dir}")

    if args.start_at:
        start_path = chunks_dir / args.start_at
        chunks = [path for path in chunks if path >= start_path]
        if not chunks:
            raise ValueError(f"No chunks found at or after {start_path.name}")

    env = os.environ.copy()
    env["DOCKER_CPUS"] = str(args.docker_cpus)
    env["FLDPNN_NPROC"] = str(args.fldpnn_nproc)
    failed_rows: list[dict[str, object]] = []
    filtered_chunks_dir = chunks_dir / ".missing_for_esmdispred"

    if args.batch_missing:
        missing_records: list[FastaRecord] = []
        total_records = 0
        for chunk in chunks:
            records = read_fasta_records(chunk)
            total_records += len(records)
            missing_records.extend(
                record for record in records if not caid_exists(record.protein_id, output_dir)
            )

        if not missing_records:
            print(f"All outputs exist for {total_records} records.", flush=True)
            return

        batches = build_record_batches(
            missing_records,
            args.missing_records_per_run,
            args.missing_residues_per_run,
        )
        print(
            f"Collected {len(missing_records)}/{total_records} missing output(s); "
            f"running {len(batches)} combined batch(es).",
            flush=True,
        )

        for batch_index, batch_records in enumerate(batches, start=1):
            batch_fasta = filtered_chunks_dir / f"missing_batch_{batch_index:04d}.fasta"
            if not args.dry_run:
                write_fasta_records(batch_fasta, batch_records)
            residues = sum(len(record.sequence) for record in batch_records)
            source_chunks = sorted({record.source_chunk for record in batch_records})
            print(
                f"[{batch_index}/{len(batches)}] run {batch_fasta.name}: "
                f"{len(batch_records)} protein(s), {residues} residues, "
                f"from {len(source_chunks)} chunk(s)",
                flush=True,
            )
            command = [
                str(docker_script),
                str(batch_fasta),
                str(output_dir),
                str(args.model),
            ]
            returncode = run_command(command, esmdispred_dir, env, args.dry_run)
            if returncode is not None:
                failed_rows.append(
                    {
                        "chunk": batch_fasta.name,
                        "returncode": returncode,
                        "missing_outputs_before_run": len(batch_records),
                    }
                )
                print(
                    f"[{batch_index}/{len(batches)}] failed {batch_fasta.name}: "
                    f"exit {returncode}",
                    flush=True,
                )
                if not args.keep_going:
                    raise SystemExit(returncode)
        chunks = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_ids = read_fasta_ids(chunk)
        missing = missing_outputs(chunk, output_dir)
        if not missing:
            print(f"[{index}/{len(chunks)}] skip {chunk.name}: all outputs exist", flush=True)
            continue

        input_fasta = chunk
        if len(missing) < len(chunk_ids):
            filtered_chunks_dir.mkdir(exist_ok=True)
            input_fasta = filtered_chunks_dir / f"{chunk.stem}.missing.fasta"
            write_filtered_fasta(chunk, input_fasta, set(missing))

        print(
            f"[{index}/{len(chunks)}] run {chunk.name}: "
            f"{len(missing)}/{len(chunk_ids)} missing output(s)",
            flush=True,
        )
        command = [
            str(docker_script),
            str(input_fasta),
            str(output_dir),
            str(args.model),
        ]
        returncode = run_command(command, esmdispred_dir, env, args.dry_run)
        if returncode is not None:
            failed_rows.append(
                {
                    "chunk": chunk.name,
                    "returncode": returncode,
                    "missing_outputs_before_run": len(missing),
                }
            )
            print(
                f"[{index}/{len(chunks)}] failed {chunk.name}: "
                f"exit {returncode}",
                flush=True,
            )
            if not args.keep_going:
                raise SystemExit(returncode)
            continue

    if failed_rows:
        failed_path = args.failed_chunks_csv or (chunks_dir / "failed_chunks.csv")
        with failed_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["chunk", "returncode", "missing_outputs_before_run"],
            )
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"Wrote failed chunk report: {failed_path}", flush=True)


if __name__ == "__main__":
    main()
