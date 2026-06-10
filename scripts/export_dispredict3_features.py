#!/usr/bin/env python3
"""Export completed flDPnn features from Dispredict3 Docker workers."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from Bio import SeqIO


FEATURE_SUFFIXES = (".ttscore", ".ttpreds", ".ttindex")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", name],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def expected_sequences(fasta_path: Path) -> dict[str, str]:
    return {
        record.id: str(record.seq).upper()
        for record in SeqIO.parse(fasta_path, "fasta")
    }


def feature_ids(feature_dir: Path, suffix: str) -> set[str]:
    return {path.name.removesuffix(suffix) for path in feature_dir.glob(f"*{suffix}")}


def remote_feature_ids(container: str, remote_dir: str, suffix: str) -> set[str]:
    output = run(
        "docker",
        "exec",
        container,
        "bash",
        "-lc",
        f'find "{remote_dir}" -maxdepth 1 -type f -name "*{suffix}" -printf "%f\\n"',
    ).stdout
    return {
        name.removesuffix(suffix)
        for name in output.splitlines()
        if name.endswith(suffix)
    }


def fldpnn_is_running(container: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "exec",
            container,
            "bash",
            "-lc",
            "ps -eo args | grep -q '[r]un_flDPnn.py'",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def residue_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def update_skipped_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = {
                row["protein_id"]: row
                for row in csv.DictReader(handle)
            }
    existing.update({row["protein_id"]: row for row in rows})

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("protein_id", "length", "reason"),
        )
        writer.writeheader()
        writer.writerows(existing[protein_id] for protein_id in sorted(existing))
    temporary.replace(path)


def export_worker(
    worker: int,
    input_dir: Path,
    output_dir: Path,
    keep_staging: bool,
    wait: bool,
    poll_seconds: int,
    stop_container: bool,
) -> None:
    container = f"dispredict_{worker}"
    fasta_path = input_dir / f"processedinput_{worker}.fasta"
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Worker FASTA not found: {fasta_path}")
    if not container_exists(container):
        raise RuntimeError(f"Container not found: {container}")

    sequences = expected_sequences(fasta_path)
    expected = set(sequences)
    remote_dir = "/opt/Dispredict3.0/tools/fldpnn/output"
    while True:
        missing_by_suffix = {
            suffix: expected - remote_feature_ids(container, remote_dir, suffix)
            for suffix in FEATURE_SUFFIXES
        }
        missing_by_suffix = {
            suffix: missing
            for suffix, missing in missing_by_suffix.items()
            if missing
        }
        still_running = fldpnn_is_running(container)
        if not missing_by_suffix and not still_running:
            break
        details = ", ".join(
            f"{suffix}: {len(missing)} missing"
            for suffix, missing in missing_by_suffix.items()
        )
        if still_running:
            details = f"{details}, flDPnn still running" if details else "flDPnn still running"
        if not wait:
            raise RuntimeError(
                f"{container}: flDPnn is not ready ({details})"
            )
        print(
            f"{container}: waiting for flDPnn ({details})",
            flush=True,
        )
        time.sleep(poll_seconds)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = output_dir.parent / ".dispredict3_export"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"worker_{worker}_", dir=staging_parent))
    try:
        subprocess.run(
            ["docker", "cp", f"{container}:{remote_dir}/.", str(staging)],
            check=True,
        )
        for suffix in FEATURE_SUFFIXES:
            actual = feature_ids(staging, suffix)
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"{container}: exported {suffix} files are missing "
                    f"{len(missing)} proteins"
                )

        valid_ids: set[str] = set()
        skipped_rows: list[dict[str, str]] = []
        for protein_id, sequence in sequences.items():
            row_counts = {
                suffix: residue_line_count(staging / f"{protein_id}{suffix}")
                for suffix in FEATURE_SUFFIXES
            }
            if all(rows == len(sequence) for rows in row_counts.values()):
                valid_ids.add(protein_id)
                continue
            if all(rows == 0 for rows in row_counts.values()):
                skipped_rows.append(
                    {
                        "protein_id": protein_id,
                        "length": str(len(sequence)),
                        "reason": "flDPnn produced empty feature files",
                    }
                )
                continue
            details = ", ".join(
                f"{suffix}={rows}" for suffix, rows in row_counts.items()
            )
            raise RuntimeError(
                f"{container}: incomplete features for {protein_id} "
                f"({details}, expected {len(sequence)} rows each)"
            )

        copied = 0
        for protein_id in valid_ids:
            for suffix in FEATURE_SUFFIXES:
                source = staging / f"{protein_id}{suffix}"
                destination = output_dir / source.name
                source.replace(destination)
                copied += 1
        if skipped_rows:
            update_skipped_manifest(
                output_dir.parent / "skipped_flDPnn.csv",
                skipped_rows,
            )
            print(
                f"{container}: skipped {len(skipped_rows)} proteins with empty "
                f"flDPnn output"
            )
        print(
            f"{container}: exported {copied} feature files for "
            f"{len(valid_ids)} proteins"
        )
        if stop_container:
            subprocess.run(["docker", "rm", "-f", container], check=True)
            print(f"{container}: stopped after successful export")
    finally:
        if staging.exists() and not keep_staging:
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--worker-ids",
        type=int,
        nargs="+",
        help="Export only these worker numbers, for example: --worker-ids 2 3",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Dispredict3.0/ParallelDispredict3.0/temp/Parallelinputs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_proteome/Dispredict3_native/features"),
    )
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stop-containers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be positive")
    workers = args.worker_ids or list(range(1, args.workers + 1))
    for worker in workers:
        export_worker(
            worker,
            args.input_dir,
            args.output_dir,
            args.keep_staging,
            args.wait,
            args.poll_seconds,
            args.stop_containers,
        )


if __name__ == "__main__":
    main()
