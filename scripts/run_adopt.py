#!/usr/bin/env python3
"""Run ADOPT on a FASTA file and write CAID-like outputs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ADOPT_DIR = REPO_ROOT / "ADOPT"

ESM_MODEL_NAMES = {
    "esm-1b": "esm1b_t33_650M_UR50S",
    "esm-1v": "esm1v_t33_650M_UR90S_1",
}

ADOPT_STRATEGY_NAMES = {
    "train_on_cleared_1325_test_on_117_residue_split": "cleared_residue",
    "train_on_1325_cv_residue_split": "residue_cv",
    "train_on_cleared_1325_cv_residue_split": "cleared_residue_cv",
    "train_on_cleared_1325_cv_sequence_split": "cleared_sequence_cv",
    "train_on_total": "total_cleared_residue",
}


@dataclass(frozen=True)
class FastaRecord:
    header: str
    label: str
    sequence: str


def fasta_label(header: str) -> str:
    return header.strip().split()[0]


def extract_uniprot_accession(header: str) -> str:
    first_token = header.lstrip(">").strip().split()[0]
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def ensure_adopt_models() -> None:
    models_dir = ADOPT_DIR / "models"
    if any(models_dir.glob("*.onnx")):
        return

    models_zip = ADOPT_DIR / "models.zip"
    if not models_zip.is_file():
        raise FileNotFoundError(
            f"ADOPT models not found. Expected either {models_dir}/*.onnx or {models_zip}"
        )

    print(f"ADOPT: extracting local models from {models_zip}", flush=True)
    with zipfile.ZipFile(models_zip) as archive:
        archive.extractall(ADOPT_DIR)


def read_fasta_records(path: Path) -> list[FastaRecord]:
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
        records.append(
            FastaRecord(header=header, label=fasta_label(header), sequence=sequence)
        )
        header = None
        chunks = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
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


def write_fasta(path: Path, records: list[FastaRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.label}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(f"{record.sequence[start:start + 80]}\n")


def normalize_representation_filenames(repr_dir: Path) -> None:
    if not repr_dir.is_dir():
        return

    for path in repr_dir.glob("*.pt"):
        normalized_label = fasta_label(path.stem)
        if normalized_label == path.stem:
            continue
        normalized_path = path.with_name(f"{normalized_label}.pt")
        if normalized_path.exists():
            path.unlink()
        else:
            path.rename(normalized_path)


def existing_representation_labels(repr_dir: Path) -> set[str]:
    if not repr_dir.is_dir():
        return set()
    return {path.stem for path in repr_dir.glob("*.pt")}


def predict_adopt_scores(
    records: list[FastaRecord],
    repr_dir: Path,
    json_path: Path,
    model_type: str,
    train_strategy: str,
) -> None:
    import numpy as np
    import onnxruntime as rt
    import torch

    strategy_name = ADOPT_STRATEGY_NAMES[train_strategy]
    model_path = ADOPT_DIR / "models" / f"lasso_{model_type}_{strategy_name}.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"ADOPT ONNX model not found: {model_path}")

    print(f"ADOPT: running ONNX inference with {model_path}", flush=True)
    session = rt.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    output_records = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        repr_path = repr_dir / model_type / f"{record.label}.pt"
        if not repr_path.is_file():
            raise FileNotFoundError(f"ADOPT representation not found: {repr_path}")

        representation = (
            torch.load(repr_path, map_location="cpu")["representations"][33]
            .clone()
            .cpu()
            .detach()
            .numpy()
        )
        predictions = session.run([output_name], {input_name: representation})[0]
        scores = np.asarray(predictions).reshape(-1).astype(float).tolist()
        output_records.append(
            {
                "brmid": record.label,
                "sequence": record.sequence,
                "z_scores": scores,
            }
        )
        print(f"ADOPT inference [{index}/{total}] {record.label}", flush=True)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(output_records, handle)


def write_caid_outputs(json_path: Path, output_dir: Path) -> None:
    with json_path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(records)

    for index, record in enumerate(records, start=1):
        header = record["brmid"]
        sequence = record["sequence"]
        scores = record["z_scores"]
        length = min(len(sequence), len(scores))
        if len(sequence) != len(scores):
            print(
                f"[{index}/{total}] {header} length mismatch: "
                f"{len(sequence)} residues, {len(scores)} scores; writing first {length}",
                flush=True,
            )

        protein_id = extract_uniprot_accession(header)
        output_path = output_dir / f"{safe_filename(protein_id)}.caid"
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(f">{header}\n")
            for pos, residue in enumerate(sequence[:length], start=1):
                handle.write(f"{pos}\t{residue}\t{float(scores[pos - 1]):.6f}\n")
        print(f"[{index}/{total}] {protein_id} written to {output_path}", flush=True)


def run_command(command: list[str], cwd: Path) -> None:
    print(f"ADOPT: running {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ADOPT on a FASTA file")
    parser.add_argument("fasta", type=Path, help="Multi-record FASTA input")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-protein *.caid outputs",
    )
    parser.add_argument(
        "--repr-dir",
        type=Path,
        required=True,
        help="Directory for cached ADOPT/ESM residue representations",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
        help="Path for ADOPT's intermediate JSON output",
    )
    parser.add_argument(
        "--model-type",
        choices=["esm-1b", "esm-1v"],
        default="esm-1b",
        help="ADOPT ESM model type. esm-1b is the practical default.",
    )
    parser.add_argument(
        "--train-strategy",
        choices=list(ADOPT_STRATEGY_NAMES),
        default="train_on_total",
        help="ADOPT training strategy",
    )
    parser.add_argument(
        "--toks-per-batch",
        type=int,
        default=4096,
        help="Maximum ESM tokens per batch",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1022 tokens for ESM extraction",
    )
    parser.add_argument("--nogpu", action="store_true", help="Force ESM extraction on CPU")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute ADOPT JSON even if it already exists",
    )
    args = parser.parse_args()

    records = read_fasta_records(args.fasta)
    total = len(records)
    print(f"ADOPT: loaded {total} FASTA records", flush=True)

    max_sequence_length = max(len(record.sequence) for record in records)
    if max_sequence_length > 1022 and not args.truncate:
        args.truncate = True
        print(
            "ADOPT: enabling --truncate automatically because the longest "
            f"sequence has {max_sequence_length} residues and ESM-1 supports "
            "at most 1022 residues per sequence",
            flush=True,
        )

    ensure_adopt_models()
    args.repr_dir.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    model_repr_dir = args.repr_dir / args.model_type
    model_repr_dir.mkdir(parents=True, exist_ok=True)
    normalize_representation_filenames(model_repr_dir)
    existing_labels = existing_representation_labels(model_repr_dir)
    missing_records = [
        record
        for record in records
        if args.overwrite or record.label not in existing_labels
    ]

    if not missing_records:
        print(
            f"ADOPT: using cached {len(existing_labels)} representations in {model_repr_dir}",
            flush=True,
        )
    else:
        extract_fasta = args.json_output.parent / f"adopt_missing_{args.model_type}.fasta"
        write_fasta(extract_fasta, missing_records)
        print(
            f"ADOPT: extracting {len(missing_records)} missing representations "
            f"({total - len(missing_records)} cached)",
            flush=True,
        )
        extract_command = [
            sys.executable,
            "-u",
            "esm/scripts/extract.py",
            ESM_MODEL_NAMES[args.model_type],
            str(extract_fasta),
            str(model_repr_dir),
            "--repr_layers",
            "33",
            "--include",
            "per_tok",
            "--toks_per_batch",
            str(args.toks_per_batch),
        ]
        if args.truncate:
            extract_command.append("--truncate")
        if args.nogpu:
            extract_command.append("--nogpu")
        run_command(extract_command, ADOPT_DIR)

    if args.json_output.exists() and not args.overwrite:
        print(f"ADOPT: using cached JSON output {args.json_output}", flush=True)
    else:
        predict_adopt_scores(
            records,
            args.repr_dir,
            args.json_output,
            args.model_type,
            args.train_strategy,
        )

    write_caid_outputs(args.json_output, args.output_dir)


if __name__ == "__main__":
    main()
