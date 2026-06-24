#!/usr/bin/env python3
"""Precompute ProstT5 embeddings for later UdonPred runs.
    python scripts/precompute_embeddings.py \
  --human-proteome-fasta HumanProteome/human_preteome.fasta \
  --device cpu \
  --batch-size 512 \
  --max-sequence-length 512 \
  --chunk-overlap 64

"""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    udonpred_dir = repo_root / "UdonPred"
    weights_dir = udonpred_dir / "weights"

    parser = argparse.ArgumentParser(
        description="Precompute UdonPred/ProstT5 embeddings for a FASTA file."
    )
    parser.add_argument(
        "--human-proteome-fasta",
        type=Path,
        required=True,
        help="Path to the FASTA file containing the human proteome.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("results/human_proteome/features/prostt5_udonpred"),
        help="Directory where ProstT5 embedding chunks will be cached.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Maximum total sequence length per batch.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for embedding computation.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Split longer proteins into windows of this many residues.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Residue overlap between adjacent windows.",
    )
    args = parser.parse_args()

    if not args.human_proteome_fasta.is_absolute():
        args.human_proteome_fasta = repo_root / args.human_proteome_fasta
    if not args.embedding_cache_dir.is_absolute():
        args.embedding_cache_dir = repo_root / args.embedding_cache_dir

    if not args.human_proteome_fasta.is_file():
        raise FileNotFoundError(
            f"Human proteome FASTA not found: {args.human_proteome_fasta}"
        )

    args.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "run",
        "python",
        "predict.py",
        str(args.human_proteome_fasta),
        str(weights_dir),
        "--target",
        "trizod",
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--max-sequence-length",
        str(args.max_sequence_length),
        "--chunk-overlap",
        str(args.chunk_overlap),
        "--embedding-cache-dir",
        str(args.embedding_cache_dir),
        "--precompute-only",
    ]

    print(f"Running command in {udonpred_dir}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=udonpred_dir, check=True)


if __name__ == "__main__":
    main()
