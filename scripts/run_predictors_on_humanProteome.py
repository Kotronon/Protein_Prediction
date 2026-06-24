"""Run UdonPred's per-dataset models on the human proteome.
Also use External Predictors to compare agaibst other predictors.

For external predictors, we can use: SETH, metapredict, ADOPT, maybe IUPred3.
SETH: https://github.com/DagmarIlz/SETH
ADOPT: https://github.com/PeptoneLtd/ADOPT
metapredict: https://github.com/idptools/metapredict
IUPred3: requires form to sen request (https://iupred3.elte.hu/download_new)

python scripts/run_predictors_on_humanProteome.py \
  --predictors UdonPred \
  --human_proteome_fasta HumanProteome/human_preteome.fasta \
  --output_dir results/human_proteome \
  --udonpred_targets trizod chezod softdis pdbflex atlas plddt disprot \
  --device cpu \
  --batch-size 512 \
  --max-sequence-length 512 \
  --chunk-overlap 64

"""

import argparse
import shlex
import subprocess
from pathlib import Path

UDONPRED_TARGETS = [
    "trizod",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot",
]


def get_predictor_commands(args, repo_root):
    commands = []
    udonpred_dir = repo_root / "UdonPred"
    weights_dir = udonpred_dir / "weights"

    for predictor in args.predictors:
        if predictor == "UdonPred":
            for target in args.udonpred_targets:
                commands.append(
                    {
                        "cmd": [
                            "uv",
                            "run",
                            "python",
                            "predict.py",
                            str(args.human_proteome_fasta),
                            str(weights_dir),
                            "--target",
                            target,
                            "--output",
                            str(args.output_dir / "UdonPred" / target),
                            "--batch-size",
                            str(args.batch_size),
                            "--device",
                            args.device,
                            "--smooth",
                            str(args.smooth),
                            "--max-sequence-length",
                            str(args.max_sequence_length),
                            "--chunk-overlap",
                            str(args.chunk_overlap),
                            "--embedding-cache-dir",
                            str(args.embedding_cache_dir),
                        ],
                        "cwd": udonpred_dir,
                    }
                )
        elif predictor == "IUPred3":
            commands.append(
                {
                    "cmd": [
                        "python",
                        "-u",
                        str(repo_root / "scripts" / "run_iupred3.py"),
                        str(args.human_proteome_fasta),
                        "--output-dir",
                        str(args.output_dir / "IUPred3"),
                        "--iupred-type",
                        args.iupred_type,
                        "--smoothing",
                        args.iupred_smoothing,
                    ]
                    + (["--anchor"] if args.iupred_anchor else [])
                    + (["--overwrite"] if args.overwrite else []),
                    "cwd": repo_root,
                }
            )
        elif predictor == "ADOPT":
            commands.append(
                {
                    "cmd": [
                        "python",
                        "-u",
                        str(repo_root / "scripts" / "run_adopt.py"),
                        str(args.human_proteome_fasta),
                        "--output-dir",
                        str(args.output_dir / "ADOPT"),
                        "--repr-dir",
                        str(args.adopt_repr_dir),
                        "--json-output",
                        str(args.output_dir / "ADOPT" / "adopt_predictions.json"),
                        "--model-type",
                        args.adopt_model_type,
                        "--train-strategy",
                        args.adopt_train_strategy,
                        "--toks-per-batch",
                        str(args.adopt_toks_per_batch),
                    ]
                    + (["--truncate"] if args.adopt_truncate else [])
                    + (["--nogpu"] if args.adopt_nogpu else [])
                    + (["--overwrite"] if args.overwrite else []),
                    "cwd": repo_root,
                }
            )
        elif predictor == "SETH":
            pass  # TODO: add command for SETH and add SETH
        elif predictor == "flDPnn3":
            pass  # TODO: add command for flDPnn3 and add flDPnn3
        elif predictor == "metapredict":
            pass  # TODO: add command for metapredict and add metapredict
        else:
            raise ValueError(f"Unknown predictor: {predictor}")
    return commands


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run predictors on the human proteome")
    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="List of predictors to run. Currently supported: UdonPred, IUPred3, ADOPT",
    )
    parser.add_argument(
        "--human_proteome_fasta",
        type=Path,
        required=True,
        help="Path to the human proteome FASTA file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the prediction results",
    )
    parser.add_argument(
        "--udonpred_targets",
        nargs="+",
        default=UDONPRED_TARGETS,
        choices=UDONPRED_TARGETS,
        help="UdonPred model targets to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Maximum total sequence length per UdonPred batch",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for UdonPred inference",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for UdonPred predictions",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Split longer proteins into windows of this many residues",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Residue overlap for split UdonPred windows",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Directory for cached ProstT5 embeddings",
    )
    parser.add_argument(
        "--iupred-type",
        choices=["long", "short", "glob"],
        default="long",
        help="IUPred3 analysis type",
    )
    parser.add_argument(
        "--iupred-smoothing",
        choices=["no", "medium", "strong"],
        default="medium",
        help="IUPred3 smoothing type",
    )
    parser.add_argument(
        "--iupred-anchor",
        action="store_true",
        help="Also write ANCHOR2 scores for IUPred3 outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute predictor outputs that already exist when supported",
    )
    parser.add_argument(
        "--adopt-model-type",
        choices=["esm-1b", "esm-1v"],
        default="esm-1b",
        help="ADOPT model type",
    )
    parser.add_argument(
        "--adopt-train-strategy",
        default="train_on_total",
        help="ADOPT training strategy",
    )
    parser.add_argument(
        "--adopt-toks-per-batch",
        type=int,
        default=4096,
        help="Maximum ESM tokens per ADOPT embedding batch",
    )
    parser.add_argument(
        "--adopt-repr-dir",
        type=Path,
        default=None,
        help="Directory for cached ADOPT/ESM residue representations",
    )
    parser.add_argument(
        "--adopt-truncate",
        action="store_true",
        help="Truncate sequences longer than 1022 tokens during ADOPT embedding",
    )
    parser.add_argument(
        "--adopt-nogpu",
        action="store_true",
        help="Force ADOPT ESM embedding extraction on CPU",
    )
    args = parser.parse_args()

    if not args.human_proteome_fasta.is_absolute():
        args.human_proteome_fasta = repo_root / args.human_proteome_fasta
    if not args.output_dir.is_absolute():
        args.output_dir = repo_root / args.output_dir
    if args.embedding_cache_dir is None:
        args.embedding_cache_dir = args.output_dir / "features" / "prostt5_udonpred"
    elif not args.embedding_cache_dir.is_absolute():
        args.embedding_cache_dir = repo_root / args.embedding_cache_dir
    if args.adopt_repr_dir is None:
        args.adopt_repr_dir = args.output_dir / "features" / "adopt_repr"
    elif not args.adopt_repr_dir.is_absolute():
        args.adopt_repr_dir = repo_root / args.adopt_repr_dir

    if not args.human_proteome_fasta.is_file():
        raise FileNotFoundError(
            f"Human proteome FASTA not found: {args.human_proteome_fasta}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    commands = get_predictor_commands(args, repo_root)

    for command in commands:
        print(
            f"Running command in {command['cwd']}: {shlex.join(command['cmd'])}",
            flush=True,
        )
        subprocess.run(command["cmd"], cwd=command["cwd"], check=True)


if __name__ == "__main__":
    main()
