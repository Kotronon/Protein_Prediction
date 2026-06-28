#!/usr/bin/env python3
"""Evaluate a trained UdonPred checkpoint on the DisProt test split.

This uses the already precomputed HuggingFace dataset in ``UdonPred/data/disprot/hf``
and therefore does not rerun ProstT5 embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score


def load_model_from_checkpoint(checkpoint_dir: Path, udonpred_dir: Path):
    sys.path.insert(0, str(udonpred_dir.resolve()))

    from model.build_model import build_prediction_heads
    from model.model import UdonPred

    with (checkpoint_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)

    hyperparameters = config.get("architecture", {})
    prediction_heads = build_prediction_heads(hyperparameters, config)
    output_keys = set(config.get("config", {}).get("outputs", ["out_-1"]))
    model = UdonPred(None, prediction_heads, output_keys)
    state = torch.load(
        checkpoint_dir / "pytorch_model.bin",
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def iter_predictions(model, dataset, smooth: float):
    head = model.prediction_heads["out"]
    with torch.no_grad():
        for record in dataset:
            embedding = torch.as_tensor(
                record["embedding_0"], dtype=torch.float32
            ).unsqueeze(0)
            scores = head(embedding).squeeze(0).squeeze(-1).cpu().numpy()
            labels = np.asarray(record["y"], dtype=np.float32)
            scores = scores[: len(labels)]
            if smooth > 0:
                scores = gaussian_filter1d(scores.astype(np.float64), sigma=smooth)
            yield str(record["id"]), labels, scores


def evaluate(model, dataset, smooth: float, output_csv: Path | None) -> dict[str, float]:
    all_labels = []
    all_scores = []
    rows = []

    for protein_id, labels, scores in iter_predictions(model, dataset, smooth):
        mask = labels != 999
        all_labels.append(labels[mask])
        all_scores.append(scores[mask])
        if output_csv is not None:
            for residue_index, label, score in zip(
                np.nonzero(mask)[0] + 1,
                labels[mask],
                scores[mask],
                strict=False,
            ):
                rows.append(
                    {
                        "protein_id": protein_id,
                        "residue_index": int(residue_index),
                        "label": int(label),
                        "score": float(score),
                    }
                )

    labels = np.concatenate(all_labels).astype(int)
    scores = np.concatenate(all_scores)

    metrics = {
        "n_residues": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "disprot_AP": float(average_precision_score(labels, scores)),
        "disprot_AUROC": float(roc_auc_score(labels, scores)),
    }

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["protein_id", "residue_index", "label", "score"]
            )
            writer.writeheader()
            writer.writerows(rows)

    return metrics


def discover_checkpoints(checkpoint_root: Path) -> list[Path]:
    return sorted(
        path
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir() and (path / "pytorch_model.bin").exists()
    )


def checkpoint_step(checkpoint_dir: Path) -> int:
    try:
        return int(checkpoint_dir.name.split("-")[-1])
    except ValueError:
        return -1


def evaluate_many(
    checkpoint_dirs: list[Path],
    udonpred_dir: Path,
    dataset,
    smooth: float,
) -> list[dict[str, float | int | str]]:
    rows = []
    for checkpoint_dir in checkpoint_dirs:
        print(f"Evaluating {checkpoint_dir} ...")
        model = load_model_from_checkpoint(checkpoint_dir, udonpred_dir)
        metrics = evaluate(model, dataset, smooth, output_csv=None)
        rows.append(
            {
                "checkpoint": str(checkpoint_dir),
                "step": checkpoint_step(checkpoint_dir),
                "smooth": smooth,
                **metrics,
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "step",
        "smooth",
        "n_residues",
        "n_positive",
        "positive_fraction",
        "disprot_AP",
        "disprot_AUROC",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "UdonPred/checkpoints_caid4/caid4_disprot_trizod_updated_focus/checkpoint-1450"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Evaluate all checkpoint-* directories under this root.",
    )
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("UdonPred/data/disprot/hf")
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/disprot_focus_checkpoint_eval/metrics.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/disprot_focus_checkpoint_eval/predictions.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/disprot_focus_checkpoint_eval/checkpoint_summary.csv"),
    )
    args = parser.parse_args()

    dataset = load_from_disk(str(args.dataset_dir))[args.split]
    if args.checkpoint_root is not None:
        checkpoint_dirs = discover_checkpoints(args.checkpoint_root)
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint-* directories found in {args.checkpoint_root}")
        rows = evaluate_many(checkpoint_dirs, args.udonpred_dir, dataset, args.smooth)
        rows.sort(key=lambda row: float(row["disprot_AP"]), reverse=True)
        write_summary_csv(rows, args.summary_csv)
        best = rows[0]
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"best": best, "rows": rows}, indent=2) + "\n")
        print(json.dumps({"best": best}, indent=2))
        print(f"Wrote {args.summary_csv}")
        print(f"Wrote {args.output_json}")
        return

    model = load_model_from_checkpoint(args.checkpoint_dir, args.udonpred_dir)
    metrics = evaluate(model, dataset, args.smooth, args.output_csv)
    metrics.update(
        {
            "checkpoint_dir": str(args.checkpoint_dir),
            "split": args.split,
            "smooth": args.smooth,
        }
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
