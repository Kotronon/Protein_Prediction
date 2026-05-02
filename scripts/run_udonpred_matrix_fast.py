#!/usr/bin/env python3
"""Run UdonPred's 7x7 matrix with one backbone pass per test dataset.

The original matrix runner shells out to ``UdonPred/predict.py`` for every
train/test pair. That is simple, but it recomputes the ProstT5 embeddings 49
times. This runner keeps the same output layout while computing embeddings once
per test dataset and applying all ONNX heads to the shared embeddings.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from torchmetrics.functional import auroc, average_precision, spearman_corrcoef
from tqdm import tqdm


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
NEGATED_DATASETS = {"plddt", "chezod"}


def load_predict_module(udonpred_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("udonpred_predict", udonpred_dir / "predict.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {udonpred_dir / 'predict.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_complete_prediction_dir(result_dir: Path, expected_count: int) -> bool:
    return result_dir.exists() and len(list(result_dir.glob("*.caid"))) >= expected_count


def write_prediction_file(out_dir: Path, header: str, sequence: str, scores: np.ndarray, predict_module: Any) -> None:
    safe_header = header.replace("/", "_").replace("|", "_")
    file_path = out_dir / f"{safe_header}.caid"
    file_path.write_text("".join(predict_module.format_predictions(header, sequence, scores)))


def run_predictions_fast(
    udonpred_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    smooth: float,
    force: bool,
) -> None:
    predict_module = load_predict_module(udonpred_dir)
    torch_device = predict_module.resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"Loading tokenizer ({predict_module.BACKBONE_NAME}) ...")
    tokenizer = predict_module.load_tokenizer()

    print(f"Loading backbone ({predict_module.BACKBONE_NAME}) ...")
    backbone = predict_module.load_backbone(torch_device, torch_dtype)

    weights_dir = udonpred_dir / "weights"
    heads = {
        train_set: predict_module.load_head(weights_dir / f"{train_set}.onnx", torch_device)
        for train_set in DATASETS
    }
    head_io = {
        train_set: (head.get_inputs()[0].name, head.get_outputs()[0].name)
        for train_set, head in heads.items()
    }

    prefix_token_len = 1

    for test_set in DATASETS:
        entries = predict_module.read_fasta(str(udonpred_dir / "data" / test_set / "test.fasta"))
        entries = sorted(entries, key=lambda item: len(item[1]), reverse=True)
        expected_count = len(entries)

        active_train_sets = []
        for train_set in DATASETS:
            result_dir = output_dir / "predictions" / f"{train_set}_{test_set}"
            if is_complete_prediction_dir(result_dir, expected_count) and not force:
                print(f"Skipping complete predictions: {result_dir}")
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            active_train_sets.append(train_set)

        if not active_train_sets:
            continue

        print(f"Predicting {test_set} with heads: {', '.join(active_train_sets)}")
        total_batches = predict_module.count_batches(entries, batch_size)
        with torch.inference_mode():
            for batch in tqdm(predict_module.iter_batches(entries, batch_size), total=total_batches):
                headers, seqs = zip(*batch)
                seqs = list(seqs)
                max_seq_len = max(len(seq) for seq in seqs)

                input_ids, attention_mask = predict_module.tokenize_batch(tokenizer, seqs, torch_device)
                hidden = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
                hidden = hidden * attention_mask.unsqueeze(-1)
                emb = hidden[:, prefix_token_len : prefix_token_len + max_seq_len, :]
                emb_np = emb.float().cpu().numpy()

                for train_set in active_train_sets:
                    head = heads[train_set]
                    head_input_name, head_output_name = head_io[train_set]
                    scores_batch = head.run([head_output_name], {head_input_name: emb_np})[0]
                    result_dir = output_dir / "predictions" / f"{train_set}_{test_set}"

                    for header, seq, scores in zip(headers, seqs, scores_batch):
                        scores_seq = scores[: len(seq)]
                        if smooth > 0:
                            scores_seq = gaussian_filter1d(
                                scores_seq.astype(np.float64), sigma=smooth, axis=0
                            )
                        write_prediction_file(result_dir, header, seq, scores_seq, predict_module)


def read_labels(path: Path) -> dict[str, list[float]]:
    labels = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            labels[str(record["id"])] = record["y"]
    return labels


def read_results(input_dir: Path) -> dict[str, list[float]]:
    results = {}
    for path in sorted(input_dir.glob("*.caid")):
        with path.open() as handle:
            lines = handle.readlines()

        protein_id = lines[0].strip().lstrip(">")
        if len(protein_id) >= 7 and protein_id.isdigit():
            base_len = len(protein_id) - 3
            protein_id = protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])

        scores = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                scores.append(float(line.split("\t")[2]))
        results[protein_id] = scores
    return results


def flatten_masked(values: list[list[float]], mask_value: float = 999) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor([item for row in values for item in row], dtype=torch.float32)
    mask = tensor != mask_value
    return tensor[mask], mask


def bootstrap_std(preds: torch.Tensor, labels: torch.Tensor, metric_fn, n: int, **kwargs) -> float:
    metrics = []
    for _ in range(n):
        indices = torch.randint(0, len(labels), (len(labels),))
        metrics.append(metric_fn(preds[indices], labels[indices], **kwargs))
    return torch.tensor(metrics).std().item()


def load_prediction_frames(udonpred_dir: Path, output_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    results: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for test_set in DATASETS:
        labels = read_labels(udonpred_dir / "data" / test_set / "test.jsonl")
        for train_set in DATASETS:
            preds = read_results(output_dir / "predictions" / f"{train_set}_{test_set}")
            missing = sorted(set(labels) - set(preds))
            if missing:
                raise ValueError(
                    f"{train_set}_{test_set} is missing {len(missing)} predictions; "
                    f"first missing id: {missing[0]}"
                )
            results[train_set][test_set] = pd.DataFrame(
                {"label": pd.Series(labels), "pred": pd.Series(preds)}
            )
    return results


def compute_metrics(
    frames: dict[str, dict[str, pd.DataFrame]],
    bootstrap_samples: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    stds: dict[str, dict[str, float]] = defaultdict(dict)

    for train_set in DATASETS:
        for test_set in DATASETS:
            frame = frames[train_set][test_set]
            labels, mask = flatten_masked(frame["label"].tolist())
            preds = torch.tensor(
                [item for row in frame["pred"].tolist() for item in row],
                dtype=torch.float32,
            )[mask]

            if train_set in NEGATED_DATASETS:
                preds = -preds
            if test_set in NEGATED_DATASETS:
                labels = -labels

            if test_set == "disprot":
                binary_labels = labels.to(torch.int)
                auroc_value = auroc(preds, binary_labels, task="binary")
                ap_value = average_precision(preds, binary_labels, task="binary")
                values[train_set]["disprot\n(AUROC)"] = float(auroc_value)
                values[train_set]["disprot\n(AP)"] = float(ap_value)
                if bootstrap_samples:
                    stds[train_set]["disprot\n(AUROC)"] = bootstrap_std(
                        preds, binary_labels, auroc, bootstrap_samples, task="binary"
                    )
                    stds[train_set]["disprot\n(AP)"] = bootstrap_std(
                        preds, binary_labels, average_precision, bootstrap_samples, task="binary"
                    )
            else:
                value = spearman_corrcoef(preds, labels)
                values[train_set][test_set] = float(value)
                if bootstrap_samples:
                    stds[train_set][test_set] = bootstrap_std(
                        preds, labels, spearman_corrcoef, bootstrap_samples
                    )

    return values, stds


def write_matrix_csv(matrix: dict[str, dict[str, float]], path: Path) -> None:
    columns = [
        "train_dataset",
        "trizod",
        "chezod",
        "softdis",
        "pdbflex",
        "atlas",
        "plddt",
        "disprot\n(AP)",
        "disprot\n(AUROC)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for train_set in DATASETS:
            row = {"train_dataset": train_set}
            for column in columns[1:]:
                value = matrix[train_set].get(column)
                row[column] = "" if value is None else f"{value:.6f}"
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/udonpred_matrix"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument("--force", action="store_true", help="Regenerate existing prediction files.")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    udonpred_dir = args.udonpred_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not args.skip_predictions:
        run_predictions_fast(
            udonpred_dir,
            output_dir,
            args.device,
            args.batch_size,
            args.smooth,
            args.force,
        )

    frames = load_prediction_frames(udonpred_dir, output_dir)
    matrix, stds = compute_metrics(frames, args.bootstrap_samples)

    write_matrix_csv(matrix, output_dir / "matrix.csv")
    if args.bootstrap_samples:
        write_matrix_csv(stds, output_dir / "matrix_std.csv")

    print(f"Wrote {output_dir / 'matrix.csv'}")
    if args.bootstrap_samples:
        print(f"Wrote {output_dir / 'matrix_std.csv'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
