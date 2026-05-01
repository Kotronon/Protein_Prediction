#!/usr/bin/env python3
"""Train UdonPred shuffled-label null models and evaluate 7x7 matrices.

For each random seed and training dataset, this script creates an isolated
training-data copy with residue labels shuffled inside the training split,
trains a UdonPred head on CPU by default, exports the selected checkpoint to
ONNX, and evaluates every shuffled head against the original seven test sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import yaml
from torchmetrics.functional import auroc, average_precision, spearman_corrcoef


DATASETS = ["trizod", "chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"]
NEGATED_DATASETS = {"plddt", "chezod"}
MASK_VALUE = 999


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def read_labels(path: Path) -> dict[str, list[float]]:
    return {str(record["id"]): record["y"] for record in read_jsonl(path)}


def is_masked_label(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == MASK_VALUE


def shuffle_training_labels(source_path: Path, target_path: Path, seed: int) -> dict[str, int]:
    """Shuffle valid residue labels across a train split while preserving masks."""
    generator = torch.Generator().manual_seed(seed)
    records = read_jsonl(source_path)

    valid_values: list[object] = []
    valid_slots: list[tuple[int, int]] = []
    masked_count = 0
    for record_index, record in enumerate(records):
        sequence = str(record["x_0"])
        labels = record["y"]
        if len(sequence) != len(labels):
            raise ValueError(
                f"{source_path}: {record['id']} has {len(sequence)} residues but {len(labels)} labels"
            )
        for label_index, value in enumerate(labels):
            if is_masked_label(value):
                masked_count += 1
            else:
                valid_slots.append((record_index, label_index))
                valid_values.append(value)

    if valid_values:
        order = torch.randperm(len(valid_values), generator=generator).tolist()
        shuffled = [valid_values[index] for index in order]
        for (record_index, label_index), value in zip(valid_slots, shuffled):
            records[record_index]["y"][label_index] = value

    write_jsonl(records, target_path)
    return {
        "records": len(records),
        "valid_labels": len(valid_values),
        "masked_labels": masked_count,
    }


def prepare_shuffled_dataset(
    udonpred_dir: Path,
    output_dir: Path,
    dataset: str,
    seed: int,
    force: bool,
) -> Path:
    source_dir = udonpred_dir / "data" / dataset
    target_dir = output_dir / f"seed_{seed}" / "data" / dataset
    train_target = target_dir / "train.jsonl"
    if train_target.exists() and not force:
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = shuffle_training_labels(
        source_dir / "train.jsonl",
        train_target,
        seed=seed,
    )
    for name in ["valid.jsonl", "valid.fasta", "test.jsonl", "test.fasta"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)

    print(
        f"Prepared shuffled {dataset} seed {seed}: "
        f"{summary['records']} records, {summary['valid_labels']} valid labels, "
        f"{summary['masked_labels']} masked labels"
    )
    return target_dir


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_training_config(
    udonpred_dir: Path,
    seed_dir: Path,
    dataset: str,
    shuffled_data_dir: Path,
    run_name: str,
    cuda_devices: str,
    train_device: str,
    num_train_epochs: int | None,
    save_steps: int | None,
    eval_steps: int | None,
) -> Path:
    """Write an isolated UdonPred config directory for a shuffled training run."""
    source_config_dir = udonpred_dir / "config"
    target_config_dir = seed_dir / "configs" / dataset
    if target_config_dir.exists():
        shutil.rmtree(target_config_dir)
    target_config_dir.mkdir(parents=True, exist_ok=True)

    for source in source_config_dir.glob("*.yaml"):
        shutil.copy2(source, target_config_dir / source.name)

    config_path = target_config_dir / "config.yaml"
    data_path = target_config_dir / "data.yaml"
    config = load_yaml(config_path)
    data = load_yaml(data_path)

    config["run_name"] = run_name
    config["cuda_devices"] = "" if train_device == "cpu" else cuda_devices
    config["force_cpu"] = train_device == "cpu"
    if num_train_epochs is not None:
        config["num_train_epochs"] = num_train_epochs
    if save_steps is not None:
        config.setdefault("saving", {})["save_steps"] = save_steps
    if eval_steps is not None:
        config.setdefault("logging", {})["eval_steps"] = eval_steps

    for name, dataset_config in data.items():
        dataset_config["fraction"] = 1 if name == dataset else 0
        if name == dataset:
            dataset_config["path"] = str(shuffled_data_dir.resolve())

    dump_yaml(config, config_path)
    dump_yaml(data, data_path)
    return target_config_dir


def run_command(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd))
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    command_env.setdefault("PYTHONUTF8", "1")
    command_env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=cwd, env=command_env, check=True)


def train_model(
    udonpred_dir: Path,
    dataset: str,
    shuffled_data_dir: Path,
    seed_dir: Path,
    seed: int,
    cuda_devices: str,
    train_device: str,
    num_train_epochs: int | None,
    save_steps: int | None,
    eval_steps: int | None,
    wandb_mode: str,
    force: bool,
) -> Path:
    run_name = f"shuffled_seed_{seed}_{dataset}"
    checkpoint_root = udonpred_dir / "checkpoints" / run_name
    selected = select_checkpoint(checkpoint_root)
    if selected and not force:
        print(f"Skipping existing training checkpoint: {selected}")
        return selected
    if checkpoint_root.exists() and force:
        shutil.rmtree(checkpoint_root)

    env = os.environ.copy()
    env["WANDB_MODE"] = wandb_mode
    env["PYTHONHASHSEED"] = str(seed)
    if train_device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["UDONPRED_FORCE_CPU"] = "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
        env.pop("UDONPRED_FORCE_CPU", None)

    config_dir = write_training_config(
        udonpred_dir,
        seed_dir,
        dataset,
        shuffled_data_dir,
        run_name,
        cuda_devices,
        train_device,
        num_train_epochs,
        save_steps,
        eval_steps,
    )
    env["UDONPRED_CONFIG_DIR"] = str(config_dir.resolve())
    run_command(["uv", "run", "run.py", "train"], cwd=udonpred_dir, env=env)

    selected = select_checkpoint(checkpoint_root)
    if selected is None:
        if (checkpoint_root / "pytorch_model.bin").exists():
            return checkpoint_root
        raise FileNotFoundError(f"No checkpoint with pytorch_model.bin found under {checkpoint_root}")
    return selected


def select_checkpoint(checkpoint_root: Path) -> Path | None:
    if not checkpoint_root.exists():
        return None
    candidates = [path for path in checkpoint_root.rglob("pytorch_model.bin")]
    if not candidates:
        return None

    def step(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", str(path))
        return int(match.group(1)) if match else -1

    return max((path.parent for path in candidates), key=step)


def export_head(udonpred_dir: Path, checkpoint_dir: Path, weights_dir: Path, dataset: str, force: bool) -> Path:
    target = weights_dir / f"{dataset}.onnx"
    if target.exists() and not force:
        print(f"Skipping existing ONNX head: {target}")
        return target

    temp_dir = weights_dir / "_export_tmp" / dataset
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "uv",
            "run",
            "export.py",
            str(checkpoint_dir),
            "--output-dir",
            str(temp_dir),
            "--checkpoints",
            ".",
        ],
        cwd=udonpred_dir,
    )
    exported = sorted(temp_dir.glob("*.onnx"))
    exported += sorted(temp_dir.parent.glob(f"{temp_dir.name}.onnx"))
    if len(exported) != 1:
        raise RuntimeError(f"Expected one exported ONNX file in {temp_dir}, found {len(exported)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported[0]), target)
    shutil.rmtree(temp_dir.parent, ignore_errors=True)
    print(f"Saved {target}")
    return target


def read_results(input_dir: Path) -> dict[str, list[float]]:
    results = {}
    for path in sorted(input_dir.glob("*.caid")):
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        protein_id = lines[0].strip().lstrip(">")
        if len(protein_id) >= 7 and protein_id.isdigit():
            base_len = len(protein_id) - 3
            protein_id = protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])
        scores = [float(line.strip().split("\t")[2]) for line in lines[1:] if line.strip()]
        results[protein_id] = scores
    return results


def run_predictions(
    udonpred_dir: Path,
    seed_dir: Path,
    datasets: list[str],
    device: str,
    batch_size: int,
    smooth: float,
    force: bool,
) -> None:
    weights_dir = seed_dir / "weights"
    for train_set in datasets:
        for test_set in DATASETS:
            result_dir = seed_dir / "predictions" / f"{train_set}_{test_set}"
            if result_dir.exists() and any(result_dir.glob("*.caid")) and not force:
                print(f"Skipping existing predictions: {result_dir}")
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            run_command(
                [
                    "uv",
                    "run",
                    "predict.py",
                    str(udonpred_dir / "data" / test_set / "test.fasta"),
                    str(weights_dir.resolve()),
                    "--target",
                    train_set,
                    "--output",
                    str(result_dir.resolve()),
                    "--device",
                    device,
                    "--batch-size",
                    str(batch_size),
                    "--smooth",
                    str(smooth),
                ],
                cwd=udonpred_dir,
            )


def flatten_masked(values: list[list[float]], mask_value: float = MASK_VALUE) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor([item for row in values for item in row], dtype=torch.float32)
    mask = tensor != mask_value
    return tensor[mask], mask


def load_prediction_frames(
    udonpred_dir: Path,
    seed_dir: Path,
    datasets: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    frames: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for test_set in DATASETS:
        labels = read_labels(udonpred_dir / "data" / test_set / "test.jsonl")
        for train_set in datasets:
            preds = read_results(seed_dir / "predictions" / f"{train_set}_{test_set}")
            missing = sorted(set(labels) - set(preds))
            if missing:
                raise ValueError(
                    f"{train_set}_{test_set} is missing {len(missing)} predictions; "
                    f"first missing id: {missing[0]}"
                )
            frames[train_set][test_set] = pd.DataFrame(
                {"label": pd.Series(labels), "pred": pd.Series(preds)}
            )
    return frames


def compute_metrics(frames: dict[str, dict[str, pd.DataFrame]], datasets: list[str]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for train_set in datasets:
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
                values[train_set]["disprot\n(AP)"] = float(
                    average_precision(preds, binary_labels, task="binary")
                )
                values[train_set]["disprot\n(AUROC)"] = float(
                    auroc(preds, binary_labels, task="binary")
                )
            else:
                value = spearman_corrcoef(preds, labels)
                values[train_set][test_set] = float(value)
    return values


def write_matrix_csv(matrix: dict[str, dict[str, float]], path: Path, datasets: list[str]) -> None:
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for train_set in datasets:
            row = {"train_dataset": train_set}
            for column in columns[1:]:
                value = matrix[train_set].get(column)
                row[column] = "" if value is None or math.isnan(value) else f"{value:.6f}"
            writer.writerow(row)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/udonpred_shuffled_labels"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 14, 15])
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument("--train-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create shuffled train splits and exit before training, prediction, or metrics.",
    )
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


def resolve_udonpred_dir(path: Path) -> Path:
    resolved = path.resolve()
    if (resolved / "run.py").exists() and (resolved / "data").exists():
        return resolved
    if Path.cwd().name == "UdonPred" and Path("run.py").exists() and Path("data").exists():
        return Path.cwd().resolve()
    raise FileNotFoundError(
        f"Could not resolve UdonPred directory from {path}. "
        "Pass --udonpred-dir explicitly if running from a different directory."
    )

def main() -> None:
    args = parse_args()
    udonpred_dir = resolve_udonpred_dir(args.udonpred_dir)
    output_dir = args.output_dir.resolve()

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        weights_dir = seed_dir / "weights"
        for dataset in args.datasets:
            shuffled_data_dir = prepare_shuffled_dataset(
                udonpred_dir, output_dir, dataset, seed, force=args.force
            )
            write_training_config(
                udonpred_dir=udonpred_dir,
                seed_dir=seed_dir,
                dataset=dataset,
                shuffled_data_dir=shuffled_data_dir,
                run_name=f"shuffled_seed_{seed}_{dataset}",
                cuda_devices=args.cuda_devices,
                train_device=args.train_device,
                num_train_epochs=args.num_train_epochs,
                save_steps=args.save_steps,
                eval_steps=args.eval_steps,
            )
            if args.prepare_only:
                continue
            if not args.skip_training:
                checkpoint = train_model(
                    udonpred_dir=udonpred_dir,
                    dataset=dataset,
                    shuffled_data_dir=shuffled_data_dir,
                    seed_dir=seed_dir,
                    seed=seed,
                    cuda_devices=args.cuda_devices,
                    train_device=args.train_device,
                    num_train_epochs=args.num_train_epochs,
                    save_steps=args.save_steps,
                    eval_steps=args.eval_steps,
                    wandb_mode=args.wandb_mode,
                    force=args.force,
                )
                export_head(udonpred_dir, checkpoint, weights_dir, dataset, force=args.force)
            elif not (weights_dir / f"{dataset}.onnx").exists():
                raise FileNotFoundError(
                    f"--skip-training requires existing ONNX head: {weights_dir / f'{dataset}.onnx'}"
                )

        if args.prepare_only:
            print(f"Prepared shuffled data for seed {seed}; exiting before training.")
            continue

        if not args.skip_predictions:
            run_predictions(
                udonpred_dir=udonpred_dir,
                seed_dir=seed_dir,
                datasets=args.datasets,
                device=args.device,
                batch_size=args.batch_size,
                smooth=args.smooth,
                force=args.force,
            )

        if args.skip_metrics:
            print(f"Skipped metrics for seed {seed}.")
            continue

        frames = load_prediction_frames(udonpred_dir, seed_dir, args.datasets)
        matrix = compute_metrics(frames, args.datasets)
        matrix_path = seed_dir / "matrix.csv"
        write_matrix_csv(matrix, matrix_path, args.datasets)
        print(f"Wrote {matrix_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
