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
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import gaussian_filter1d
from torchmetrics.functional import auroc, average_precision, spearman_corrcoef
from tqdm import tqdm


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
    include_training_test_split: bool,
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
    copy_names = ["valid.jsonl", "valid.fasta"]
    if include_training_test_split:
        copy_names += ["test.jsonl", "test.fasta"]

    for name in copy_names:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)

    split_note = (
        "with test split"
        if include_training_test_split
        else "without test split for training cache efficiency"
    )
    print(
        f"Prepared shuffled {dataset} seed {seed}: "
        f"{summary['records']} records, {summary['valid_labels']} valid labels, "
        f"{summary['masked_labels']} masked labels; {split_note}"
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


def has_checkpoint_and_head(seed: int, dataset: str, udonpred_dir: Path, seed_dir: Path) -> bool:
    checkpoint_root = udonpred_dir / "checkpoints" / f"shuffled_seed_{seed}_{dataset}"
    return select_checkpoint(checkpoint_root) is not None and (seed_dir / "weights" / f"{dataset}.onnx").exists()


def read_results(input_dir: Path) -> dict[str, list[float]]:
    results = {}
    for path in sorted(input_dir.glob("*.caid")):
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines:
            raise ValueError(f"Empty prediction file: {path}")
        protein_id = lines[0].strip().lstrip(">")
        if len(protein_id) >= 7 and protein_id.isdigit():
            base_len = len(protein_id) - 3
            protein_id = protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])
        scores = [float(line.strip().split("\t")[2]) for line in lines[1:] if line.strip()]
        results[protein_id] = scores
    return results


def load_predict_module(udonpred_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("udonpred_predict", udonpred_dir / "predict.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {udonpred_dir / 'predict.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_file_name(header: str) -> str:
    safe_header = header.replace("/", "_").replace("|", "_")
    return f"{safe_header}.caid"


def write_prediction_file(out_dir: Path, header: str, sequence: str, scores: np.ndarray, predict_module: Any) -> None:
    file_path = out_dir / prediction_file_name(header)
    temp_path = file_path.with_name(f"{file_path.name}.tmp")
    temp_path.write_text(
        "".join(predict_module.format_predictions(header, sequence, scores)),
        encoding="utf-8",
    )
    temp_path.replace(file_path)


def is_complete_prediction_dir(result_dir: Path, entries: list[tuple[str, str]]) -> bool:
    if not result_dir.exists():
        return False
    expected_files = [result_dir / prediction_file_name(header) for header, _ in entries]
    return all(path.exists() and path.stat().st_size > 0 for path in expected_files)


def entries_digest(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for header, sequence in entries:
        digest.update(header.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sequence.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cached_embedding_batches(
    cache_root: Path,
    test_set: str,
    entries: list[tuple[str, str]],
    cache_metadata: dict[str, str],
) -> list[Path] | None:
    dataset_cache_dir = cache_root / test_set
    manifest_path = dataset_cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    batch_files = [dataset_cache_dir / name for name in manifest.get("batch_files", [])]
    if (
        manifest.get("version") != 1
        or manifest.get("entry_count") != len(entries)
        or manifest.get("entries_sha256") != entries_digest(entries)
        or any(manifest.get(key) != value for key, value in cache_metadata.items())
        or not batch_files
        or not all(path.exists() for path in batch_files)
    ):
        return None
    return batch_files


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    temp_path.replace(path)


def remove_cache_tree(path: Path, cache_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = cache_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to remove cache path outside cache root: {path}")
    shutil.rmtree(resolved_path)


def write_embedding_manifest(
    cache_root: Path,
    test_set: str,
    entries: list[tuple[str, str]],
    batch_files: list[Path],
    batch_size: int,
    cache_metadata: dict[str, str],
) -> None:
    manifest = {
        "version": 1,
        "test_set": test_set,
        "entry_count": len(entries),
        "entries_sha256": entries_digest(entries),
        "batch_size": batch_size,
        "batch_files": [path.name for path in batch_files],
        **cache_metadata,
    }
    (cache_root / test_set / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def iter_computed_embedding_batches(
    cache_root: Path,
    test_set: str,
    entries: list[tuple[str, str]],
    predict_module: Any,
    tokenizer: Any,
    backbone: Any,
    torch_device: str,
    batch_size: int,
    cache_metadata: dict[str, str],
) -> Iterator[tuple[list[str], list[str], np.ndarray]]:
    dataset_cache_dir = cache_root / test_set
    if dataset_cache_dir.exists():
        remove_cache_tree(dataset_cache_dir, cache_root)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing and saving embeddings for {test_set} ...")
    batch_files: list[Path] = []
    prefix_token_len = 1
    total_batches = predict_module.count_batches(entries, batch_size)

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(predict_module.iter_batches(entries, batch_size), total=total_batches)
        ):
            headers, seqs = zip(*batch)
            seqs = list(seqs)
            max_seq_len = max(len(seq) for seq in seqs)

            input_ids, attention_mask = predict_module.tokenize_batch(tokenizer, seqs, torch_device)
            hidden = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
            hidden = hidden * attention_mask.unsqueeze(-1)
            emb = hidden[:, prefix_token_len : prefix_token_len + max_seq_len, :]
            emb_np = emb.float().cpu().numpy()

            batch_path = dataset_cache_dir / f"batch_{batch_index:05d}.npz"
            save_npz_atomic(
                batch_path,
                headers_json=np.array(json.dumps(list(headers))),
                seqs_json=np.array(json.dumps(seqs)),
                embeddings=emb_np,
            )
            batch_files.append(batch_path)
            yield list(headers), seqs, emb_np

    write_embedding_manifest(cache_root, test_set, entries, batch_files, batch_size, cache_metadata)


def load_embedding_batch(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with np.load(path) as data:
        headers = json.loads(str(data["headers_json"].item()))
        seqs = json.loads(str(data["seqs_json"].item()))
        embeddings = data["embeddings"]
    return headers, seqs, embeddings


def iter_cached_embedding_batches(batch_files: list[Path]) -> Iterator[tuple[list[str], list[str], np.ndarray]]:
    for batch_path in tqdm(batch_files):
        yield load_embedding_batch(batch_path)


def apply_heads_to_embedding_batch(
    seed_dir: Path,
    test_set: str,
    active_train_sets: list[str],
    heads: dict[str, Any],
    head_io: dict[str, tuple[str, str]],
    headers: list[str],
    seqs: list[str],
    emb_np: np.ndarray,
    smooth: float,
    predict_module: Any,
) -> None:
    for train_set in active_train_sets:
        head = heads[train_set]
        head_input_name, head_output_name = head_io[train_set]
        scores_batch = head.run([head_output_name], {head_input_name: emb_np})[0]
        result_dir = seed_dir / "predictions" / f"{train_set}_{test_set}"

        for header, seq, scores in zip(headers, seqs, scores_batch):
            scores_seq = scores[: len(seq)]
            if smooth > 0:
                scores_seq = gaussian_filter1d(
                    scores_seq.astype(np.float64), sigma=smooth, axis=0
                )
            write_prediction_file(result_dir, header, seq, scores_seq, predict_module)


def run_predictions(
    udonpred_dir: Path,
    seed_dir: Path,
    datasets: list[str],
    device: str,
    batch_size: int,
    smooth: float,
    force: bool,
    embedding_cache_dir: Path | None,
    force_embeddings: bool,
) -> None:
    predict_module = load_predict_module(udonpred_dir)
    torch_device = predict_module.resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32
    cache_root = embedding_cache_dir or (seed_dir.parent / "embeddings")
    cache_metadata = {
        "backbone_name": str(predict_module.BACKBONE_NAME),
        "prefix_token": str(predict_module.PREFIX_TOKEN),
        "backbone_compute_dtype": str(torch_dtype),
    }
    weights_dir = seed_dir / "weights"
    heads: dict[str, Any] = {}
    head_io: dict[str, tuple[str, str]] = {}

    def ensure_heads_loaded(train_sets: list[str]) -> None:
        for train_set in train_sets:
            if train_set in heads:
                continue
            head = predict_module.load_head(weights_dir / f"{train_set}.onnx", torch_device)
            heads[train_set] = head
            head_io[train_set] = (head.get_inputs()[0].name, head.get_outputs()[0].name)

    tokenizer = None
    backbone = None
    entries_by_test_set = {
        test_set: sorted(
            predict_module.read_fasta(str(udonpred_dir / "data" / test_set / "test.fasta")),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        for test_set in DATASETS
    }

    for test_set in DATASETS:
        entries = entries_by_test_set[test_set]

        active_train_sets = []
        for train_set in datasets:
            result_dir = seed_dir / "predictions" / f"{train_set}_{test_set}"
            if is_complete_prediction_dir(result_dir, entries) and not force:
                print(f"Skipping complete predictions: {result_dir}")
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            active_train_sets.append(train_set)

        if not active_train_sets:
            continue

        ensure_heads_loaded(active_train_sets)
        batch_files = (
            None
            if force_embeddings
            else cached_embedding_batches(cache_root, test_set, entries, cache_metadata)
        )
        if batch_files is None:
            if tokenizer is None:
                print(f"Loading tokenizer ({predict_module.BACKBONE_NAME}) ...")
                tokenizer = predict_module.load_tokenizer()
            if backbone is None:
                print(f"Loading backbone ({predict_module.BACKBONE_NAME}) ...")
                backbone = predict_module.load_backbone(torch_device, torch_dtype)
            embedding_batches = iter_computed_embedding_batches(
                cache_root,
                test_set,
                entries,
                predict_module,
                tokenizer,
                backbone,
                torch_device,
                batch_size,
                cache_metadata,
            )
        else:
            print(f"Using cached embeddings for {test_set}: {cache_root / test_set}")
            embedding_batches = iter_cached_embedding_batches(batch_files)

        print(f"Predicting {test_set} with heads: {', '.join(active_train_sets)}")
        for headers, seqs, emb_np in embedding_batches:
            apply_heads_to_embedding_batch(
                seed_dir,
                test_set,
                active_train_sets,
                heads,
                head_io,
                headers,
                seqs,
                emb_np,
                smooth,
                predict_module,
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
    for test_set in DATASETS:
        reference_frame = frames[datasets[0]][test_set]
        labels, mask = flatten_masked(reference_frame["label"].tolist())
        if test_set in NEGATED_DATASETS:
            labels = -labels

        for train_set in datasets:
            frame = frames[train_set][test_set]
            preds = torch.tensor(
                [item for row in frame["pred"].tolist() for item in row],
                dtype=torch.float32,
            )[mask]

            if train_set in NEGATED_DATASETS:
                preds = -preds

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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for saved ProstT5 test-set embeddings. Defaults to "
            "<output-dir>/embeddings, shared by all shuffled seeds."
        ),
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Recompute saved embeddings even when a matching cache exists.",
    )
    parser.add_argument(
        "--include-training-test-split",
        action="store_true",
        help=(
            "Copy test.jsonl and test.fasta into shuffled training data directories. "
            "By default only train/validation files are prepared because evaluation "
            "uses the original UdonPred test sets."
        ),
    )
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
                udonpred_dir,
                output_dir,
                dataset,
                seed,
                force=args.force,
                include_training_test_split=args.include_training_test_split,
            )
            if args.prepare_only:
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
                continue
            if not args.skip_training:
                if has_checkpoint_and_head(seed, dataset, udonpred_dir, seed_dir) and not args.force:
                    print(
                        "Skipping config rewrite for existing checkpoint and ONNX head: "
                        f"shuffled_seed_{seed}_{dataset}"
                    )
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
                embedding_cache_dir=(
                    args.embedding_cache_dir.resolve()
                    if args.embedding_cache_dir is not None
                    else None
                ),
                force_embeddings=args.force_embeddings,
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
