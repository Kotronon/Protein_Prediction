#!/usr/bin/env python3
"""Run the ESM-1b/Dispredict3 stage natively with resumable CAID outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import esm
import joblib
import numpy as np
import torch
from Bio import SeqIO
from sklearn.exceptions import InconsistentVersionWarning


ESM_LAYERS = tuple(range(34))
ESM_CHUNK_SIZE = 1022
THRESHOLD = 0.382


class PortableLightGBM:
    """Minimal vectorized evaluator for this model's numeric binary trees."""

    def __init__(self, path: Path) -> None:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model.get("objective") != "binary sigmoid:1":
            raise ValueError(f"Unsupported LightGBM objective: {model.get('objective')}")
        if model.get("average_output"):
            raise ValueError("LightGBM average_output models are not supported")
        self.trees = [tree["tree_structure"] for tree in model["tree_info"]]
        self.original_num_features = int(model["max_feature_idx"]) + 1
        used_features: set[int] = set()
        for tree in self.trees:
            self._collect_features(tree, used_features)
        self.feature_indices = np.asarray(sorted(used_features), dtype=np.int64)
        feature_positions = {
            feature: position
            for position, feature in enumerate(self.feature_indices.tolist())
        }
        for tree in self.trees:
            self._remap_features(tree, feature_positions)
        self.num_features = len(self.feature_indices)

    @staticmethod
    def _collect_features(node: dict[str, object], features: set[int]) -> None:
        if "leaf_value" in node:
            return
        features.add(int(node["split_feature"]))
        PortableLightGBM._collect_features(node["left_child"], features)
        PortableLightGBM._collect_features(node["right_child"], features)

    @staticmethod
    def _remap_features(
        node: dict[str, object],
        feature_positions: dict[int, int],
    ) -> None:
        if "leaf_value" in node:
            return
        original = int(node["split_feature"])
        node["split_feature"] = feature_positions[original]
        PortableLightGBM._remap_features(node["left_child"], feature_positions)
        PortableLightGBM._remap_features(node["right_child"], feature_positions)

    @staticmethod
    def _evaluate_tree(
        node: dict[str, object],
        features: np.ndarray,
        indices: np.ndarray,
        output: np.ndarray,
    ) -> None:
        if "leaf_value" in node:
            output[indices] = float(node["leaf_value"])
            return

        if node.get("decision_type") != "<=":
            raise ValueError(
                f"Unsupported LightGBM decision type: {node.get('decision_type')}"
            )
        feature_index = int(node["split_feature"])
        values = features[indices, feature_index]
        go_left = values <= float(node["threshold"])
        missing = np.isnan(values)
        if bool(node.get("default_left")):
            go_left |= missing
        else:
            go_left &= ~missing

        if np.any(go_left):
            PortableLightGBM._evaluate_tree(
                node["left_child"],
                features,
                indices[go_left],
                output,
            )
        if np.any(~go_left):
            PortableLightGBM._evaluate_tree(
                node["right_child"],
                features,
                indices[~go_left],
                output,
            )

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError(
                f"Expected feature matrix with {self.num_features} columns, "
                f"got {features.shape}"
            )
        indices = np.arange(len(features))
        raw_scores = np.zeros(len(features), dtype=np.float64)
        tree_scores = np.empty(len(features), dtype=np.float64)
        for tree in self.trees:
            self._evaluate_tree(tree, features, indices, tree_scores)
            raw_scores += tree_scores
        positive = raw_scores >= 0
        probabilities = np.empty_like(raw_scores)
        probabilities[positive] = 1.0 / (1.0 + np.exp(-raw_scores[positive]))
        exp_scores = np.exp(raw_scores[~positive])
        probabilities[~positive] = exp_scores / (1.0 + exp_scores)
        return probabilities


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Run this script from a native macOS terminal "
            "with an arm64 PyTorch installation."
        )
    return torch.device(requested)


def output_is_complete(path: Path, protein_id: str, sequence: str) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("protein_id") == protein_id
        and metadata.get("length") == len(sequence)
        and metadata.get("sequence_sha256")
        == hashlib.sha256(sequence.encode("ascii")).hexdigest()
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def skipped_flDPnn_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["protein_id"]
            for row in csv.DictReader(handle)
            if row.get("protein_id")
        }


def esm_features(
    sequence: str,
    protein_id: str,
    model: torch.nn.Module,
    batch_converter: object,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(sequence), ESM_CHUNK_SIZE):
        chunk = sequence[start : start + ESM_CHUNK_SIZE]
        _, _, tokens = batch_converter([(protein_id, chunk)])
        tokens = tokens.to(device)
        with torch.inference_mode():
            result = model(tokens, repr_layers=ESM_LAYERS, return_contacts=False)

        representations = (
            torch.stack(
                [
                    result["representations"][layer][0, 1 : len(chunk) + 1]
                    for layer in ESM_LAYERS
                ],
                dim=1,
            )
            .float()
            .cpu()
            .numpy()
        )
        means = representations.mean(axis=2, keepdims=True)
        chunks.append(
            np.concatenate((representations, means), axis=2).reshape(len(chunk), -1)
        )

        del result, tokens, representations, means
    return np.concatenate(chunks, axis=0)


def reduce_features(
    raw_features: np.ndarray,
    scaler: object,
    pca: object,
    row_batch_size: int,
    component_indices: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(raw_features).all():
        bad_count = raw_features.size - np.count_nonzero(np.isfinite(raw_features))
        raise ValueError(f"ESM features contain {bad_count} non-finite values")

    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
    pca_mean = np.asarray(pca.mean_, dtype=np.float64)
    components = np.asarray(pca.components_, dtype=np.float64)[component_indices]
    if raw_features.shape[1] != scaler_mean.shape[0]:
        raise ValueError(
            f"Feature width {raw_features.shape[1]} does not match scaler width "
            f"{scaler_mean.shape[0]}"
        )

    reduced: list[np.ndarray] = []
    for start in range(0, len(raw_features), row_batch_size):
        batch = raw_features[start : start + row_batch_size].astype(
            np.float64, copy=False
        )
        scaled = (batch - scaler_mean) / scaler_scale
        if not np.isfinite(scaled).all():
            raise ValueError(
                f"Scaling produced non-finite values for residue rows "
                f"{start + 1}-{start + len(batch)}"
            )

        # Apple's Accelerate backend can emit spurious floating-point warnings
        # for this very wide matrix product. Validate the result explicitly.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            transformed = (scaled - pca_mean) @ components.T
        if getattr(pca, "whiten", False):
            transformed /= np.sqrt(
                np.asarray(pca.explained_variance_, dtype=np.float64)[
                    component_indices
                ]
            )
        if not np.isfinite(transformed).all():
            raise ValueError(
                f"PCA produced non-finite values for residue rows "
                f"{start + 1}-{start + len(batch)}"
            )
        reduced.append(transformed)
    return np.concatenate(reduced, axis=0)


def format_caid(
    protein_id: str,
    sequence: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> str:
    lines = [f">{protein_id}"]
    lines.extend(
        f"{index}\t{residue}\t{probability:.3f}\t{label}"
        for index, (residue, probability, label) in enumerate(
            zip(sequence, probabilities, labels), start=1
        )
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, nargs="+")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("results/human_proteome/Dispredict3_native/features"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/human_proteome/Dispredict3_native/models"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_proteome/Dispredict3_native/caid"),
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--row-batch-size", type=int, default=64)
    parser.add_argument(
        "--skipped-manifest",
        type=Path,
        help="CSV of proteins without flDPnn features; defaults beside feature-dir",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row_batch_size < 1:
        raise ValueError("--row-batch-size must be positive")
    skipped_manifest = (
        args.skipped_manifest
        if args.skipped_manifest is not None
        else args.feature_dir.parent / "skipped_flDPnn.csv"
    )
    unsupported_ids = skipped_flDPnn_ids(skipped_manifest)

    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        scaler = joblib.load(args.model_dir / "scaler.pkl")
        pca = joblib.load(args.model_dir / "pca.pkl")
    classifier = PortableLightGBM(args.model_dir / "dispredict3_lightgbm.json")
    pca_feature_count = np.asarray(pca.components_).shape[0]
    fld_feature_count = classifier.original_num_features - pca_feature_count
    fld_indices = classifier.feature_indices[
        classifier.feature_indices < fld_feature_count
    ]
    pca_indices = (
        classifier.feature_indices[
            classifier.feature_indices >= fld_feature_count
        ]
        - fld_feature_count
    )
    print(
        f"Using {len(fld_indices)}/{fld_feature_count} flDPnn and "
        f"{len(pca_indices)}/{pca_feature_count} PCA features",
        flush=True,
    )

    print("Loading ESM-1b...", flush=True)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    records = [
        record
        for fasta_path in args.fasta
        for record in SeqIO.parse(fasta_path, "fasta")
    ]
    record_ids = [record.id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate protein IDs found across input FASTA files")
    completed = 0
    resumed = 0
    unsupported = 0
    started = time.monotonic()

    for index, record in enumerate(records, start=1):
        protein_id = record.id
        sequence = str(record.seq).upper()
        output_path = args.output_dir / f"{protein_id}.caid"
        if protein_id in unsupported_ids:
            unsupported += 1
            print(
                f"[{index}/{len(records)}] {protein_id}: skipped, no flDPnn features",
                flush=True,
            )
            continue
        if not args.overwrite and output_is_complete(output_path, protein_id, sequence):
            resumed += 1
            continue

        feature_paths = {
            suffix: args.feature_dir / f"{protein_id}.{suffix}"
            for suffix in ("ttscore", "ttpreds", "ttindex")
        }
        missing = [str(path) for path in feature_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{protein_id}: missing flDPnn features: {', '.join(missing)}"
            )

        protein_started = time.monotonic()
        raw_esm = esm_features(sequence, protein_id, model, batch_converter, device)
        esm_elapsed = time.monotonic() - protein_started
        reduced_esm = reduce_features(
            raw_esm,
            scaler,
            pca,
            args.row_batch_size,
            pca_indices,
        )
        pca_elapsed = time.monotonic() - protein_started - esm_elapsed
        del raw_esm

        fld_scores = np.loadtxt(feature_paths["ttscore"], dtype=np.float32)
        if fld_scores.ndim == 1:
            fld_scores = fld_scores.reshape(-1, 1)
        fld_predictions = np.loadtxt(
            feature_paths["ttpreds"], dtype=np.float32
        )
        if fld_predictions.ndim == 1:
            fld_predictions = fld_predictions.reshape(-1, 1)
        indices = np.loadtxt(feature_paths["ttindex"], dtype=str)
        indices = np.atleast_2d(indices)
        fld_features = np.hstack((fld_scores, fld_predictions))
        if len(reduced_esm) != len(sequence) or len(fld_features) != len(sequence):
            raise ValueError(
                f"{protein_id}: length mismatch: sequence={len(sequence)}, "
                f"ESM={len(reduced_esm)}, flDPnn={len(fld_features)}"
            )
        if fld_features.shape[1] != fld_feature_count:
            raise ValueError(
                f"{protein_id}: expected {fld_feature_count} flDPnn columns, "
                f"got {fld_features.shape[1]}"
            )

        probabilities = classifier.predict(
            np.hstack((fld_features[:, fld_indices], reduced_esm))
        )
        labels = (probabilities >= THRESHOLD).astype(np.int8)
        if indices.shape[0] != len(sequence):
            raise ValueError(
                f"{protein_id}: ttindex length {indices.shape[0]} != {len(sequence)}"
            )

        atomic_write_text(
            output_path,
            format_caid(protein_id, sequence, probabilities, labels),
        )
        metadata = {
            "protein_id": protein_id,
            "length": len(sequence),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "device": str(device),
        }
        atomic_write_text(
            output_path.with_suffix(".json"),
            json.dumps(metadata, sort_keys=True) + "\n",
        )
        completed += 1
        elapsed = time.monotonic() - protein_started
        print(
            f"[{index}/{len(records)}] {protein_id}: {len(sequence)} residues "
            f"in {elapsed:.1f}s (ESM {esm_elapsed:.1f}s, PCA {pca_elapsed:.1f}s)",
            flush=True,
        )

    total_elapsed = time.monotonic() - started
    print(
        f"Finished: {completed} computed, {resumed} resumed, "
        f"{unsupported} unsupported, "
        f"{total_elapsed / 3600:.2f}h",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
