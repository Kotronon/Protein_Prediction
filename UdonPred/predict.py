import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

BACKBONE_NAME = "Rostlab/ProstT5_fp16"
BACKBONE_MODEL_TYPE = "T5EncoderModel"
BACKBONE_TOKENIZER_TYPE = "T5Tokenizer"
PREFIX_TOKEN = "<AA2fold>"

_NONSTANDARD = re.compile(r"[UZOB*]")


@dataclass(frozen=True)
class WorkItem:
    original_index: int
    start: int
    header: str
    sequence: str


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read FASTA file and return list of (header, sequence) tuples."""
    entries: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []

    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(re.sub(r"\s+", "", line))

    if header is not None:
        entries.append((header, "".join(seq_chunks)))

    return entries


def format_predictions(
    header: str,
    sequence: str,
    scores,
) -> List[str]:
    lines: List[str] = [f">{header}\n"]
    for idx, (aa, row) in enumerate(zip(sequence, scores), start=1):
        if hasattr(row, "__len__") and not isinstance(row, str):
            val = row[0] if len(row) > 0 else 0.0
        else:
            val = row
        lines.append(f"{idx}\t{aa}\t{float(val):.3f}\t\n")
    return lines


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError("Device must be one of: auto, cpu, cuda")


def iter_batches(items: List[Tuple[str, str]], max_total_len: int):
    batch: List[Tuple[str, str]] = []
    total_len = 0
    for header, seq in items:
        seq_len = len(seq)
        if batch and total_len + seq_len > max_total_len:
            yield batch
            batch = []
            total_len = 0
        batch.append((header, seq))
        total_len += seq_len
        if total_len >= max_total_len:
            yield batch
            batch = []
            total_len = 0
    if batch:
        yield batch


def count_batches(items: List[Tuple[str, str]], max_total_len: int) -> int:
    count = 0
    total_len = 0
    for _, seq in items:
        seq_len = len(seq)
        if count == 0 and total_len == 0 and seq_len == 0:
            continue
        if total_len and total_len + seq_len > max_total_len:
            count += 1
            total_len = 0
        total_len += seq_len
        if total_len >= max_total_len:
            count += 1
            total_len = 0
    if total_len:
        count += 1
    return count


def split_long_sequences(
    entries: List[Tuple[str, str]], max_sequence_len: int, overlap: int
) -> List[WorkItem]:
    if max_sequence_len <= 0:
        raise ValueError("max_sequence_len must be greater than 0")
    if overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if overlap >= max_sequence_len:
        raise ValueError("chunk_overlap must be smaller than max_sequence_len")

    stride = max_sequence_len - overlap
    work_items: List[WorkItem] = []
    for original_index, (header, sequence) in enumerate(entries):
        if len(sequence) <= max_sequence_len:
            work_items.append(WorkItem(original_index, 0, header, sequence))
            continue

        for start in range(0, len(sequence), stride):
            end = min(start + max_sequence_len, len(sequence))
            work_items.append(
                WorkItem(original_index, start, header, sequence[start:end])
            )
            if end == len(sequence):
                break

    return work_items


def iter_work_batches(items: List[WorkItem], max_total_len: int):
    batch: List[WorkItem] = []
    total_len = 0
    for item in items:
        seq_len = len(item.sequence)
        if batch and total_len + seq_len > max_total_len:
            yield batch
            batch = []
            total_len = 0
        batch.append(item)
        total_len += seq_len
        if total_len >= max_total_len:
            yield batch
            batch = []
            total_len = 0
    if batch:
        yield batch


def count_work_batches(items: List[WorkItem], max_total_len: int) -> int:
    count = 0
    total_len = 0
    for item in items:
        seq_len = len(item.sequence)
        if total_len and total_len + seq_len > max_total_len:
            count += 1
            total_len = 0
        total_len += seq_len
        if total_len >= max_total_len:
            count += 1
            total_len = 0
    if total_len:
        count += 1
    return count


def embedding_cache_path(cache_dir: Path, item: WorkItem) -> Path:
    digest = hashlib.sha256()
    digest.update(BACKBONE_NAME.encode("utf-8"))
    digest.update(b"\0")
    digest.update(item.header.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(item.start).encode("ascii"))
    digest.update(b"\0")
    digest.update(item.sequence.encode("ascii", errors="replace"))
    return cache_dir / f"{item.original_index:06d}_{item.start:08d}_{digest.hexdigest()[:16]}.npy"


def pad_embeddings(embeddings: List[np.ndarray], max_seq_len: int) -> np.ndarray:
    hidden_size = embeddings[0].shape[-1]
    padded = np.zeros((len(embeddings), max_seq_len, hidden_size), dtype=np.float32)
    for index, embedding in enumerate(embeddings):
        padded[index, : embedding.shape[0], :] = embedding
    return padded


def load_tokenizer():
    tokeniser_base = getattr(import_module("transformers"), BACKBONE_TOKENIZER_TYPE)
    return tokeniser_base.from_pretrained(
        BACKBONE_NAME,
        use_fast=False,
        do_lower_case=False,
        legacy=True,
    )


def load_backbone(torch_device: str, dtype):
    backbone_base = getattr(import_module("transformers"), BACKBONE_MODEL_TYPE)
    model = backbone_base.from_pretrained(BACKBONE_NAME)
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    return model.eval().to(torch_device).to(dtype)


def tokenize_batch(
    tokenizer,
    seqs: List[str],
    torch_device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    texts = [
        f"{PREFIX_TOKEN} " + " ".join(list(_NONSTANDARD.sub("X", seq))) for seq in seqs
    ]
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    )
    return encoding["input_ids"].to(torch_device), encoding["attention_mask"].to(
        torch_device
    )


def load_head(onnx_path: Path, torch_device: str) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch_device == "cuda"
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)


def run_exported(
    entries: List[Tuple[str, str]],
    model_dir: str,
    target: str,
    max_total_seq_len: int,
    output_path: str | None,
    device: str,
    smooth: float = 1.5,
    max_sequence_len: int | None = None,
    chunk_overlap: int = 64,
    embedding_cache_dir: str | None = None,
    precompute_only: bool = False,
) -> None:
    """Run prediction using the HuggingFace backbone and ONNX prediction head."""
    torch_device = resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32
    if max_sequence_len is None:
        max_sequence_len = max_total_seq_len

    prefix_token_len = 1
    tokenizer = None
    backbone = None

    if precompute_only and not embedding_cache_dir:
        raise ValueError("--precompute-only requires --embedding-cache-dir")

    if precompute_only:
        head = None
        head_input_name = None
        head_output_name = None
    else:
        onnx_path = Path(model_dir) / f"{target}.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX head not found: {onnx_path}")
        print(f"Loading head ({onnx_path}) ...")
        head = load_head(onnx_path, torch_device)
        head_input_name = head.get_inputs()[0].name
        head_output_name = head.get_outputs()[0].name

    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    if embedding_cache_dir:
        cache_dir = Path(embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using embedding cache ({cache_dir}) ...")
    else:
        cache_dir = None

    work_items = split_long_sequences(entries, max_sequence_len, chunk_overlap)
    work_items = sorted(work_items, key=lambda item: len(item.sequence), reverse=True)
    if precompute_only:
        prediction_sums = None
        prediction_counts = None
    else:
        prediction_sums = [
            np.zeros(len(sequence), dtype=np.float64) for _, sequence in entries
        ]
        prediction_counts = [
            np.zeros(len(sequence), dtype=np.uint16) for _, sequence in entries
        ]

    total_batches = count_work_batches(work_items, max_total_seq_len)
    with torch.inference_mode():
        for batch in tqdm(
            iter_work_batches(work_items, max_total_seq_len), total=total_batches
        ):
            seqs = [item.sequence for item in batch]
            max_seq_len = max(len(s) for s in seqs)
            embeddings: List[np.ndarray | None] = [None] * len(batch)
            missing_indices = []
            missing_items = []

            if cache_dir:
                for index, item in enumerate(batch):
                    cache_path = embedding_cache_path(cache_dir, item)
                    if cache_path.exists():
                        embeddings[index] = np.load(cache_path)
                    else:
                        missing_indices.append(index)
                        missing_items.append(item)
            else:
                missing_indices = list(range(len(batch)))
                missing_items = batch

            if missing_items:
                if tokenizer is None:
                    print(f"Loading tokenizer ({BACKBONE_NAME}) ...")
                    tokenizer = load_tokenizer()
                if backbone is None:
                    print(f"Loading backbone ({BACKBONE_NAME}) ...")
                    backbone = load_backbone(torch_device, torch_dtype)

                missing_seqs = [item.sequence for item in missing_items]
                missing_max_seq_len = max(len(s) for s in missing_seqs)
                input_ids, attention_mask = tokenize_batch(
                    tokenizer, missing_seqs, torch_device
                )

                hidden = backbone(
                    input_ids, attention_mask=attention_mask
                ).last_hidden_state
                hidden = hidden * attention_mask.unsqueeze(-1)
                emb = hidden[
                    :, prefix_token_len : prefix_token_len + missing_max_seq_len, :
                ]

                missing_emb_np = emb.float().cpu().numpy()
                for batch_index, item, embedding in zip(
                    missing_indices, missing_items, missing_emb_np
                ):
                    embedding = embedding[: len(item.sequence)].astype(
                        np.float16, copy=False
                    )
                    embeddings[batch_index] = embedding
                    if cache_dir:
                        np.save(embedding_cache_path(cache_dir, item), embedding)

            if any(embedding is None for embedding in embeddings):
                raise RuntimeError("Internal error: missing embedding after cache lookup")
            if precompute_only:
                continue

            emb_np = pad_embeddings(embeddings, max_seq_len)
            scores_batch = head.run([head_output_name], {head_input_name: emb_np})[0]

            for item, scores in zip(batch, scores_batch):
                scores_seq = np.asarray(scores[: len(item.sequence)])
                if scores_seq.ndim > 1:
                    scores_seq = scores_seq[:, 0]
                start = item.start
                end = start + len(item.sequence)
                prediction_sums[item.original_index][start:end] += scores_seq
                prediction_counts[item.original_index][start:end] += 1

    if precompute_only:
        return

    for index, (header, sequence) in enumerate(entries):
        if not np.all(prediction_counts[index]):
            raise RuntimeError(f"Incomplete prediction coverage for {header}")
        scores_seq = prediction_sums[index] / prediction_counts[index]
        if smooth > 0:
            scores_seq = gaussian_filter1d(
                scores_seq.astype(np.float64), sigma=smooth, axis=0
            )
        formatted_lines = format_predictions(header, sequence, scores_seq)
        if out_dir:
            safe_header = header.replace("/", "_").replace("|", "_")
            file_path = out_dir / f"{safe_header}.caid"
            with file_path.open("w") as f:
                f.writelines(formatted_lines)
        else:
            sys.stdout.writelines(formatted_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict different definitions of protein disorder."
    )
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "model_dir", type=str, help="Directory containing ONNX head files"
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="trizod",
        help="Prediction type — must match a {target}.onnx file in model_dir",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path (will save each sequence as a .caid file)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2000,
        help="Max total sequence length per batch (dynamic batching)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--smooth",
        "-s",
        type=float,
        default=1.5,
        metavar="SIGMA",
        help="Sigma for Gaussian smoothing applied to per-residue scores (0 to disable, default: 1.5)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Split longer proteins into windows of this many residues before inference.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Residue overlap used when splitting proteins longer than --max-sequence-length.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=None,
        help="Directory for cached ProstT5 embeddings. Existing chunks are reused.",
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only populate --embedding-cache-dir; do not run an ONNX prediction head.",
    )

    args = parser.parse_args()

    entries = read_fasta(args.fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    entries = sorted(entries, key=lambda item: len(item[1]), reverse=True)

    if not Path(args.model_dir).is_dir():
        raise ValueError("Model directory not found.")

    run_exported(
        entries,
        args.model_dir,
        args.target,
        args.batch_size,
        args.output,
        args.device,
        args.smooth,
        args.max_sequence_length,
        args.chunk_overlap,
        args.embedding_cache_dir,
        args.precompute_only,
    )


if __name__ == "__main__":
    main()
