import argparse
import os
from importlib import import_module
from pathlib import Path
from typing import Dict, List

import torch
import yaml


def load_config(checkpoint_dir: str) -> Dict:
    """Load configuration from checkpoint directory."""
    checkpoint_config = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(checkpoint_config):
        raise FileNotFoundError(
            f"Checkpoint config.yaml not found at: {checkpoint_config}"
        )
    with open(checkpoint_config, "r") as f:
        return yaml.safe_load(f)


def load_hyperparameters(config: Dict) -> Dict:
    """Load hyperparameters if specified in config."""
    hyperparam_path = config.get("config", {}).get("hyperparameter_path")
    if hyperparam_path and os.path.exists(hyperparam_path):
        with open(hyperparam_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def collect_output_keys(config: Dict) -> List[str]:
    """Collect output keys from configuration."""
    output_keys = set()
    for _, ds_config in config.get("data", {}).items():
        if ds_config.get("fraction", 0) > 0:
            keys = set()
            if "losses" in ds_config:
                for key_losses in ds_config["losses"].values():
                    for loss in key_losses:
                        keys.add(loss["output"])
            if "metrics" in ds_config:
                for key_metrics in ds_config["metrics"].values():
                    for metric in key_metrics:
                        keys.add(metric["output"])
            output_keys |= keys
    outputs = config.get("config", {}).get("outputs")
    if outputs:
        return list(outputs)
    return sorted(output_keys)


def discover_checkpoints(root: Path) -> List[Path]:
    """Find all checkpoint directories under root (contain pytorch_model.bin)."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        if "pytorch_model.bin" in filenames and "config.yaml" in filenames:
            found.append(Path(dirpath))
    return sorted(found)


def export_checkpoint(
    checkpoint_dir: Path,
    out_base: Path,
) -> None:
    """Export all prediction heads from one checkpoint to ONNX.

    Args:
        checkpoint_dir: Path to the individual checkpoint directory.
        out_base: Base output path (without extension). A single head is saved
                  as ``out_base.onnx``; multiple heads as ``out_base_{name}.onnx``.
    """
    config = load_config(str(checkpoint_dir))

    build_prediction_heads = getattr(
        import_module("model.build_model"), "build_prediction_heads"
    )
    UdonPred = getattr(import_module("model.model"), "UdonPred")

    hyperparameters = load_hyperparameters(config)
    prediction_heads = build_prediction_heads(hyperparameters, config)
    output_keys = set(collect_output_keys(config))

    model = UdonPred(None, prediction_heads, output_keys)

    state_path = checkpoint_dir / "pytorch_model.bin"
    model.load_state_dict(
        torch.load(str(state_path), weights_only=True, map_location="cpu"),
        strict=False,
    )
    model.eval()

    out_base.parent.mkdir(parents=True, exist_ok=True)

    hidden_dim = config["config"]["input_dim"]
    example_embeddings = torch.randn(2, 128, hidden_dim)

    heads = list(model.prediction_heads.items())
    exported_heads = []
    for head_name, prediction_head in heads:
        prediction_head.eval()
        if len(heads) == 1:
            out_path = out_base.with_suffix(".onnx")
        else:
            out_path = out_base.parent / f"{out_base.name}_{head_name}.onnx"

        with torch.no_grad():
            torch.onnx.export(
                prediction_head,
                (example_embeddings,),
                str(out_path),
                dynamo=True,
                input_names=["embedding"],
                output_names=["score"],
                dynamic_shapes=(
                    {0: torch.export.Dim("batch"), 1: torch.export.Dim("seq_len")},
                ),
                external_data=False,
                optimize=True,
            )
        print(f"    Saved → {out_path}")
        exported_heads.append(head_name)

    print(f"    Heads exported: {', '.join(exported_heads)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export prediction heads to ONNX from one or more checkpoints."
    )
    parser.add_argument(
        "checkpoint_root",
        type=str,
        help="Root folder containing checkpoint subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for exported ONNX models.",
    )
    parser.add_argument(
        "--checkpoints",
        "-c",
        nargs="+",
        default=None,
        metavar="REL_PATH",
        help=(
            "Relative paths of specific checkpoints to export "
            "(default: all discovered checkpoints)."
        ),
    )

    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    if not root.is_dir():
        raise ValueError(f"Checkpoint root not found: {root}")

    output_root = Path(args.output_dir)

    if args.checkpoints:
        checkpoints = [root / rel for rel in args.checkpoints]
        for cp in checkpoints:
            if not cp.is_dir():
                raise ValueError(f"Checkpoint directory not found: {cp}")
            if not (cp / "pytorch_model.bin").exists():
                raise ValueError(f"pytorch_model.bin missing in: {cp}")
    else:
        checkpoints = discover_checkpoints(root)
        if not checkpoints:
            raise ValueError(f"No checkpoints found under: {root}")
        print(f"Discovered {len(checkpoints)} checkpoint(s).")

    # Group by parent so we know whether to include the checkpoint suffix.
    from collections import defaultdict

    by_parent: dict = defaultdict(list)
    for cp in checkpoints:
        by_parent[cp.parent].append(cp)

    for checkpoint_dir in checkpoints:
        rel = checkpoint_dir.relative_to(root)
        siblings = by_parent[checkpoint_dir.parent]
        if len(siblings) == 1:
            name = str(rel.parent).replace("/", "_")
        else:
            name = str(rel).replace("/", "_")
        out_base = output_root / name
        print(f"Exporting {rel} → {out_base}.onnx")
        export_checkpoint(checkpoint_dir, out_base)

    print(f"Done. Models saved to {output_root}")


if __name__ == "__main__":
    main()
