#!/usr/bin/env python3
"""Check whether PyTorch can use the GPU for UdonPred runs."""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA GPU is visible to PyTorch. Use --device cpu or fix the CUDA/PyTorch setup.")
        return 1

    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        print(f"GPU {index}: {props.name} ({total_gb:.2f} GB VRAM)")

    device = torch.device("cuda:0")
    x = torch.randn((512, 512), device=device)
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"CUDA tensor test: OK ({float(y[0, 0]):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

