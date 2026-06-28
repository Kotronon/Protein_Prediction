#!/usr/bin/env python3
"""Create DisProt-focused UdonPred retraining configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


BASE_CONFIG_DIR = Path("UdonPred/config_caid4/caid4_disprot_trizod_updated_focus")
OUTPUT_ROOT = Path("UdonPred/config_caid4")
HELPER_DATASETS = ("chezod", "plddt", "softdis", "trizod_updated")


def read_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def write_common_files(output_dir: Path, run_name: str) -> None:
    config = read_yaml(BASE_CONFIG_DIR / "config.yaml")
    config["run_name"] = run_name
    write_yaml(output_dir / "config.yaml", config)
    write_yaml(output_dir / "architecture.yaml", read_yaml(BASE_CONFIG_DIR / "architecture.yaml"))
    write_yaml(output_dir / "optimise.yaml", read_yaml(BASE_CONFIG_DIR / "optimise.yaml"))
    (output_dir / "run_command.txt").write_text(
        "cd UdonPred\n"
        f"uv run python run.py train --config-dir config_caid4/{run_name} "
        "--output-dir checkpoints_caid4 --optimised-parameters-dir optimised_parameters_caid4\n"
    )


def make_disprot_only(base_data: dict) -> dict:
    data = copy.deepcopy(base_data)
    for dataset_name, dataset_config in data.items():
        dataset_config["fraction"] = 1 if dataset_name == "disprot" else 0
    return data


def make_weighted_multitask(base_data: dict) -> dict:
    data = copy.deepcopy(base_data)
    for dataset_name, dataset_config in data.items():
        if dataset_name == "disprot":
            dataset_config["fraction"] = 1
            for losses in dataset_config["losses"].values():
                for loss in losses:
                    loss["weight"] = 5
        elif dataset_name in HELPER_DATASETS:
            dataset_config["fraction"] = 1
            for losses in dataset_config["losses"].values():
                for loss in losses:
                    loss["weight"] = 0.1
        else:
            dataset_config["fraction"] = 0
    return data


def create_config(name: str, data: dict) -> None:
    output_dir = OUTPUT_ROOT / name
    write_common_files(output_dir, name)
    write_yaml(output_dir / "data.yaml", data)
    print(f"Wrote {output_dir}")


def main() -> None:
    base_data = read_yaml(BASE_CONFIG_DIR / "data.yaml")
    create_config("caid4_disprot_only", make_disprot_only(base_data))
    create_config("caid4_disprot_weighted_multitask", make_weighted_multitask(base_data))


if __name__ == "__main__":
    main()
