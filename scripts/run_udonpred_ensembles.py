#!/usr/bin/env python3
"""Fit and evaluate validation-trained UdonPred head ensembles.

The script turns ``notebooks/04_ensembles.ipynb`` into a reproducible CLI
step. It uses validation-set predictions for model selection/fitting and only
evaluates the resulting strategies on the held-out test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = [
    "trizod",
    "trizod_updated",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot",
]
NEGATED_DATASETS = {"chezod", "plddt"}
MASK_VALUE = 999.0
METRIC_COLUMNS = [
    "trizod",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot\n(AP)",
    "disprot\n(AUROC)",
]


def metric_columns_for_datasets(datasets: list[str]) -> list[str]:
    columns = []
    for dataset in datasets:
        columns.extend(metric_columns_for_dataset(dataset))
    return columns


def resolve_active_datasets(excluded: list[str], replace_trizod_with_updated: bool) -> list[str]:
    excluded_set = set(excluded)
    unknown = sorted(excluded_set - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets in --exclude-datasets: {', '.join(unknown)}")
    base = ["trizod_updated" if replace_trizod_with_updated else "trizod"]
    base.extend(["chezod", "softdis", "pdbflex", "atlas", "plddt", "disprot"])
    datasets = [dataset for dataset in base if dataset not in excluded_set]
    if not datasets:
        raise ValueError("At least one dataset must remain after --exclude-datasets")
    return datasets


def resolve_focus_datasets(focus_datasets: list[str], active_datasets: list[str]) -> list[str]:
    if not focus_datasets:
        return active_datasets
    inactive = [dataset for dataset in focus_datasets if dataset not in active_datasets]
    if inactive:
        raise ValueError(
            "Focus datasets must also be active evaluation/input datasets. "
            f"Inactive focus datasets: {', '.join(inactive)}"
        )
    return focus_datasets


def dataset_label(focus_datasets: list[str], active_datasets: list[str]) -> str:
    return "all" if focus_datasets == active_datasets else "+".join(focus_datasets)


def read_jsonl_records(path: Path) -> dict[str, dict[str, object]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            records[str(raw["id"])] = {
                "sequence": str(raw["x_0"]),
                "labels": np.asarray(raw["y"], dtype=np.float64),
            }
    return records


def normalize_prediction_id(protein_id: str) -> str:
    protein_id = protein_id.strip().lstrip(">").split()[0]
    if len(protein_id) >= 6 and protein_id.isdigit():
        base_len = len(protein_id) - 3
        return protein_id[:base_len] + "_" + "_".join(protein_id[base_len:])
    return protein_id


def read_caid_dir(input_dir: Path) -> dict[str, np.ndarray]:
    predictions = {}
    for path in sorted(input_dir.glob("*.caid")):
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        if not lines:
            raise ValueError(f"Empty prediction file: {path}")
        scores = [float(line.split()[2]) for line in lines[1:] if line.strip()]
        predictions[normalize_prediction_id(lines[0])] = np.asarray(scores, dtype=np.float64)
    if not predictions:
        raise FileNotFoundError(f"No CAID predictions found in {input_dir}")
    return predictions


def count_fasta_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def prediction_dir_complete(result_dir: Path, fasta_path: Path) -> bool:
    return result_dir.exists() and len(list(result_dir.glob("*.caid"))) == count_fasta_records(
        fasta_path
    )


def labels_in_disorder_direction(labels: np.ndarray, dataset: str) -> np.ndarray:
    return -labels if dataset in NEGATED_DATASETS else labels


def preds_in_disorder_direction(preds: np.ndarray, train_dataset: str) -> np.ndarray:
    return -preds if train_dataset in NEGATED_DATASETS else preds


def metric_columns_for_dataset(dataset: str) -> list[str]:
    return ["disprot\n(AP)", "disprot\n(AUROC)"] if dataset == "disprot" else [dataset]


def primary_metric(dataset: str) -> str:
    return "disprot\n(AP)" if dataset == "disprot" else dataset


def evaluate_vector(labels: np.ndarray, preds: np.ndarray, dataset: str) -> dict[str, float]:
    if len(labels) == 0:
        return {column: math.nan for column in metric_columns_for_dataset(dataset)}
    if dataset == "disprot":
        binary_labels = labels.astype(int)
        return {
            "disprot\n(AP)": float(average_precision_score(binary_labels, preds)),
            "disprot\n(AUROC)": float(roc_auc_score(binary_labels, preds)),
        }
    if np.unique(labels).size < 2 or np.unique(preds).size < 2:
        return {dataset: math.nan}
    return {dataset: float(spearmanr(preds, labels).statistic)}


def prediction_command(
    udonpred_dir: Path,
    fasta_path: Path,
    weights_dir: Path,
    train_dataset: str,
    result_dir: Path,
    device: str,
    batch_size: int,
    smooth: float,
) -> list[str]:
    if shutil.which("uv"):
        return [
            "uv",
            "run",
            "predict.py",
            str(fasta_path),
            str(weights_dir),
            "--target",
            train_dataset,
            "--output",
            str(result_dir),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--smooth",
            str(smooth),
        ]
    return [
        sys.executable,
        "predict.py",
        str(fasta_path),
        str(weights_dir),
        "--target",
        train_dataset,
        "--output",
        str(result_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--smooth",
        str(smooth),
    ]


def run_missing_predictions(
    args: argparse.Namespace,
    split: str,
    prediction_root: Path,
    active_datasets: list[str],
    skip_predictions: bool,
) -> None:
    jobs = []
    for train_dataset in active_datasets:
        for target_dataset in active_datasets:
            fasta_path = args.udonpred_dir / "data" / target_dataset / f"{split}.fasta"
            result_dir = prediction_root / f"{train_dataset}_{target_dataset}"
            if args.force_validation or not prediction_dir_complete(result_dir, fasta_path):
                jobs.append((train_dataset, target_dataset, fasta_path, result_dir))

    print(f"Missing or forced {split} prediction jobs: {len(jobs)}")
    if skip_predictions and jobs:
        first = jobs[0][3]
        raise FileNotFoundError(
            f"{split} predictions are incomplete, starting with {first}. "
            f"Rerun without --skip-{split}-predictions to generate them."
        )

    for train_dataset, _target_dataset, fasta_path, result_dir in jobs:
        result_dir.mkdir(parents=True, exist_ok=True)
        cmd = prediction_command(
            args.udonpred_dir,
            fasta_path,
            args.weights_dir,
            train_dataset,
            result_dir,
            args.device,
            args.batch_size,
            args.smooth,
        )
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=args.udonpred_dir, check=True)


def load_aligned_stack(
    udonpred_dir: Path,
    split: str,
    prediction_root: Path,
    target_dataset: str,
    train_datasets: list[str],
) -> dict[str, object]:
    records = read_jsonl_records(udonpred_dir / "data" / target_dataset / f"{split}.jsonl")
    pred_maps = {
        train_dataset: read_caid_dir(prediction_root / f"{train_dataset}_{target_dataset}")
        for train_dataset in train_datasets
    }

    label_chunks = []
    pred_chunks_by_train = {train_dataset: [] for train_dataset in train_datasets}
    residue_count = 0
    for protein_id, record in records.items():
        labels = np.asarray(record["labels"], dtype=np.float64)
        mask = np.isfinite(labels) & (labels != MASK_VALUE)
        if not np.any(mask):
            continue
        for train_dataset, pred_map in pred_maps.items():
            if protein_id not in pred_map:
                raise ValueError(f"Missing prediction for {protein_id} in {train_dataset}_{target_dataset}")
            preds = pred_map[protein_id]
            if len(preds) != len(labels):
                raise ValueError(
                    f"{train_dataset}_{target_dataset}/{protein_id}: "
                    f"prediction length {len(preds)} != label length {len(labels)}"
                )
            pred_chunks_by_train[train_dataset].append(
                preds_in_disorder_direction(preds, train_dataset)[mask]
            )
        label_chunks.append(labels_in_disorder_direction(labels, target_dataset)[mask])
        residue_count += int(mask.sum())

    x = np.column_stack(
        [np.concatenate(pred_chunks_by_train[train_dataset]) for train_dataset in train_datasets]
    )
    y = np.concatenate(label_chunks)
    finite_rows = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if not np.all(finite_rows):
        dropped = int((~finite_rows).sum())
        print(f"Dropping {dropped} non-finite aligned residues for {split}/{target_dataset}")
        x = x[finite_rows]
        y = y[finite_rows]
    return {"dataset": target_dataset, "split": split, "X": x, "y": y, "n_residues": residue_count}


def load_split_stacks(
    udonpred_dir: Path,
    split: str,
    prediction_root: Path,
    train_datasets: list[str],
    target_datasets: list[str],
) -> dict[str, dict[str, object]]:
    return {
        target_dataset: load_aligned_stack(
            udonpred_dir, split, prediction_root, target_dataset, train_datasets
        )
        for target_dataset in target_datasets
    }


def score_individual_heads(
    stacks: dict[str, dict[str, object]],
    train_datasets: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for train_index, train_dataset in enumerate(train_datasets):
        row = {"train_dataset": train_dataset}
        for target_dataset, stack in stacks.items():
            row.update(evaluate_vector(stack["y"], stack["X"][:, train_index], target_dataset))
        rows.append(row)
    return pd.DataFrame(rows).set_index("train_dataset")[metric_columns]


def concatenate_stacks(
    stacks: list[dict[str, object]],
    max_fit_residues: int,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([stack["X"] for stack in stacks])
    if columns is not None:
        x = x[:, columns]
    y = np.concatenate([stack["y"] for stack in stacks])
    if max_fit_residues and len(y) > max_fit_residues:
        indices = rng.choice(len(y), size=max_fit_residues, replace=False)
        x = x[indices]
        y = y[indices]
    return x, y


def fit_convex_weights(
    stacks: list[dict[str, object]],
    max_fit_residues: int,
    rng: np.random.Generator,
    columns: tuple[int, ...] | None = None,
) -> np.ndarray:
    x, y = concatenate_stacks(stacks, max_fit_residues, rng, columns)
    n_heads = x.shape[1]
    start = np.full(n_heads, 1.0 / n_heads)
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_heads

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.mean(residual * residual))

    result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Convex-weight optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    weights[weights < 1e-10] = 0.0
    return weights / weights.sum()


def fit_ridge_model(
    stacks: list[dict[str, object]],
    alpha: float,
    max_fit_residues: int,
    rng: np.random.Generator,
) -> Ridge:
    x, y = concatenate_stacks(stacks, max_fit_residues, rng)
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return model


def iter_head_subsets(n_heads: int, max_subset_size: int) -> list[tuple[int, ...]]:
    if n_heads < 2:
        return []
    upper = n_heads if max_subset_size <= 0 else min(max_subset_size, n_heads)
    return [
        subset
        for size in range(2, upper + 1)
        for subset in combinations(range(n_heads), size)
    ]


def score_validation_prediction(
    stack: dict[str, object], preds: np.ndarray, dataset: str
) -> float:
    score = evaluate_vector(stack["y"], preds, dataset)[primary_metric(dataset)]
    return float(score) if np.isfinite(score) else -math.inf


def select_subset_mean(
    stacks: list[dict[str, object]],
    datasets: list[str],
    subsets: list[tuple[int, ...]],
) -> dict[str, object]:
    best: dict[str, object] = {"subset": subsets[0], "validation_score": -math.inf}
    for subset in subsets:
        scores = [
            score_validation_prediction(stack, stack["X"][:, subset].mean(axis=1), dataset)
            for stack, dataset in zip(stacks, datasets)
        ]
        score = float(np.mean(scores))
        if score > float(best["validation_score"]):
            best = {"subset": subset, "validation_score": score}
    return best


def select_subset_convex(
    stacks: list[dict[str, object]],
    datasets: list[str],
    subsets: list[tuple[int, ...]],
    max_fit_residues: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    best: dict[str, object] = {
        "subset": subsets[0],
        "weights": np.full(len(subsets[0]), 1.0 / len(subsets[0])),
        "validation_score": -math.inf,
    }
    for subset in subsets:
        weights = fit_convex_weights(stacks, max_fit_residues, rng, subset)
        scores = [
            score_validation_prediction(stack, stack["X"][:, subset] @ weights, dataset)
            for stack, dataset in zip(stacks, datasets)
        ]
        score = float(np.mean(scores))
        if score > float(best["validation_score"]):
            best = {"subset": subset, "weights": weights, "validation_score": score}
    return best


def write_subset_choices(
    output_dir: Path,
    train_datasets: list[str],
    choices: list[dict[str, object]],
) -> None:
    rows = []
    for choice in choices:
        subset = tuple(choice["subset"])
        weights = choice.get("weights")
        if weights is None:
            weights = np.full(len(subset), 1.0 / len(subset))
        rows.append(
            {
                "strategy": choice["strategy"],
                "target_dataset": choice["target_dataset"],
                "validation_score": float(choice["validation_score"]),
                "heads": "+".join(train_datasets[index] for index in subset),
                **{
                    train_datasets[index]: float(weight)
                    for index, weight in zip(subset, np.asarray(weights, dtype=np.float64))
                },
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "ensemble_subset_choices.csv", index=False, float_format="%.6f"
    )


def write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, float_format="%.6f")


def write_weights(
    output_dir: Path,
    train_datasets: list[str],
    selected_head_by_target: dict[str, str],
    global_convex_weights: np.ndarray,
    per_target_convex_weights: dict[str, np.ndarray],
    global_ridge: Ridge,
    per_target_ridge: dict[str, Ridge],
    global_fit_label: str,
) -> None:
    rows = [
        {
            "strategy": "global_convex_validation",
            "target_dataset": global_fit_label,
            "intercept": 0.0,
            **dict(zip(train_datasets, global_convex_weights)),
        },
        {
            "strategy": "global_ridge_stacking_validation",
            "target_dataset": global_fit_label,
            "intercept": float(global_ridge.intercept_),
            **dict(zip(train_datasets, global_ridge.coef_)),
        },
    ]
    for dataset, weights in per_target_convex_weights.items():
        rows.append(
            {
                "strategy": "per_target_convex_validation",
                "target_dataset": dataset,
                "intercept": 0.0,
                **dict(zip(train_datasets, weights)),
            }
        )
    for dataset, model in per_target_ridge.items():
        rows.append(
            {
                "strategy": "per_target_ridge_stacking_validation",
                "target_dataset": dataset,
                "intercept": float(model.intercept_),
                **dict(zip(train_datasets, model.coef_)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "ensemble_weights.csv", index=False, float_format="%.6f")
    pd.Series(selected_head_by_target, name="selected_train_dataset").to_csv(
        output_dir / "validation_selected_heads.csv"
    )


def plot_outputs(ensemble_matrix: pd.DataFrame, ensemble_delta: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw_heatmap(
        frame: pd.DataFrame,
        output: Path,
        title: str,
        cmap: str,
        center_zero: bool = False,
        fmt: str = ".3f",
    ) -> None:
        data = frame.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        if center_zero and len(finite):
            limit = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 1e-6)
            vmin, vmax = -limit, limit
        else:
            vmin = vmax = None
        fig_width = max(10, 0.9 * len(frame.columns) + 4)
        fig_height = max(5, 0.45 * len(frame.index) + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(frame.columns)))
        ax.set_xticklabels([str(column).replace("\n", " ") for column in frame.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(frame.index)))
        ax.set_yticklabels(frame.index)
        ax.set_xlabel("Test dataset / metric")
        ax.set_ylabel("Strategy")
        ax.set_title(title)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                value = data[row, column]
                if np.isfinite(value):
                    ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)

    draw_heatmap(
        ensemble_matrix,
        output_dir / "ensemble_matrix_heatmap.png",
        "Validation-trained UdonPred ensemble performance",
        "viridis",
    )
    draw_heatmap(
        ensemble_delta.drop(index="best_individual_test_oracle", errors="ignore"),
        output_dir / "ensemble_delta_vs_best_individual_heatmap.png",
        "Delta versus best individual UdonPred head",
        "coolwarm",
        center_zero=True,
        fmt="+.3f",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udonpred-dir", type=Path, default=Path("UdonPred"))
    parser.add_argument("--weights-dir", type=Path, default=Path("UdonPred/weights"))
    parser.add_argument(
        "--test-prediction-root",
        type=Path,
        default=Path("results/udonpred_matrix/predictions"),
    )
    parser.add_argument(
        "--valid-prediction-root",
        type=Path,
        default=Path("results/udonpred_validation_matrix/predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/ensembles"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--smooth", type=float, default=1.5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument(
        "--max-fit-residues",
        type=int,
        default=0,
        help="Subsample validation residues for fitting; 0 uses all validation residues.",
    )
    parser.add_argument(
        "--max-subset-size",
        type=int,
        default=0,
        help="Largest head subset tried by subset-ensemble search; 0 uses all active heads.",
    )
    parser.add_argument(
        "--ensemble-focus-datasets",
        nargs="*",
        default=[],
        choices=DATASETS,
        help="Fit/select global ensembles only on these validation datasets; default uses all active datasets.",
    )
    parser.add_argument("--force-validation", action="store_true")
    parser.add_argument("--skip-test-predictions", action="store_true")
    parser.add_argument("--skip-validation-predictions", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=[],
        choices=DATASETS,
        help="Exclude datasets both as input heads and as evaluation targets.",
    )
    parser.add_argument(
        "--replace-trizod-with-updated",
        action="store_true",
        help="Use trizod_updated instead of trizod as both input head and target.",
    )
    return parser.parse_args()


def validate_device(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA was requested, but torch is not installed.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False in this Python "
            "environment. Use --device cpu here, or run on an NVIDIA/CUDA machine."
        )


def main() -> None:
    args = parse_args()
    args.udonpred_dir = args.udonpred_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.test_prediction_root = args.test_prediction_root.resolve()
    args.valid_prediction_root = args.valid_prediction_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_device(args.device)
    rng = np.random.default_rng(args.random_seed)
    active_datasets = resolve_active_datasets(
        args.exclude_datasets, args.replace_trizod_with_updated
    )
    focus_datasets = resolve_focus_datasets(args.ensemble_focus_datasets, active_datasets)
    focus_label = dataset_label(focus_datasets, active_datasets)
    metric_columns = metric_columns_for_datasets(active_datasets)

    run_missing_predictions(
        args,
        "test",
        args.test_prediction_root,
        active_datasets,
        args.skip_test_predictions,
    )
    run_missing_predictions(
        args,
        "valid",
        args.valid_prediction_root,
        active_datasets,
        args.skip_validation_predictions,
    )
    test_stacks = load_split_stacks(
        args.udonpred_dir, "test", args.test_prediction_root, active_datasets, active_datasets
    )
    valid_stacks = load_split_stacks(
        args.udonpred_dir, "valid", args.valid_prediction_root, active_datasets, active_datasets
    )

    alignment_summary = pd.DataFrame(
        [
            {
                "split": split,
                "dataset": dataset,
                "n_residues": stack["n_residues"],
                "n_heads": stack["X"].shape[1],
            }
            for split, stacks in (("test", test_stacks), ("valid", valid_stacks))
            for dataset, stack in stacks.items()
        ]
    )
    alignment_summary.to_csv(args.output_dir / "alignment_summary.csv", index=False)
    pd.Series(focus_datasets, name="focus_dataset").to_csv(
        args.output_dir / "ensemble_focus_datasets.csv", index=False
    )

    recomputed_individual = score_individual_heads(test_stacks, active_datasets, metric_columns)
    published_path = args.test_prediction_root.parent / "matrix.csv"
    if published_path.exists():
        published = pd.read_csv(published_path).set_index("train_dataset")
        published = published.reindex(index=active_datasets, columns=metric_columns)
        published = published.apply(pd.to_numeric, errors="coerce")
        max_abs_delta = (recomputed_individual - published).abs().max().max()
        print(f"Max absolute delta versus {published_path}: {max_abs_delta:.6f}")
        if max_abs_delta > 5e-4:
            raise ValueError("Recomputed individual-head matrix does not match the saved matrix.")

    best_individual_scores = recomputed_individual.max(axis=0)
    best_individual_heads = recomputed_individual.idxmax(axis=0)
    valid_individual = score_individual_heads(valid_stacks, active_datasets, metric_columns)
    selected_head_by_target = {
        dataset: valid_individual[primary_metric(dataset)].idxmax() for dataset in active_datasets
    }

    global_fit_stacks = [valid_stacks[dataset] for dataset in focus_datasets]
    global_convex_weights = fit_convex_weights(global_fit_stacks, args.max_fit_residues, rng)
    per_target_convex_weights = {
        dataset: fit_convex_weights([valid_stacks[dataset]], args.max_fit_residues, rng)
        for dataset in active_datasets
    }
    global_ridge = fit_ridge_model(
        global_fit_stacks, args.ridge_alpha, args.max_fit_residues, rng
    )
    per_target_ridge = {
        dataset: fit_ridge_model(
            [valid_stacks[dataset]], args.ridge_alpha, args.max_fit_residues, rng
        )
        for dataset in active_datasets
    }
    subsets = iter_head_subsets(len(active_datasets), args.max_subset_size)
    subset_choices: list[dict[str, object]] = []
    global_subset_mean: dict[str, object] | None = None
    global_subset_convex: dict[str, object] | None = None
    per_target_subset_mean: dict[str, dict[str, object]] = {}
    per_target_subset_convex: dict[str, dict[str, object]] = {}
    if subsets:
        valid_stack_list = [valid_stacks[dataset] for dataset in focus_datasets]
        global_subset_mean = select_subset_mean(valid_stack_list, focus_datasets, subsets)
        global_subset_mean.update(
            {"strategy": "global_subset_mean_validation", "target_dataset": focus_label}
        )
        subset_choices.append(global_subset_mean)

        global_subset_convex = select_subset_convex(
            valid_stack_list, focus_datasets, subsets, args.max_fit_residues, rng
        )
        global_subset_convex.update(
            {"strategy": "global_subset_convex_validation", "target_dataset": focus_label}
        )
        subset_choices.append(global_subset_convex)

        for dataset in active_datasets:
            per_target_subset_mean[dataset] = select_subset_mean(
                [valid_stacks[dataset]], [dataset], subsets
            )
            per_target_subset_mean[dataset].update(
                {"strategy": "per_target_subset_mean_validation", "target_dataset": dataset}
            )
            subset_choices.append(per_target_subset_mean[dataset])

            per_target_subset_convex[dataset] = select_subset_convex(
                [valid_stacks[dataset]],
                [dataset],
                subsets,
                args.max_fit_residues,
                rng,
            )
            per_target_subset_convex[dataset].update(
                {"strategy": "per_target_subset_convex_validation", "target_dataset": dataset}
            )
            subset_choices.append(per_target_subset_convex[dataset])
        write_subset_choices(args.output_dir, active_datasets, subset_choices)
    write_weights(
        args.output_dir,
        active_datasets,
        selected_head_by_target,
        global_convex_weights,
        per_target_convex_weights,
        global_ridge,
        per_target_ridge,
        focus_label,
    )

    def strategy_predictions(strategy: str, dataset: str, stack: dict[str, object]) -> np.ndarray:
        x = stack["X"]
        if strategy == "simple_mean_all_heads":
            return x.mean(axis=1)
        if strategy == "validation_selected_single_head":
            return x[:, active_datasets.index(selected_head_by_target[dataset])]
        if strategy == "global_convex_validation":
            return x @ global_convex_weights
        if strategy == "per_target_convex_validation":
            return x @ per_target_convex_weights[dataset]
        if strategy == "global_ridge_stacking_validation":
            return global_ridge.predict(x)
        if strategy == "per_target_ridge_stacking_validation":
            return per_target_ridge[dataset].predict(x)
        if strategy == "global_subset_mean_validation":
            assert global_subset_mean is not None
            return x[:, global_subset_mean["subset"]].mean(axis=1)
        if strategy == "per_target_subset_mean_validation":
            choice = per_target_subset_mean[dataset]
            return x[:, choice["subset"]].mean(axis=1)
        if strategy == "global_subset_convex_validation":
            assert global_subset_convex is not None
            return x[:, global_subset_convex["subset"]] @ global_subset_convex["weights"]
        if strategy == "per_target_subset_convex_validation":
            choice = per_target_subset_convex[dataset]
            return x[:, choice["subset"]] @ choice["weights"]
        raise ValueError(f"Unknown strategy: {strategy}")

    strategies = [
        "best_individual_test_oracle",
        "simple_mean_all_heads",
        "validation_selected_single_head",
        "global_convex_validation",
        "per_target_convex_validation",
        "global_ridge_stacking_validation",
        "per_target_ridge_stacking_validation",
    ]
    if subsets:
        strategies.extend(
            [
                "global_subset_mean_validation",
                "per_target_subset_mean_validation",
                "global_subset_convex_validation",
                "per_target_subset_convex_validation",
            ]
        )
    rows = []
    for strategy in strategies:
        row = {"strategy": strategy}
        if strategy == "best_individual_test_oracle":
            row.update(best_individual_scores.to_dict())
        else:
            for dataset, stack in test_stacks.items():
                row.update(evaluate_vector(stack["y"], strategy_predictions(strategy, dataset, stack), dataset))
        rows.append(row)

    ensemble_matrix = pd.DataFrame(rows).set_index("strategy")[metric_columns]
    ensemble_delta = ensemble_matrix.subtract(best_individual_scores, axis="columns")
    summary_rows = []
    for strategy in ensemble_matrix.index:
        deltas = ensemble_delta.loc[strategy]
        summary_rows.append(
            {
                "strategy": strategy,
                "mean_score": float(ensemble_matrix.loc[strategy].mean()),
                "mean_delta_vs_best_individual": float(deltas.mean()),
                "wins_vs_best_individual": int((deltas > 1e-6).sum()),
                "ties_vs_best_individual": int((deltas.abs() <= 1e-6).sum()),
                "losses_vs_best_individual": int((deltas < -1e-6).sum()),
                "best_delta": float(deltas.max()),
                "worst_delta": float(deltas.min()),
            }
        )
    ensemble_summary = pd.DataFrame(summary_rows).set_index("strategy")

    write_matrix(recomputed_individual, args.output_dir / "test_individual_matrix.csv")
    write_matrix(valid_individual, args.output_dir / "validation_individual_matrix.csv")
    write_matrix(ensemble_matrix, args.output_dir / "ensemble_matrix.csv")
    write_matrix(ensemble_delta, args.output_dir / "ensemble_delta_vs_best_individual.csv")
    ensemble_summary.to_csv(args.output_dir / "ensemble_summary.csv", float_format="%.6f")
    pd.Series(best_individual_heads, name="best_individual_head").to_csv(
        args.output_dir / "best_individual_heads.csv"
    )
    if not args.skip_plots:
        plot_outputs(ensemble_matrix, ensemble_delta, args.output_dir)
    print(f"Wrote ensemble outputs to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
