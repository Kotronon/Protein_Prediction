#!/usr/bin/env python3
"""Compare per-residue predictor outputs on the same proteome.

The default input is the UdonPred human-proteome output layout:

    results/human_proteome/UdonPred/{trizod,chezod,...}/*.caid

Outputs:
* pairwise_agreement.csv: residue-level and protein-level agreement per pair
* contested_regions.csv: windows where scaled predictor scores disagree most strongly
* predictor_vs_annotation_agreement.csv: predictor agreement next to annotation agreement

SETH: https://github.com/DagmarIlz/SETH/tree/main
ADOPT: https://github.com/PeptoneLtd/ADOPT
IUPred3: requires form to sen request (https://iupred3.elte.hu/download_new)
metapredict: https://github.com/idptools/metapredict

python scripts/compare_predictors.py \
  --predictor trizod=results/human_proteome/UdonPred/trizod \
  --predictor chezod=results/human_proteome/UdonPred/chezod \
  --predictor softdis=results/human_proteome/UdonPred/softdis \
  --predictor pdbflex=results/human_proteome/UdonPred/pdbflex \
  --predictor atlas=results/human_proteome/UdonPred/atlas \
  --predictor plddt=results/human_proteome/UdonPred/plddt \
  --predictor disprot=results/human_proteome/UdonPred/disprot \
  --predictor SETH=results/human_proteome/SETH/seth_human_proteome.caid \
  --predictor IUPred3=results/human_proteome/IUPred3 \
  --predictor ADOPT=results/human_proteome/ADOPT\
  --predictor metapredict=results/human_proteome/metapredict/metapredict_human_proteome.caid \
  --output-dir results/compare_predictors_with_seth_and_iupred3_and_adopt_and_metapredict   

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


DEFAULT_NEGATED_PREDICTORS = {"chezod", "plddt", "ADOPT"}


@dataclass(frozen=True)
class PredictionRecord:
    protein_id: str
    sequence: str
    scores: np.ndarray


PredictionsByProtein = dict[str, PredictionRecord]
PredictionsByPredictor = dict[str, PredictionsByProtein]
PredictorPaths = dict[str, Path]


def extract_uniprot_accession(header: str) -> str:
    """Return UniProt accession from a FASTA/CAID header when possible."""
    header = header.lstrip(">").strip()
    first_token = header.split()[0] if header else ""
    parts = first_token.split("|")
    if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return first_token


def read_caid_records(path: Path) -> list[PredictionRecord]:
    """Read one or more CAID records from a file.

    UdonPred writes one CAID file per protein. Some external predictors, such
    as SETH, write all proteins into one CAID-like file with repeated FASTA
    headers. This parser supports both layouts.
    """
    records: list[PredictionRecord] = []
    protein_id: str | None = None
    residues: list[str] = []
    scores: list[float] = []

    def flush_record() -> None:
        nonlocal protein_id, residues, scores
        if protein_id is None:
            return
        if not scores:
            raise ValueError(f"{path}: no residue scores found for {protein_id}")
        records.append(
            PredictionRecord(
                protein_id=protein_id,
                sequence="".join(residues),
                scores=np.asarray(scores, dtype=np.float64),
            )
        )
        protein_id = None
        residues = []
        scores = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                protein_id = extract_uniprot_accession(line)
                continue
            if protein_id is None:
                raise ValueError(f"{path}:{line_number}: residue row found before first header")

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: expected index, residue, score")
            residues.append(parts[1])
            try:
                scores.append(float(parts[2]))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: expected numeric score in column 3, got {parts[2]!r}"
                ) from exc

    flush_record()
    if not records:
        raise ValueError(f"{path}: no CAID records found")
    return records


def load_predictions(path: Path) -> PredictionsByProtein:
    """Load one CAID file or all CAID files below a predictor directory."""
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.caid"))
    else:
        raise FileNotFoundError(f"Prediction path not found: {path}")

    if not files:
        raise ValueError(f"No *.caid files found in {path}")

    predictions: PredictionsByProtein = {}
    for caid_file in files:
        for record in read_caid_records(caid_file):
            if record.protein_id in predictions:
                raise ValueError(f"Duplicate protein id {record.protein_id!r} in {path}")
            predictions[record.protein_id] = record
    return predictions


def discover_predictor_paths(predictions_root: Path) -> PredictorPaths:
    """Discover immediate child directories that contain CAID files."""
    if not predictions_root.is_dir():
        raise FileNotFoundError(f"Predictions root not found: {predictions_root}")

    predictor_paths = {
        child.name: child
        for child in sorted(predictions_root.iterdir())
        if child.is_dir() and any(child.glob("*.caid"))
    }
    if len(predictor_paths) < 2:
        raise ValueError(
            f"Need at least two predictor directories with *.caid files under {predictions_root}"
        )
    return predictor_paths


def parse_predictor_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--predictor values must use NAME=PATH, for example trizod=results/.../trizod"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Predictor name must not be empty")
    return name, Path(path)


def maybe_negate_scores(
    predictions: PredictionsByProtein,
    predictor_name: str,
    negated_predictors: set[str],
) -> PredictionsByProtein:
    """Flip predictors whose native direction is lower = more disorder."""
    if predictor_name.lower() not in negated_predictors:
        return predictions

    return {
        protein_id: PredictionRecord(
            protein_id=record.protein_id,
            sequence=record.sequence,
            scores=-record.scores,
        )
        for protein_id, record in predictions.items()
    }


def drop_length_mismatched_proteins(
    predictions: PredictionsByPredictor,
) -> PredictionsByPredictor:
    """Keep only proteins whose score length is identical across all predictors."""
    common_ids = set.intersection(*(set(values) for values in predictions.values()))
    keep_ids: set[str] = set()
    dropped = 0

    for protein_id in common_ids:
        lengths = {
            predictor_name: len(predictor_predictions[protein_id].scores)
            for predictor_name, predictor_predictions in predictions.items()
        }
        if len(set(lengths.values())) == 1:
            keep_ids.add(protein_id)
        else:
            dropped += 1

    filtered = {
        predictor_name: {
            protein_id: record
            for protein_id, record in predictor_predictions.items()
            if protein_id in keep_ids
        }
        for predictor_name, predictor_predictions in predictions.items()
    }
    print(
        "Dropped "
        f"{dropped} proteins with predictor length mismatches; "
        f"kept {len(keep_ids)} full-length matched proteins"
    )
    return filtered


def finite_pair_values(
    left: PredictionsByProtein,
    right: PredictionsByProtein,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return matched residue values and matched per-protein mean values."""
    residue_left: list[np.ndarray] = []
    residue_right: list[np.ndarray] = []
    protein_left: list[float] = []
    protein_right: list[float] = []
    common_ids = sorted(set(left) & set(right))

    for protein_id in common_ids:
        left_record = left[protein_id]
        right_record = right[protein_id]
        length = min(len(left_record.scores), len(right_record.scores))
        if length == 0:
            continue
        if len(left_record.scores) != len(right_record.scores):
            print(
                f"Warning: length mismatch for {protein_id}; using first {length} residues",
                file=sys.stderr,
            )

        x = left_record.scores[:length]
        y = right_record.scores[:length]
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue

        x = x[mask]
        y = y[mask]
        residue_left.append(x)
        residue_right.append(y)
        protein_left.append(float(np.mean(x)))
        protein_right.append(float(np.mean(y)))

    if residue_left:
        flat_left = np.concatenate(residue_left)
        flat_right = np.concatenate(residue_right)
    else:
        flat_left = np.asarray([], dtype=np.float64)
        flat_right = np.asarray([], dtype=np.float64)

    return (
        flat_left,
        flat_right,
        np.asarray(protein_left, dtype=np.float64),
        np.asarray(protein_right, dtype=np.float64),
        len(common_ids),
    )


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return math.nan
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def zscore(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0 or not math.isfinite(std):
        return np.zeros_like(values)
    return (values - mean) / std


def compute_pairwise_agreements(predictions: PredictionsByPredictor) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_name, right_name in combinations(sorted(predictions), 2):
        residue_left, residue_right, protein_left, protein_right, common_proteins = (
            finite_pair_values(predictions[left_name], predictions[right_name])
        )
        rows.append(
            {
                "predictor_a": left_name,
                "predictor_b": right_name,
                "common_proteins": common_proteins,
                "matched_residues": len(residue_left),
                "residue_spearman": correlation(residue_left, residue_right, "spearman"),
                "residue_pearson": correlation(residue_left, residue_right, "pearson"),
                "residue_zscore_mae": float(
                    np.mean(np.abs(zscore(residue_left) - zscore(residue_right)))
                )
                if len(residue_left)
                else math.nan,
                "protein_spearman": correlation(protein_left, protein_right, "spearman"),
                "protein_pearson": correlation(protein_left, protein_right, "pearson"),
                "protein_zscore_mae": float(
                    np.mean(np.abs(zscore(protein_left) - zscore(protein_right)))
                )
                if len(protein_left)
                else math.nan,
            }
        )
    return rows


def scale_predictions(
    predictions: PredictionsByPredictor,
    method: str,
) -> PredictionsByPredictor:
    if method == "raw":
        return predictions
    if method != "zscore":
        raise ValueError(f"Unknown contested scaling method: {method}")

    scaled: PredictionsByPredictor = {}
    for predictor_name, predictor_predictions in predictions.items():
        finite_scores = np.concatenate(
            [
                record.scores[np.isfinite(record.scores)]
                for record in predictor_predictions.values()
                if np.any(np.isfinite(record.scores))
            ]
        )
        mean = float(np.mean(finite_scores))
        std = float(np.std(finite_scores))
        if std == 0 or not math.isfinite(std):
            std = 1.0
        scaled[predictor_name] = {
            protein_id: PredictionRecord(
                protein_id=record.protein_id,
                sequence=record.sequence,
                scores=(record.scores - mean) / std,
            )
            for protein_id, record in predictor_predictions.items()
        }
    return scaled


def compute_contested_regions(
    predictions: PredictionsByPredictor,
    window_size: int,
    top_n: int,
) -> list[dict[str, object]]:
    predictor_names = sorted(predictions)
    common_ids = set.intersection(*(set(values) for values in predictions.values()))
    rows: list[dict[str, object]] = []

    for protein_id in sorted(common_ids):
        records = [predictions[name][protein_id] for name in predictor_names]
        length = min(len(record.scores) for record in records)
        if length == 0:
            continue
        score_matrix = np.vstack([record.scores[:length] for record in records])
        finite_columns = np.all(np.isfinite(score_matrix), axis=0)
        if not np.any(finite_columns):
            continue

        residue_std = np.full(length, np.nan, dtype=np.float64)
        residue_range = np.full(length, np.nan, dtype=np.float64)
        residue_std[finite_columns] = np.std(score_matrix[:, finite_columns], axis=0)
        residue_range[finite_columns] = np.ptp(score_matrix[:, finite_columns], axis=0)

        for start in range(0, length, window_size):
            end = min(start + window_size, length)
            window_std = residue_std[start:end]
            if np.all(np.isnan(window_std)):
                continue
            center = start + int(np.nanargmax(window_std))
            row: dict[str, object] = {
                "protein_id": protein_id,
                "start_residue": start + 1,
                "end_residue": end,
                "window_size": end - start,
                "mean_std": float(np.nanmean(window_std)),
                "max_std": float(np.nanmax(window_std)),
                "max_range": float(np.nanmax(residue_range[start:end])),
                "max_disagreement_residue": center + 1,
                "mean_score": float(np.nanmean(score_matrix[:, start:end])),
            }
            for name, record in zip(predictor_names, records):
                row[f"{name}_scaled_score_at_max"] = float(record.scores[center])
            rows.append(row)

    rows.sort(key=lambda row: (row["mean_std"], row["max_std"]), reverse=True)
    return rows[:top_n]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.6g}" if isinstance(value, float) and math.isfinite(value) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def normalize_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left.lower(), right.lower())))





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("results/human_proteome/UdonPred"),
        help="Directory containing one subdirectory per predictor/model.",
    )
    parser.add_argument(
        "--predictor",
        action="append",
        type=parse_predictor_argument,
        help="Explicit predictor as NAME=PATH. Repeat to compare external predictors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/compare_predictors"),
        help="Where to write comparison CSV files.",
    )
    parser.add_argument(
        "--negate",
        nargs="*",
        default=sorted(DEFAULT_NEGATED_PREDICTORS),
        help="Predictor names whose scores should be multiplied by -1 before comparison.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Residues per contested-region window.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of contested windows to write.",
    )
    parser.add_argument(
        "--contested-scale",
        choices=["zscore", "raw"],
        default="zscore",
        help="Scale used before ranking contested regions. zscore avoids raw-score scale artifacts.",
    )
    parser.add_argument(
        "--drop-length-mismatches",
        action="store_true",
        help=(
            "Exclude proteins whose score lengths differ across any loaded predictor. "
            "Use this to remove ADOPT-truncated long proteins from the comparison."
        ),
    )
    parser.add_argument(
        "--annotation-ceiling-csv",
        type=Path,
        default=Path("results/annotation_ceiling/annotation_ceiling_summary.csv"),
        help="Existing annotation-ceiling summary to compare against. Missing file is skipped.",
    )
    parser.add_argument(
        "--annotation-metric",
        default="spearman",
        help="Annotation metric to join into predictor_vs_annotation_agreement.csv.",
    )
    parser.add_argument(
        "--recompute-annotation-ceiling",
        action="store_true",
        help="Recompute annotation agreement from UdonPred/data/*/test.jsonl before joining.",
    )
    parser.add_argument(
        "--udonpred-dir",
        type=Path,
        default=Path("UdonPred"),
        help="UdonPred checkout containing data/*/test.jsonl for annotation ceilings.",
    )
    parser.add_argument(
        "--annotation-output-dir",
        type=Path,
        default=Path("results/annotation_ceiling"),
        help="Where recomputed annotation-ceiling files are written.",
    )
    parser.add_argument(
        "--annotation-datasets",
        nargs="+",
        default=None,
        help="Datasets to use when recomputing annotation ceilings. Defaults to all UdonPred datasets.",
    )
    parser.add_argument(
        "--annotation-threshold",
        type=float,
        default=0.5,
        help="Threshold used for continuous-vs-binary annotation metrics.",
    )
    parser.add_argument(
        "--annotation-min-residues",
        type=int,
        default=10,
        help="Minimum comparable residues required for an annotation overlap.",
    )
    parser.add_argument(
        "--verbose-annotations",
        action="store_true",
        help="Print per-pair annotation overlap counts while recomputing ceilings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")

    if args.predictor:
        predictor_paths = dict(args.predictor)
        if len(predictor_paths) < 2:
            raise ValueError("Pass at least two --predictor NAME=PATH values")
    else:
        predictor_paths = discover_predictor_paths(args.predictions_root)

    negated_predictors = {name.lower() for name in args.negate}
    predictions: PredictionsByPredictor = {}
    for name, path in predictor_paths.items():
        loaded = load_predictions(path)
        predictions[name] = maybe_negate_scores(loaded, name, negated_predictors)
        print(f"Loaded {name}: {len(loaded)} proteins from {path}")

    if args.drop_length_mismatches:
        predictions = drop_length_mismatched_proteins(predictions)

    pairwise_rows = compute_pairwise_agreements(predictions)
    
    contested_predictions = scale_predictions(predictions, args.contested_scale)
    contested_rows = compute_contested_regions(
        contested_predictions,
        window_size=args.window_size,
        top_n=args.top_n,
    )

    pairwise_path = args.output_dir / "pairwise_agreement.csv"
    contested_path = args.output_dir / "contested_regions.csv"
    write_csv(pairwise_rows, pairwise_path)
    write_csv(contested_rows, contested_path)

    print(f"Wrote {pairwise_path}")
    print(f"Wrote {contested_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
