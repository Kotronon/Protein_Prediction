from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_global_predictor_behavior import (  # noqa: E402
    annotation_comparison,
    binned_predictor_scores,
    cluster_order,
    length_effect_tables,
    pairwise_matrix,
    qc_against_reference,
)
from compare_predictors import PredictionRecord  # noqa: E402


def record(protein_id: str, sequence: str, scores: list[float]) -> PredictionRecord:
    return PredictionRecord(protein_id, sequence, np.asarray(scores, dtype=float))


class GlobalBehaviorTests(unittest.TestCase):
    def test_pairwise_matrix_and_cluster_keep_similar_predictors_adjacent(self) -> None:
        pairwise = pd.DataFrame(
            [
                {"predictor_a": "a", "predictor_b": "b", "residue_spearman": 0.9},
                {"predictor_a": "a", "predictor_b": "c", "residue_spearman": 0.1},
                {"predictor_a": "b", "predictor_b": "c", "residue_spearman": 0.2},
            ]
        )
        matrix = pairwise_matrix(pairwise, "residue_spearman")
        self.assertEqual(matrix.loc["a", "b"], matrix.loc["b", "a"])
        order = cluster_order(matrix)
        self.assertEqual(abs(order.index("a") - order.index("b")), 1)

    def test_qc_detects_length_and_sequence_mismatches(self) -> None:
        reference = {
            "p1": record("p1", "ABC", [1, 2, 3]),
            "p2": record("p2", "DE", [1, 2]),
        }
        predictor = {
            "p1": record("p1", "AB", [1, 2]),
            "p2": record("p2", "DX", [1, 2]),
            "p3": record("p3", "Q", [1]),
        }
        row = qc_against_reference(reference, predictor, "test", {})
        self.assertEqual(row["common_proteins_with_reference"], 2)
        self.assertEqual(row["length_mismatched_common_proteins"], 1)
        self.assertEqual(row["overlap_sequence_mismatched_proteins"], 1)
        self.assertEqual(row["exact_sequence_mismatched_proteins"], 2)

    def test_length_tables_report_all_bins_and_sample_sizes(self) -> None:
        index = ["short", "medium", "long", "very_long"]
        means = pd.DataFrame(
            {"trizod": [0.1, 0.2, 0.3, 0.4], "DisPredict3": [0.4, 0.3, 0.2, 0.1]},
            index=index,
        )
        lengths = pd.Series([100, 300, 700, 1500], index=index, name="protein_length")
        effects, bins, agreements, proteins = length_effect_tables(means, lengths)
        self.assertEqual(bins["n_proteins"].tolist(), [1, 1, 1, 1])
        self.assertEqual(len(agreements), 4)
        self.assertEqual(len(proteins), 4)
        self.assertTrue((effects["n_proteins"] == 4).all())
        score_bins = binned_predictor_scores(means, lengths)
        self.assertEqual(len(score_bins), 8)
        self.assertTrue((score_bins["n_proteins"] == 1).all())

    def test_annotation_comparison_uses_auroc_for_disprot(self) -> None:
        pairwise = pd.DataFrame(
            [
                {
                    "predictor_a": "trizod",
                    "predictor_b": "disprot",
                    "residue_spearman": 0.7,
                }
            ]
        )
        annotation = pd.DataFrame(
            [
                {
                    "dataset_a": "trizod",
                    "dataset_b": "disprot",
                    "comparison_level": "exact",
                    "metric": "spearman",
                    "value": 0.1,
                    "n_proteins_overlap": 2,
                    "n_residues_compared": 10,
                },
                {
                    "dataset_a": "trizod",
                    "dataset_b": "disprot",
                    "comparison_level": "exact",
                    "metric": "auroc",
                    "value": 0.8,
                    "n_proteins_overlap": 2,
                    "n_residues_compared": 10,
                },
            ]
        )
        rows, summary = annotation_comparison(pairwise, annotation)
        selected = rows[(rows["dataset_a"] == "disprot") & (rows["dataset_b"] == "trizod")].iloc[0]
        self.assertEqual(selected["annotation_metric"], "auroc")
        self.assertEqual(selected["annotation_agreement"], 0.8)
        self.assertTrue(math.isnan(summary["all_primary_pair_spearman"]))


if __name__ == "__main__":
    unittest.main()
