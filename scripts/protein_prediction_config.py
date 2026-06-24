"""Shared project constants used by analysis and runner scripts."""

from __future__ import annotations


UDONPRED_DATASETS = (
    "trizod",
    "chezod",
    "softdis",
    "pdbflex",
    "atlas",
    "plddt",
    "disprot",
)

# These targets use the opposite score direction in UdonPred evaluation files.
NEGATED_UDONPRED_DATASETS = frozenset({"chezod", "plddt"})

UDONPRED_DISORDER_MODELS = tuple(
    dataset for dataset in UDONPRED_DATASETS if dataset != "pdbflex"
)
