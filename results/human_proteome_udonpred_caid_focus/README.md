# Human Proteome UdonPred/CAID Focus

Focused comparison of the six disorder-oriented UdonPred heads (`trizod`, `chezod`, `softdis`, `atlas`, `plddt`, `disprot`) against `PUNCH2-Light`, `DisoFLAG`, `DisorderUnetLM`, and `DisPredict3.0`.

Outputs:

- `spearman_heatmaps_without_se.png`: residue-level and protein-level Spearman heatmaps with values only.
- `spearman_heatmaps_with_se.png`: same heatmaps with standard error below each value.
- `top10_protein_overlap_without_se.png`: top-10% most disorder-prone protein overlap with values only.
- `top10_protein_overlap_with_se.png`: same top-10% heatmap with standard error below each value.
- `presentation_cluster_annotated_protein_spearman.png`: presentation-focused protein-level Spearman heatmap with highlighted UdonPred cluster, PUNCH2-Light bridge, and CAID-style block.
- `presentation_cluster_annotated_spearman_heatmaps.png`: presentation-focused residue-level and protein-level Spearman heatmaps with the same highlighted regions.
- `*_matrix.csv`: numeric matrices used for the plots.
- `focused_pairwise_spearman_with_se.csv` and `focused_top10_protein_overlap_with_se.csv`: pairwise table exports.

All heatmaps use the official Batlow colour map via `cmcrameri.cm.batlow`.

Standard error notes:

- Spearman SE uses a Fisher-z delta-method approximation: `(1 - rho^2) / sqrt(n - 3)`.
- Residue-level `n` is the matched residue count from `pairwise_agreement.csv`.
- Protein-level `n` is the common protein count from `pairwise_agreement.csv`.
- Top-10% overlap SE uses a binomial proportion approximation: `100 * sqrt(p * (1 - p) / n)`, where `p` is the overlap fraction and `n` is the number of proteins per top-10% set.
