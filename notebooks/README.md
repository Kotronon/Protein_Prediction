# Notebook Workflow

Run notebooks in this order:

1. `01_setup_and_exploration.ipynb` - environment checks and initial data exploration.
2. `02_shuffled_labels.ipynb` - shuffled-label null workflow.
3. `03_establish_baselines.ipynb` - simple baseline evaluation.
4. `04_annotation_ceiling.ipynb` - exact annotation ceiling analysis.
5. `05_human_proteome_annotation_ceiling.ipynb` - human-proteome annotation coverage.
6. `06_slide_enhanced_ceiling_analysis.ipynb` - slide-oriented ceiling diagnostics.
7. `07_ensembles.ipynb` - UdonPred ensemble analysis.
8. `08_seth_ensemble.ipynb` - SETH ensemble comparison.
9. `09_ensembles_ceiling_headroom.ipynb` - ceiling-aware ensemble headroom.
10. `10_ensembles_ceiling_selected_head.ipynb` - selected-head ceiling analysis.
11. `11_mmseqs_ceiling_aware_weighting.ipynb` - MMseqs-aware weighting.
12. `12_predictor_agreement.ipynb` - predictor agreement and contested-region case studies.
13. `13_contested_regions_weekly_presentation.ipynb` - weekly presentation figures.

Notebook outputs are intentionally cleared before commit. Re-run the relevant
notebook locally to regenerate inline displays from the tracked CSV/JSON result
tables.
