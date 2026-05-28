# Protein Prediction

Group project for protein intrinsic disorder prediction using UdonPred and TriZOD.

This repository contains:

- `UdonPred/`: cloned UdonPred code, ONNX prediction heads, configs, and evaluation data.
- `TriZOD/`: TriZOD filtering-tier data used for G-score analysis.
- `notebooks/`: group notebooks for setup checks, TriZOD exploration, 7x7 matrix execution, and result evaluation.
- `scripts/`: reusable command-line scripts for GPU checks and matrix generation.
- `docs/`: setup notes and the project PDF.

## Quick Start

UdonPred uses `uv`, so the default group setup follows that workflow.

```bash
python3 -m pip install uv
cd UdonPred
uv sync
cd ..
```

Register the notebook kernel:

```bash
cd UdonPred
uv run python -m ipykernel install --user --name protein-prediction --display-name "Protein Prediction (UdonPred uv)"
cd ..
```

Then open:

[notebooks/01_setup_and_exploration.ipynb](notebooks/01_setup_and_exploration.ipynb)

Use the kernel named `Protein Prediction (UdonPred uv)`.

More detailed setup instructions are in [docs/setup.md](docs/setup.md).

## Smoke Test

Run a small UdonPred prediction:

```bash
cd UdonPred
uv run predict.py ../examples/smoke.fasta weights \
  --target trizod \
  --device cpu \
  --batch-size 200 \
  --smooth 0
cd ..
```

The first run may take several minutes because it downloads the ProstT5 backbone from Hugging Face.

## Data Layout

The 7x7 UdonPred evaluation expects:

```text
UdonPred/data/
  trizod/
    train.jsonl
    valid.jsonl
    test.jsonl
    test.fasta
  chezod/
  softdis/
  pdbflex/
  atlas/
  plddt/
  disprot/
```

Each of the seven datasets should contain at least `test.fasta` and `test.jsonl` for matrix evaluation.

TriZOD files are stored separately:

```text
TriZOD/
  unfiltered.json
  tolerant.json
  moderate.json
  strict.json
```

These files are newline-delimited JSON records, even though they use the `.json` extension.

## Notebook Workflow

The starter notebook supports:

- environment and dependency checks
- small UdonPred inference smoke test
- TriZOD file inspection
- TriZOD G-score distribution plots
- comparison of TriZOD filtering tiers
- UdonPred data readiness checks
- launching the 7x7 matrix run
- loading and visualizing `results/udonpred_matrix/matrix.csv`
- summarizing best training dataset per test metric

The expensive matrix run is disabled by default in the notebook:

```python
RUN_FULL_MATRIX = False
```

Set it to `True` only when you are ready to run the full evaluation.

## Reproduce The 7x7 Matrix

From the repository root:

```bash
python scripts/run_udonpred_matrix.py --device cpu
```

On a GPU machine:

```bash
python scripts/run_udonpred_matrix.py --device cuda --batch-size 100
```

The script writes:

```text
results/udonpred_matrix/matrix.csv
results/udonpred_matrix/predictions/
```

The prediction folders can be large and are ignored by Git. The CSV summaries are allowed by `.gitignore` and can be committed.

If predictions already exist and you only want to recompute metrics:

```bash
python scripts/run_udonpred_matrix.py --skip-predictions
```

To include bootstrap standard deviations:

```bash
python scripts/run_udonpred_matrix.py --skip-predictions --bootstrap-samples 100
```

## Simple Baselines

Before interpreting the UdonPred matrix, run lightweight baselines that estimate
how much signal is available from trivial sequence features:

```bash
python scripts/run_simple_baselines.py
```

This writes:

```text
results/simple_baselines/matrix.csv
```

The script evaluates:

- `aa_composition_logreg`: balanced logistic regression on global amino acid
  composition, trained separately on each UdonPred training dataset.
- `coil_propensity`: a train-independent per-residue Chou-Fasman-style coil
  propensity heuristic.

The output uses the same metric convention as the UdonPred matrix: Spearman for
continuous datasets, AP/AUROC for DisProt, and negated CheZOD/pLDDT labels so
higher scores consistently mean more disorder.

## Annotation Ceiling

Estimate agreement between real annotations for proteins that overlap across
UdonPred evaluation datasets:

```bash
UdonPred/.venv/bin/python scripts/estimate_annotation_ceiling.py --output-dir results/annotation_ceiling
```

This writes `annotation_ceiling_summary.csv`,
`annotation_ceiling_summary.json`, and `overlap_details.csv`. The output is an
inter-annotation agreement estimate, not model performance and not a
shuffled-label baseline. The script also writes PNG plots for overlap size,
continuous-pair Spearman agreement, and DisProt agreement.

To add MMseqs2 local-alignment ceilings at multiple identity thresholds:

```bash
python scripts/estimate_annotation_ceiling.py \
  --use-mmseqs \
  --output-dir results/annotation_ceiling_mmseqs
```

This keeps the exact-match ceiling and adds `mmseqs_100`, `mmseqs_98`,
`mmseqs_95`, `mmseqs_90`, `mmseqs_85`, and `mmseqs_80` comparison levels by
default. MMseqs hits must pass the identity threshold, at least 80% aligned
coverage on both proteins, and the normal minimum comparable-residue filter.
Only aligned residue pairs without gaps are compared. Use
`--mmseqs-identities` and `--mmseqs-min-coverage` to change those cutoffs. The
MMseqs run also writes threshold-comparison plots such as
`ceiling_mmseqs_overlap_by_identity.png` and
`ceiling_mmseqs_primary_agreement_by_identity.png`.

## Normalized Headroom

After the UdonPred matrix, simple baselines, and annotation ceiling are ready,
compute normalized headroom:

```bash
UdonPred/.venv/bin/python scripts/compute_normalized_headroom.py
```

This writes reusable CSVs to `results/normalized_headroom/`, including
`normalized_headroom_vs_best_simple_baseline.csv`, `cell_status.csv`, and the
seed-13 shuffled-label null summaries. The primary formula is
`(UdonPred - best_simple_baseline) / (annotation_ceiling - best_simple_baseline)`.
Missing off-diagonal annotation ceilings remain blank.

## GPU Workflow

For another computer with an NVIDIA GPU, use:

[docs/gpu_setup.md](docs/gpu_setup.md)

Short version:

```bash
cd UdonPred
uv run python ../scripts/check_gpu.py
cd ..
```

For large GPUs:

```bash
BATCH_SIZE=500 ./scripts/run_matrix_gpu.sh
```

The matrix run skips completed prediction folders by default, so it can be resumed if interrupted.

## What To Commit

Commit source, notebooks, docs, configs, and final summary CSVs:

```bash
git add README.md docs notebooks scripts requirements.txt .gitignore
git add results/udonpred_matrix/matrix.csv
git add results/udonpred_matrix/matrix_std.csv  # if created
```

Do not commit local environments, model caches, checkpoints, or generated prediction folders:

```text
.venv/
UdonPred/.venv/
.cache/
outputs/
results/udonpred_matrix/predictions/
```

These are ignored by `.gitignore`.

## Main References In This Repo

- Assignment PDF: [docs/proj4_disorder.pdf](docs/proj4_disorder.pdf)
- Setup guide: [docs/setup.md](docs/setup.md)
- GPU guide: [docs/gpu_setup.md](docs/gpu_setup.md)
- Starter notebook: [notebooks/01_setup_and_exploration.ipynb](notebooks/01_setup_and_exploration.ipynb)
- Matrix script: [scripts/run_udonpred_matrix.py](scripts/run_udonpred_matrix.py)
