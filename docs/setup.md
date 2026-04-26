# Group Setup

This project uses UdonPred for protein intrinsic disorder prediction and TriZOD data for analysis.

## 1. Clone The Repository

```bash
git clone <your-repo-url>
cd Protein_Prediction
```

## 2. Install `uv` And Sync UdonPred

Use Python 3.13 if available. The current UdonPred `pyproject.toml` declares `requires-python >=3.13`.

UdonPred's own README uses `uv`, so this is the default setup for the group project.

```bash
python3 -m pip install uv
cd UdonPred
uv sync
```

`uv sync` creates and manages the environment for UdonPred. The first run may take a while because it installs PyTorch, Transformers, ONNX Runtime, and the rest of the dependencies.

## Optional: Fallback Without `uv`

Use this only if `uv` causes problems on a teammate's machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

With the fallback `.venv` setup, replace `uv run predict.py ...` with `../.venv/bin/python predict.py ...` when running commands from inside `UdonPred/`.

## 3. Run UdonPred Commands

```bash
cd UdonPred
uv run predict.py ../examples/smoke.fasta weights \
  --target trizod \
  --device cpu \
  --batch-size 200 \
  --smooth 0
```

## 4. Register The Jupyter Kernel

```bash
cd UdonPred
uv run python -m ipykernel install --user --name protein-prediction --display-name "Protein Prediction (UdonPred uv)"
```

In Jupyter, select the kernel named `Protein Prediction (UdonPred uv)`.

## 5. Smoke Test UdonPred Inference

The first run downloads the ProstT5 backbone from Hugging Face and can take several minutes.

Use the `uv run predict.py ...` command from the section above.

You should see one disorder score per residue.

To write CAID-style output files instead of printing to the terminal:

```bash
cd UdonPred
uv run predict.py ../examples/smoke.fasta weights \
  --target trizod \
  --device cpu \
  --output ../outputs/smoke_trizod
```

## 6. Reproduce The 7x7 Evaluation Matrix

After downloading the UdonPred data, the required test files should exist under `UdonPred/data/<dataset>/test.fasta` and `UdonPred/data/<dataset>/test.jsonl`.

For a GPU machine, use the dedicated guide:

[docs/gpu_setup.md](gpu_setup.md)

Run the full matrix from the repository root:

```bash
python scripts/run_udonpred_matrix.py --device cpu
```

This runs 49 prediction jobs: seven trained heads against seven test sets. On CPU this can take many hours because every test sequence is embedded with ProstT5. Use a GPU machine if available:

```bash
python scripts/run_udonpred_matrix.py --device cuda
```

The script writes:

```text
results/udonpred_matrix/matrix.csv
results/udonpred_matrix/predictions/
```

If predictions already exist and you only want to recompute metrics:

```bash
python scripts/run_udonpred_matrix.py --skip-predictions
```

To include bootstrap standard deviations like the UdonPred notebook:

```bash
python scripts/run_udonpred_matrix.py --skip-predictions --bootstrap-samples 100
```

## 7. Data Layout Expected By UdonPred Training

UdonPred training expects JSONL datasets here:

```text
UdonPred/data/trizod/train.jsonl
UdonPred/data/trizod/valid.jsonl
UdonPred/data/trizod/test.jsonl
UdonPred/data/chezod/train.jsonl
UdonPred/data/chezod/valid.jsonl
UdonPred/data/chezod/test.jsonl
...
```

The seven dataset names in `UdonPred/config/data.yaml` are:

```text
trizod
chezod
disprot
pdbflex
plddt
softdis
atlas
```

The current repository includes UdonPred code, ONNX prediction heads, and TriZOD files. The `TriZOD/*.json` files are newline-delimited JSON records, despite the `.json` extension. Reproducing the full 7x7 cross-dataset matrix also requires the UdonPred training/evaluation JSONL data from the linked Figshare dataset.

## 8. When To Use Notebooks

Use notebooks for exploratory analysis:

- checking setup
- running small inference examples
- plotting score distributions
- comparing predictor outputs
- preparing figures for discussion

Use scripts for long-running training, repeated evaluation, and final reproducible results.
