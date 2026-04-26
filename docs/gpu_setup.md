# GPU Setup

Use this guide on the computer that will run the expensive UdonPred 7x7 matrix.

## 1. Install System Requirements

You need:

- NVIDIA GPU
- NVIDIA driver installed
- Python 3.13 if possible
- `uv`

Check the GPU:

```bash
nvidia-smi
```

If `nvidia-smi` is not found or shows no GPU, fix the NVIDIA driver before continuing.

## 2. Clone And Sync

```bash
git clone <your-repo-url>
cd Protein_Prediction
python3 -m pip install uv
cd UdonPred
uv sync
cd ..
```

## 3. Check CUDA From Python

```bash
cd UdonPred
uv run python ../scripts/check_gpu.py
cd ..
```

Expected result:

```text
CUDA available: True
CUDA tensor test: OK
```

If CUDA is false, install a CUDA-enabled PyTorch build in UdonPred's `uv` environment. On many Linux systems:

```bash
cd UdonPred
uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
uv run python ../scripts/check_gpu.py
cd ..
```

## 4. Run The Matrix

For a 4 GB GPU such as GTX 1050 Ti, start conservatively:

```bash
BATCH_SIZE=100 ./scripts/run_matrix_gpu.sh
```

For 8 GB+ VRAM, try:

```bash
BATCH_SIZE=500 ./scripts/run_matrix_gpu.sh
```

The output is:

```text
results/udonpred_matrix/matrix.csv
results/udonpred_matrix/predictions/
```

Only commit the CSV summary:

```bash
git add results/udonpred_matrix/matrix.csv
git commit -m "Add UdonPred 7x7 evaluation matrix"
git push
```

The generated prediction folders stay ignored by Git.

## 5. Resume Or Recompute

The matrix script skips existing prediction folders by default. If the run stops halfway, run the same command again and it will continue from missing pairs.

To force regeneration:

```bash
python scripts/run_udonpred_matrix.py --device cuda --batch-size 100 --force
```

To only recompute `matrix.csv` from existing predictions:

```bash
python scripts/run_udonpred_matrix.py --skip-predictions
```

