# Dispredict3.0 native MPS handoff

The Docker workers calculate the legacy flDPnn features. The native runner
uses the Apple GPU for ESM-1b and writes one resumable CAID file per protein.

## Current run

Do not rerun `parallelDispredict.sh` while the existing containers are alive.
The export command can wait until each worker has produced all flDPnn files,
copy them to the host, and stop that worker after a successful copy:

```bash
.venv-dispredict3-mps/bin/python scripts/export_dispredict3_features.py \
  --workers 3 \
  --wait \
  --stop-containers
```

Feature export is large and can take time. A container is stopped only after
its files have been copied and validated.

## Native MPS stage

Run all three accepted FASTA chunks in one process so ESM-1b is loaded once:

```bash
.venv-dispredict3-mps/bin/python scripts/run_dispredict3_mps.py \
  Dispredict3.0/ParallelDispredict3.0/temp/Parallelinputs/processedinput_1.fasta \
  Dispredict3.0/ParallelDispredict3.0/temp/Parallelinputs/processedinput_2.fasta \
  Dispredict3.0/ParallelDispredict3.0/temp/Parallelinputs/processedinput_3.fasta \
  --device mps
```

Outputs are written below:

```text
results/human_proteome/Dispredict3_native/caid/
```

Each completed protein has a `.caid` file and a sequence-hash `.json` file.
Rerunning the same command skips those proteins. Use `--overwrite` only to
recompute existing results.

If MPS runs out of memory, lower the PCA row batch size:

```bash
--row-batch-size 8
```
