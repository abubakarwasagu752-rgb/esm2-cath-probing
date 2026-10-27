# Leakage-controlled probing of the CATH hierarchy in ESM-2

Code, tables and checksums for: Saadu A, Jongruja N. *Leakage-controlled probing of CATH structural hierarchy across ESM-2 layers.*

## Environment
conda env `esmprobe`: Python, PyTorch 2.6.0, transformers 5.20.2, MMseqs2 build fcf516ab (commit fcf516aba34be4569b3eb5bda2b9db9e26a8f278).
Jobs were run with Slurm (`*.slurm`).

## Pipeline (run in this order)
1. `01_prepare_data.py` - filter CATH v4.3.0, cap per superfamily
2. `02a_allvsall.py`, `02_cluster_and_split.py`, `02b_leakage_check.py` - relatedness groups, five splits, independent leakage check
3. `03_extract_embeddings.py`, `03b_composition_features.py`, `03c_check_embeddings.py` - ESM-2 35M/150M/650M embeddings and baselines
4. `04g_probe_gpu.py`, `04h_valscores.py` - linear probes and validation scores
5. `06_retrieval.py`, `06b_retrieval_metrics.py`, `05_rsa_gpu.py`, `08_cka_gpu.py` - retrieval, RSA, CKA
6. `07_analyze.py`, `10_paired.py`, `11_phase4.py` - bootstrap intervals, paired comparisons, validation-selected layers
7. `09_figures.py` - figures

## Contents
- `tables/` result tables used in the paper
- `*_SHA256.txt`, `data/CHECKSUMS.txt` checksums of data, scripts and results
- Embeddings (`emb/`) and per-run probe outputs (`results_v2/`) are not included because of size.

## Data
CATH v4.3.0 is from the CATH database. ESM-2 weights are from the ESM repository.

## Licence
See `LICENSE`.
