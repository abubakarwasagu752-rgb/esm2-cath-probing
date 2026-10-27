#!/bin/bash
#SBATCH --job-name=verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/verify_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
python verify_random.py
python - <<'PY'
import numpy as np
for f in ["esm2_t12_35M","esm2_t30_150M","esm2_t33_650M"]:
    E = np.load(f"emb/{f}.npy", mmap_mode="r"); print(f, E.shape, E.dtype)
PY
sha256sum emb/esm2_t12_35M_random_s*.npy emb/composition.npy emb/esm2_t33_650M.npy | tee -a emb/EMB_SHA256.txt
