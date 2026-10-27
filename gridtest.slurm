#!/bin/bash
#SBATCH --job-name=gridtest
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gridtest_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
python 04g_probe_gpu.py --emb emb/esm2_t33_650M.npy --tag gt --splits Hhold --seeds 0 --tasks C A T --layers 10 20 33 \
  --C-grid 0.0003 0.001 0.003 0.01 0.03 0.1 --max-iter 2000 --out results_gridtest 2>&1 | grep -v -i warn
python - <<'PY'
import numpy as np
for t in "CAT":
    z = np.load(f"results_gridtest/probe_gt/Hhold_s0_{t}.npz")
    print(t, "layers", z["layers"], "chosen_C", z["chosen_C"], "n_iter", z["n_iter"], "max_grad", z["max_grad"])
PY
