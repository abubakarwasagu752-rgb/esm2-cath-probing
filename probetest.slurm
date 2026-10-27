#!/bin/bash
#SBATCH --job-name=ptest
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ptest_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for T in C A T; do
  echo "== task $T"; ( time python 04g_probe_gpu.py --emb emb/esm2_t12_35M.npy --tag ptest --splits Hhold --seeds 0 --tasks $T --layers 6 --out results_test ) 2>&1 | grep -v Warn
done
