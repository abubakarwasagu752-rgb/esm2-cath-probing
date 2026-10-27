#!/bin/bash
#SBATCH --job-name=pv2a
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pv2a_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
G="0.0003 0.001 0.003 0.01 0.03 0.1 1 10"
for m in esm2_t12_35M esm2_t30_150M; do
  python 04g_probe_gpu.py --emb emb/$m.npy --tag $m --C-grid $G --max-iter 2000 --out results_v2 2>&1 | grep -v -i warn
done
for s in 0 1 2; do
  python 04g_probe_gpu.py --emb emb/esm2_t12_35M_random_s$s.npy --tag esm2_t12_35M_random_s$s --C-grid $G --max-iter 2000 --out results_v2 2>&1 | grep -v -i warn
done
