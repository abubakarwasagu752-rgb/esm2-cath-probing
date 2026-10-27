#!/bin/bash
#SBATCH --job-name=pv2c
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pv2b_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
G="0.0003 0.001 0.003 0.01 0.03 0.1 1 10"
python 04g_probe_gpu.py --emb emb/esm2_t33_650M.npy --tag esm2_t33_650M --C-grid $G --max-iter 2000 --out results_v2 2>&1 | grep -v -i warn
