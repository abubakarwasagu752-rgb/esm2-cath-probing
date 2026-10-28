#!/bin/bash
#SBATCH --job-name=cext
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/cext_%A_%a.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
G="0.0003 0.001 0.003 0.01 0.03 0.1 1 10 100 1000"
S=$SLURM_ARRAY_TASK_ID
python 04g_probe_gpu.py --emb emb/esm2_t12_35M_random_s${S}.npy --tag esm2_t12_35M_random_s${S} \
  --splits id30 Hhold --C-grid $G --max-iter 2000 --out results_v2_cext 2>&1 | grep -v -i warn
