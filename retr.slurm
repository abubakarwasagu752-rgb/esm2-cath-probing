#!/bin/bash
#SBATCH --job-name=retr
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/retr_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for m in esm2_t12_35M esm2_t30_150M esm2_t33_650M composition; do python 06_retrieval.py --emb emb/$m.npy --tag $m; done
for s in 0 1 2; do python 06_retrieval.py --emb emb/esm2_t12_35M_random_s$s.npy --tag esm2_t12_35M_random_s$s; done
