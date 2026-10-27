#!/bin/bash
#SBATCH --job-name=val
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/val_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
for m in esm2_t12_35M esm2_t30_150M esm2_t33_650M; do python 04h_valscores.py --emb emb/$m.npy --tag $m 2>&1 | grep -v -i warn; done
