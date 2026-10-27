#!/bin/bash
#SBATCH --job-name=rsa
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rsa_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for m in esm2_t12_35M esm2_t30_150M esm2_t33_650M composition esm2_t12_35M_random_s0 esm2_t12_35M_random_s1 esm2_t12_35M_random_s2; do
  python 05_rsa_gpu.py --emb emb/$m.npy --tag $m
done
echo ALL_DONE
