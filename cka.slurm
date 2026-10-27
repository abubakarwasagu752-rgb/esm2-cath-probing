#!/bin/bash
#SBATCH --job-name=cka
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cka_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
python 08_cka_gpu.py
