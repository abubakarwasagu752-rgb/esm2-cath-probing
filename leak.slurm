#!/bin/bash
#SBATCH --partition=gpu3060,gpu4090,gpua4000,gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --job-name=leak
#SBATCH --output=leak_%j.log

source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for s in 0 1 2; do python 13_leak_examples.py --out leak_ex_s$s --layer 28 --seed $s; done
