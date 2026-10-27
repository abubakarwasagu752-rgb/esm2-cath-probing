#!/bin/bash
#SBATCH --job-name=fig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fig_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
python 09_figures.py
