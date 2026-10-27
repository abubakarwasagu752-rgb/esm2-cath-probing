#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/prep_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
python 01_prepare_data.py
