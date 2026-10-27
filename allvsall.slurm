#!/bin/bash
#SBATCH --job-name=avsa
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/avsa_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for d in data data_cap5 data_cap10 data_cap15; do
  echo "=== $d ==="; python 02a_allvsall.py --data $d --threads 16; wc -l $d/allvsall_hits.tsv
done
