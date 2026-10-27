#!/bin/bash
#SBATCH --job-name=split
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/split_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
for d in data data_cap5 data_cap10 data_cap15; do
  echo "######## $d"
  python 02_cluster_and_split.py --data $d --threads 16 || echo "SPLIT FAILED for $d"
  mkdir -p phase1_$d
  python 02b_leakage_check.py --data $d --out phase1_$d --threads 16
done
