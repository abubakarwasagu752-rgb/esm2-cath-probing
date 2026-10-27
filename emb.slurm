#!/bin/bash
#SBATCH --job-name=emb
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/emb_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; mkdir -p emb
python 03b_composition_features.py
for spec in "esm2_t12_35M_UR50D esm2_t12_35M" "esm2_t30_150M_UR50D esm2_t30_150M" "esm2_t33_650M_UR50D esm2_t33_650M"; do
  set -- $spec
  python 03_extract_embeddings.py --model facebook/$1 --out emb/$2.npy && \
  python 03c_check_embeddings.py emb/$2.npy facebook/$1 && sha256sum emb/$2.npy >> emb/EMB_SHA256.txt
done
for s in 0 1 2; do
  python 03_extract_embeddings.py --model facebook/esm2_t12_35M_UR50D --random-init --seed $s --out emb/esm2_t12_35M_random_s$s.npy && \
  sha256sum emb/esm2_t12_35M_random_s$s.npy >> emb/EMB_SHA256.txt
done
sha256sum emb/composition.npy >> emb/EMB_SHA256.txt
