#!/bin/bash
#SBATCH --job-name=freeze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/freeze_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
python freeze.py
{ echo "mmseqs $(~/mmseqs/bin/mmseqs version)"; echo "torch/transformers: $(python -c 'import torch,transformers;print(torch.__version__,transformers.__version__)')"; date -u +"frozen_utc %Y-%m-%dT%H:%M:%SZ"; } > data/FREEZE_INFO.txt
find data data_cap5 data_cap10 data_cap15 -type f -not -path '*/mmseqs/*' \( -name 'domains.*' -o -name 'clusters_*.tsv' -o -name 'splits*' -o -name 'allvsall_hits.tsv' -o -name 'groups_id30.npy' -o -name 'dataset_stats.json' -o -name 'RAW_SHA256.txt' \) | sort | xargs sha256sum > data/CHECKSUMS.txt
sha256sum common.py 01_prepare_data.py 02a_allvsall.py 02_cluster_and_split.py 02b_leakage_check.py > data/SCRIPTS_SHA256.txt
wc -l data/CHECKSUMS.txt; cat data/FREEZE_INFO.txt
