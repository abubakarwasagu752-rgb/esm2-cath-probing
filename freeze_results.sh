#!/bin/bash
cd ~/esm2_v3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
find results_v2 tables figures -type f | sort | xargs sha256sum > RESULTS_SHA256.txt
sha256sum common.py 01_prepare_data.py 02a_allvsall.py 02_cluster_and_split.py 02b_leakage_check.py 03_extract_embeddings.py 03b_composition_features.py 03c_check_embeddings.py 04g_probe_gpu.py 05_rsa_gpu.py 06_retrieval.py 06b_retrieval_metrics.py 07_analyze.py 08_cka_gpu.py 09_figures.py 10_paired.py > SCRIPTS_FINAL_SHA256.txt
{ echo "mmseqs $(~/mmseqs/bin/mmseqs version)"; python -c 'import torch,transformers,sklearn,scipy,numpy,pandas;print("torch",torch.__version__,"transformers",transformers.__version__,"sklearn",sklearn.__version__,"scipy",scipy.__version__,"numpy",numpy.__version__,"pandas",pandas.__version__)'; date -u +"results_frozen_utc %Y-%m-%dT%H:%M:%SZ"; } > RESULTS_INFO.txt
wc -l RESULTS_SHA256.txt; cat RESULTS_INFO.txt
cd ~ && tar czf esm2_v3_results_$(date +%Y%m%d).tar.gz esm2_v3/results_v2 esm2_v3/tables esm2_v3/figures esm2_v3/*.py esm2_v3/*.slurm esm2_v3/*.sh esm2_v3/RESULTS_SHA256.txt esm2_v3/SCRIPTS_FINAL_SHA256.txt esm2_v3/RESULTS_INFO.txt
ls -la ~/esm2_v3_results_*.tar.gz
