#!/bin/bash
# Final freeze: checksums + tarball of code, tables, figures, supplement and probe results (no embeddings)
set -e
cd ~/esm2_v3
D=$(date +%Y%m%d)
# 1. figures package
tar czf figures_final.tar.gz figures
# 2. checksums
ls *.py *.slurm *.sh LICENSE README.md .zenodo.json 2>/dev/null | sort | xargs sha256sum > SCRIPTS_FINAL_SHA256.txt
find tables figures supp results_v2 results_v2_cext -type f 2>/dev/null | sort | xargs sha256sum > RESULTS_SHA256.txt
{ echo "Frozen: $(date -Is)"; echo "Host: $(hostname)"; python -c "import torch,transformers,sklearn;print('torch',torch.__version__,'transformers',transformers.__version__,'sklearn',sklearn.__version__)"; echo "Files checksummed: $(wc -l < RESULTS_SHA256.txt)"; } > RESULTS_INFO.txt
# 3. archive
tar czf esm2_v3_results_${D}.tar.gz tables figures supp results_v2 results_v2_cext *.py *.slurm *.sh LICENSE README.md .zenodo.json SCRIPTS_FINAL_SHA256.txt RESULTS_SHA256.txt RESULTS_INFO.txt data/CHECKSUMS.txt emb/EMB_SHA256.txt 2>/dev/null
ls -la esm2_v3_results_${D}.tar.gz figures_final.tar.gz
cat RESULTS_INFO.txt
