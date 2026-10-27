#!/bin/bash
#SBATCH --job-name=floor
#SBATCH --partition=gpu3060
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/floor_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3; export PYTHONWARNINGS=ignore
python 04g_probe_gpu.py --emb emb/esm2_t33_650M.npy --tag ft --splits Hhold id30 --seeds 0 --tasks C A T --layers 10 20 30 33 \
  --C-grid 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 --max-iter 2000 --out results_floor 2>&1 | grep -v -i warn
python - <<'PY'
import numpy as np, warnings
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
for sp in ["Hhold","id30"]:
    for t in "CAT":
        a = np.load(f"results_v2/probe_esm2_t33_650M/{sp}_s0_{t}.npz"); b = np.load(f"results_floor/probe_ft/{sp}_s0_{t}.npz")
        for j, l in enumerate(b["layers"]):
            i = list(a["layers"]).index(l)
            print(sp, t, "L%d" % l, "C old %g new %g" % (a["chosen_C"][i], b["chosen_C"][j]),
                  "bal.acc old %.3f new %.3f" % (balanced_accuracy_score(a["y"], a["preds"][i]), balanced_accuracy_score(b["y"], b["preds"][j])))
PY
