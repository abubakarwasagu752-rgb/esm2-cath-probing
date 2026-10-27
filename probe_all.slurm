#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probe_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
cd ~/esm2_v3
export PYTHONWARNINGS=ignore
for m in esm2_t12_35M esm2_t30_150M esm2_t33_650M; do
  python 04g_probe_gpu.py --emb emb/$m.npy --tag $m 2>&1 | grep -v -i warn
done
for s in 0 1 2; do
  python 04g_probe_gpu.py --emb emb/esm2_t12_35M_random_s$s.npy --tag esm2_t12_35M_random_s$s 2>&1 | grep -v -i warn
done
python 04g_probe_gpu.py --emb emb/composition.npy --tag composition 2>&1 | grep -v -i warn
for m in esm2_t12_35M esm2_t30_150M esm2_t33_650M; do
  python 04g_probe_gpu.py --emb emb/$m.npy --tag $m --shuffle-control --splits Hhold id30 --tune none --C 1.0 2>&1 | grep -v -i warn
done
python - <<'PY'
import glob, numpy as np, collections
for d in sorted(glob.glob("results/probe_*")):
    cap, mg, cs, n = [], 0.0, collections.Counter(), 0
    for f in glob.glob(d + "/*.npz"):
        z = np.load(f); n += 1; mg = max(mg, float(z["max_grad"].max()))
        cap += [(f.split("/")[-1], int(l)) for l, i in zip(z["layers"], z["n_iter"]) if i >= 500]
        if "_shuf" not in f: cs.update(z["chosen_C"].tolist())
    print(d, n, "files | max|grad| %.1e" % mg, "| capped fits", len(cap), "| chosen C", dict(sorted(cs.items())))
PY
