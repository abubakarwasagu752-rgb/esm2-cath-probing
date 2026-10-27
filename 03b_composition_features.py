import argparse, numpy as np
from pathlib import Path
from common import load_domains
AA = "ACDEFGHIKLMNPQRSTVWY"
ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="emb/composition.npy")
a = ap.parse_args(); dom = load_domains(a.data)
X = np.zeros((len(dom), 1, 21), dtype=np.float32)
for i, s in enumerate(dom["seq"]):
    X[i, 0, :20] = np.array([s.count(c) for c in AA], dtype=np.float32) / len(s)
    X[i, 0, 20] = np.log(len(s))
Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.save(a.out, X); print("saved", a.out, X.shape)
