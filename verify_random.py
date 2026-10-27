import numpy as np
from common import load_domains
n = len(load_domains("data"))
P = np.load("emb/esm2_t12_35M.npy", mmap_mode="r")
idx = np.random.default_rng(0).choice(n, 500, replace=False); idx.sort()
p = np.asarray(P[idx]).astype(np.float32)
R = {}
for s in (0, 1, 2):
    E = np.load(f"emb/esm2_t12_35M_random_s{s}.npy", mmap_mode="r")
    fin = all(np.isfinite(np.asarray(E[a:a+2000]).astype(np.float32)).all() for a in range(0, n, 2000))
    r = np.asarray(E[idx]).astype(np.float32); R[s] = r
    cos = {l: float(((r[:, l] * p[:, l]).sum(1) / (np.linalg.norm(r[:, l], axis=1) * np.linalg.norm(p[:, l], axis=1) + 1e-9)).mean()) for l in (1, 6, 12)}
    print(f"seed {s}: shape {E.shape} finite {fin} cos-to-pretrained {cos}")
for a, b in ((0, 1), (0, 2)):
    l = 12
    c = ((R[a][:, l] * R[b][:, l]).sum(1) / (np.linalg.norm(R[a][:, l], axis=1) * np.linalg.norm(R[b][:, l], axis=1) + 1e-9)).mean()
    print(f"seeds {a} vs {b}, layer 12 mean cosine: {c:.3f}")
