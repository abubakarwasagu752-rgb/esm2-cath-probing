import numpy as np, torch
from common import load_emb
rng = np.random.default_rng(0)
tags = ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"]
arrs = {t: load_emb(f"emb/{t}.npy") for t in tags}
N = arrs[tags[0]].shape[0]; idx = np.sort(rng.choice(N, 5000, replace=False))
def feats(t):
    A = torch.from_numpy(np.asarray(arrs[t][idx], dtype=np.float32)).cuda(); return A - A.mean(0, keepdim=True)
def cka(X, Y):
    c = torch.linalg.norm(X.T @ Y) ** 2; return float(c / (torch.linalg.norm(X.T @ X) * torch.linalg.norm(Y.T @ Y)))
F = {t: feats(t) for t in tags}; out = {}
for i, a in enumerate(tags):
    for b in tags[i:]:
        M = np.zeros((F[a].shape[1], F[b].shape[1]))
        for p in range(M.shape[0]):
            for q in range(M.shape[1]):
                if a == b and q < p: M[p, q] = M[q, p]; continue
                M[p, q] = cka(F[a][:, p], F[b][:, q])
        out[(a, b)] = M
        if a == b:
            dg = np.diag(M); print(a, "diag min %.6f max %.6f" % (dg.min(), dg.max()), "| max |M-M.T| %.2e" % np.abs(M - M.T).max(), flush=True)
            assert np.abs(dg - 1).max() < 1e-3, "CKA diagonal is not 1: bug"
        np.save(f"results_v2/cka_{a}__{b}.npy", M)
Xs = F[tags[0]][:, 6]; perm = torch.as_tensor(rng.permutation(len(Xs))).cuda()
print("shuffled-rows control (should be near 0):", round(cka(Xs, Xs[perm]), 4))
for t in tags:
    M = out[(t, t)]; print(t, "CKA(layer, last):", " ".join("%.2f" % v for v in M[:, -1]))
