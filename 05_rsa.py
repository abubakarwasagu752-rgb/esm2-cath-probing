import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kendalltau, rankdata
from common import load_domains, load_emb, load_split_codes, cath_codes, cath_distance_matrix
ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="Hhold"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--perms", type=int, default=500); ap.add_argument("--out", default="results_v2")
a = ap.parse_args()
dom = load_domains("data"); codes = cath_codes(dom); arr = load_emb(a.emb); L = arr.shape[1]
code = load_split_codes("data", a.split, a.seed); tr = np.where(code == 0)[0]; te = np.where(code == 1)[0]
rng = np.random.default_rng(0); n = min(a.n, len(te)); iu = np.triu_indices(n, 1)
mu = [None] * L; sd = [None] * L
for l in range(L):
    Xt = np.asarray(arr[tr, l, :], dtype=np.float32); mu[l] = Xt.mean(0); sd[l] = np.maximum(Xt.std(0), 1e-8)
rows = []; perm_rows = None
for draw in range(a.draws):
    idx = np.sort(rng.choice(te, n, replace=False))
    Dc = cath_distance_matrix(codes, idx); dc = Dc[iu].astype(np.int8)
    vals, counts = np.unique(dc, return_counts=True); cum = np.cumsum(counts)
    lut = np.zeros(5); lut[vals] = cum - (counts - 1) / 2.0
    rc = lut[dc]; rcc = rc - rc.mean(); nc = np.linalg.norm(rcc)
    Rl = np.empty((L, len(dc)), dtype=np.float32)
    for l in range(L):
        X = (np.asarray(arr[idx, l, :], dtype=np.float32) - mu[l]) / sd[l]
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        d = (1.0 - X @ X.T)[iu]
        r = rankdata(d); r = r - r.mean(); r /= np.linalg.norm(r); Rl[l] = r
        rows.append(dict(draw=draw, layer=l, spearman=float(r @ rcc / nc), kendall_b=float(kendalltau(d, dc, variant="b").statistic)))
    if draw == 0:
        obs = np.array([x["spearman"] for x in rows[-L:]]); ge = np.zeros(L); gm = np.zeros(L)
        for _ in range(a.perms):
            p = rng.permutation(n); rp = lut[Dc[np.ix_(p, p)][iu]] - rc.mean()
            st = Rl @ rp.astype(np.float32) / nc; ge += st >= obs; gm += st.max() >= obs
        perm_rows = pd.DataFrame(dict(layer=np.arange(L), spearman_draw0=obs, p=(1 + ge) / (1 + a.perms), p_fwe=(1 + gm) / (1 + a.perms)))
    print(a.tag, "draw", draw, "done", flush=True)
Path(a.out).mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f"{a.out}/rsa_{a.tag}_{a.split}.csv", index=False)
perm_rows.to_csv(f"{a.out}/rsa_{a.tag}_{a.split}_perm.csv", index=False)
