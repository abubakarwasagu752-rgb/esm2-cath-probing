import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.stats import kendalltau
from common import load_domains, load_emb, load_split_codes, cath_codes, cath_distance_matrix
ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--split", default="Hhold"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--perms", type=int, default=500); ap.add_argument("--out", default="results_v2")
a = ap.parse_args(); dev = "cuda"
dom = load_domains("data"); codes = cath_codes(dom); arr = load_emb(a.emb); L = arr.shape[1]
code = load_split_codes("data", a.split, a.seed); tr = np.where(code == 0)[0]; te = np.where(code == 1)[0]
rng = np.random.default_rng(0); n = min(a.n, len(te)); iu = np.triu_indices(n, 1)
iu_t = (torch.as_tensor(iu[0], device=dev), torch.as_tensor(iu[1], device=dev))
mu, sd = [], []
for l in range(L):
    Xt = np.asarray(arr[tr, l, :], dtype=np.float32); mu.append(torch.tensor(Xt.mean(0), device=dev)); sd.append(torch.tensor(np.maximum(Xt.std(0), 1e-8), device=dev))
def rank_t(x):  # average ranks (ties averaged) on GPU
    u, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    cum = torch.cumsum(cnt, 0).double(); r = cum - (cnt.double() - 1) / 2
    return r[inv]
rows = []; perm_rows = None
for draw in range(a.draws):
    idx = np.sort(rng.choice(te, n, replace=False))
    Dc = cath_distance_matrix(codes, idx); dc = Dc[iu].astype(np.int8)
    vals, counts = np.unique(dc, return_counts=True); cum = np.cumsum(counts)
    lut = np.zeros(5); lut[vals] = cum - (counts - 1) / 2.0
    rc = torch.tensor(lut[dc], device=dev, dtype=torch.float64); rcc = rc - rc.mean(); nc = rcc.norm()
    sub = rng.choice(len(dc), 300000, replace=False)
    Rl = torch.empty((L, len(dc)), device=dev, dtype=torch.float32)
    for l in range(L):
        X = (torch.tensor(np.asarray(arr[idx, l, :], dtype=np.float32), device=dev) - mu[l]) / sd[l]
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
        d = (1.0 - X @ X.T)[iu_t]
        r = rank_t(d); r = r - r.mean(); r = r / r.norm(); Rl[l] = r.float()
        tau = kendalltau(d[sub].cpu().numpy(), dc[sub], variant="b").statistic
        rows.append(dict(draw=draw, layer=l, spearman=float((r @ rcc / nc)), kendall_b_subsample=float(tau)))
    if draw == 0:
        obs = torch.tensor([x["spearman"] for x in rows[-L:]], device=dev); ge = torch.zeros(L, device=dev); gm = torch.zeros(L, device=dev)
        for _ in range(a.perms):
            p = rng.permutation(n); rp = torch.tensor(lut[Dc[np.ix_(p, p)][iu]], device=dev, dtype=torch.float64); rp = rp - rc.mean()
            st = (Rl.double() @ rp) / nc; ge += (st >= obs).float(); gm += (st.max() >= obs).float()
        perm_rows = pd.DataFrame(dict(layer=np.arange(L), spearman_draw0=obs.cpu().numpy(), p=((1 + ge) / (1 + a.perms)).cpu().numpy(), p_fwe=((1 + gm) / (1 + a.perms)).cpu().numpy()))
    print(a.tag, "draw", draw, "done", flush=True)
Path(a.out).mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f"{a.out}/rsa_{a.tag}_{a.split}.csv", index=False)
perm_rows.to_csv(f"{a.out}/rsa_{a.tag}_{a.split}_perm.csv", index=False)
