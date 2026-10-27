import numpy as np, pandas as pd
from common import load_domains
R = 1000; rng = np.random.default_rng(1)
dom = load_domains("data"); H, T = dom["H"].values, dom["T"].values; g30 = np.load("data/groups_id30.npy")
pk = pd.read_csv("tables/probe_peaks.csv")
M = ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"]
def peak(m, sp, t, s=0): return int(np.load(f"results_v2/val_{m}/{sp}_s{s}_{t}.npy").argmax())
def corr(m, sp, t, s):
    z = np.load(f"results_v2/probe_{m}/{sp}_s{s}_{t}.npz"); i = list(z["layers"]).index(peak(m, sp, t, s))
    return z["preds"][i] == z["y"], z["y"], z["te_idx"]
rows = []
for sp in ["id80", "id50", "id30", "Hhold", "Thold"]:
    for t in (["C", "A"] if sp == "Thold" else ["C", "A", "T"]):
        for a, b in [(M[0], M[1]), (M[1], M[2]), (M[0], M[2])]:
            reps, pts = [], []
            for s in range(3):
                ca, y, te = corr(a, sp, t, s); cb, y2, _ = corr(b, sp, t, s); assert (y == y2).all()
                g = {"Hhold": H, "Thold": T}.get(sp, g30)[te]
                ku, ki = np.unique(y, return_inverse=True); K = len(ku); boot = np.zeros(R); pt = 0.0
                for k in range(K):
                    idx = np.where(ki == k)[0]; gk, gi = np.unique(g[idx], return_inverse=True); G = len(gk)
                    tot = np.bincount(gi).astype(float)
                    ha = np.bincount(gi, weights=ca[idx].astype(float), minlength=G); hb = np.bincount(gi, weights=cb[idx].astype(float), minlength=G)
                    W = rng.multinomial(G, np.full(G, 1.0 / G), size=R).astype(float)
                    boot += (W @ (hb - ha)) / (W @ tot); pt += (hb - ha).sum() / tot.sum()
                reps.append(boot / K); pts.append(pt / K)
            pooled = np.array([reps[rng.integers(3)][r] for r in range(R)])
            lo, hi = np.percentile(pooled, [2.5, 97.5])
            rows.append(dict(split=sp, task=t, larger=b, smaller=a, diff=np.mean(pts), lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
d = pd.DataFrame(rows); d.to_csv("tables/paired_scale_selected.csv", index=False)
pd.set_option("display.width", 250); print(d.round(3).to_string(index=False))
