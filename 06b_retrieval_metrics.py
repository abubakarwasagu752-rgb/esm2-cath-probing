import glob, numpy as np, pandas as pd
from common import load_domains, cath_codes
R = 500; rng = np.random.default_rng(0)
dom = load_domains("data"); codes = cath_codes(dom)
H, T = dom["H"].values, dom["T"].values; g30 = np.load("data/groups_id30.npy")
LV = {"C": 3, "A": 2, "T": 1, "H": 0}
def grp(split, te): return {"Hhold": H, "Thold": T}.get(split, g30)[te]
def one(tag, split, seed):
    z = np.load(f"results_v2/retrieval_{tag}/{split}_s{seed}.npz"); nn, te, tr = z["nn"], z["te_idx"], z["tr_idx"]
    g = grp(split, te); gu, gi = np.unique(g, return_inverse=True); G = len(gu)
    W = rng.multinomial(G, np.full(G, 1 / G), size=R).astype(float)
    out = {}
    for lv in "CATH":
        c = codes[lv]; seen = np.isin(c[te], np.unique(c[tr]))
        if not seen.any(): out[lv] = (np.full(nn.shape[0], np.nan), np.full((R, nn.shape[0]), np.nan)); continue
        m = (c[nn] == c[te][None, :]) & seen[None, :]
        tot = np.bincount(gi, weights=seen.astype(float), minlength=G)
        hit = np.stack([np.bincount(gi, weights=m[l].astype(float), minlength=G) for l in range(nn.shape[0])], 1)
        out[lv] = (hit.sum(0) / tot.sum(), (W @ hit) / (W @ tot)[:, None])
    d = np.full(nn.shape, 4.0)
    for lv, v in LV.items(): d[codes[lv][nn] == codes[lv][te][None, :]] = v
    d = np.full(nn.shape, 4.0)
    for lv, v in (("C", 3), ("A", 2), ("T", 1), ("H", 0)): d[codes[lv][nn] == codes[lv][te][None, :]] = v
    out["dist"] = (d.mean(1), None)
    return out
rows = []
tags = ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M", "composition", "esm2_t12_35M_random_s0", "esm2_t12_35M_random_s1", "esm2_t12_35M_random_s2"]
for tag in tags:
    for split in ["id30", "Hhold", "Thold"]:
        rs = [one(tag, split, s) for s in range(3)]
        for lv in ["C", "A", "T", "H", "dist"]:
            pt = np.mean([r[lv][0] for r in rs], 0)
            if lv != "dist" and not np.isnan(pt).all():
                bt = np.mean([r[lv][1] for r in rs], 0); lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
            else: lo = hi = np.full_like(pt, np.nan)
            sd = np.std([r[lv][0] for r in rs], 0)
            for l in range(len(pt)): rows.append(dict(tag=tag, split=split, metric=lv, layer=l, value=pt[l], lo=lo[l], hi=hi[l], seed_sd=sd[l]))
df = pd.DataFrame(rows); df.to_csv("tables/retrieval_by_layer.csv", index=False)
pd.set_option("display.width", 250)
print("points outside interval:", int(((df.value < df.lo) | (df.value > df.hi)).sum()), "of", int(df.lo.notna().sum()))
b = df[df.metric.isin(["C", "A", "T"]) & df.tag.str.startswith("esm2_t") & ~df.tag.str.contains("random")]
b = b.dropna(subset=["value"])
best = b.loc[b.groupby(["tag", "split", "metric"]).value.idxmax()]
print(best[["tag", "split", "metric", "layer", "value", "lo", "hi"]].round(3).to_string(index=False))
c = df[(df.tag == "composition") & df.metric.isin(["C", "A", "T"])]; print(c[["split", "metric", "value"]].round(3).to_string(index=False))
