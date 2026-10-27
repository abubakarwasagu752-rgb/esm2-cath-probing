import numpy as np, pandas as pd
from common import load_domains
rng = np.random.default_rng(2); R = 1000
dom = load_domains("data"); g30 = np.load("data/groups_id30.npy")
def boot(corr, y, g):
    ku, ki = np.unique(y, return_inverse=True); b = np.zeros(R); pt = 0.0
    for k in range(len(ku)):
        idx = np.where(ki == k)[0]; gk, gi = np.unique(g[idx], return_inverse=True); n = len(gk)
        tot = np.bincount(gi).astype(float); h = np.bincount(gi, weights=corr[idx].astype(float), minlength=n)
        W = rng.multinomial(n, np.full(n, 1 / n), size=R).astype(float); b += (W @ h) / (W @ tot); pt += h.sum() / tot.sum()
    return pt / len(ku), b / len(ku)
pk = pd.read_csv("tables/probe_peaks.csv"); rows = []
for m in ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"]:
    for sp in ["id80", "id50", "id30", "Hhold", "Thold"]:
        grp = {"Hhold": dom["H"].values, "Thold": dom["T"].values}.get(sp, g30)
        for t in (["C", "A"] if sp == "Thold" else ["C", "A", "T"]):
            pts, reps, ls = [], [], []
            for s in range(3):
                z = np.load(f"results_v2/probe_{m}/{sp}_s{s}_{t}.npz"); v = np.load(f"results_v2/val_{m}/{sp}_s{s}_{t}.npy")
                l = int(v.argmax()); ls.append(l); p, b = boot(z["preds"][l] == z["y"], z["y"], grp[z["te_idx"]]); pts.append(p); reps.append(b)
            pooled = np.array([reps[rng.integers(3)][r] for r in range(R)]); lo, hi = np.percentile(pooled, [2.5, 97.5])
            rows.append(dict(model=m, split=sp, task=t, selected_layers=str(ls), test_bal_acc=np.mean(pts), lo=lo, hi=hi,
                             peak_on_test=pk[(pk.model == m) & (pk.split == sp) & (pk.task == t)].peak_bal_acc.iloc[0]))
d = pd.DataFrame(rows); d["optimism"] = d.peak_on_test - d.test_bal_acc; d.to_csv("tables/phase4_selected_layer.csv", index=False)
pd.set_option("display.width", 250); print(d.round(3).to_string(index=False))
