import glob, sys, os, numpy as np, pandas as pd, warnings
from sklearn.metrics import f1_score
from common import load_domains
warnings.filterwarnings("ignore")
R = 1000; rng = np.random.default_rng(0)
dom = load_domains("data"); H = dom["H"].values; T = dom["T"].values
g30 = np.load("data/groups_id30.npy")
def groups(split, te_idx):
    return {"Hhold": H, "Thold": T}.get(split, g30)[te_idx]
def bal_acc_boot(preds, y, g, R):
    # preds (L,n); returns point (L,), boot (R,L)
    gu, gi = np.unique(g, return_inverse=True); ku, ki = np.unique(y, return_inverse=True)
    G, K, L = len(gu), len(ku), preds.shape[0]
    tot = np.zeros((G, K)); np.add.at(tot, (gi, ki), 1)
    hit = np.zeros((L, G, K))
    for l in range(L):
        c = preds[l] == y; np.add.at(hit[l], (gi[c], ki[c]), 1)
    W = rng.multinomial(G, np.full(G, 1 / G), size=R).astype(np.float64)
    tR = W @ tot; present = tR > 0
    bootv = np.empty((R, L))
    for l in range(L):
        hR = W @ hit[l]; rec = np.where(present, hR / np.maximum(tR, 1), 0)
        bootv[:, l] = rec.sum(1) / present.sum(1)
    t0 = tot.sum(0); pt = np.array([(hit[l].sum(0)[t0 > 0] / t0[t0 > 0]).mean() for l in range(L)])
    return pt, bootv
def bal_acc_boot(preds, y, g, R):
    """Class-stratified cluster bootstrap: each class resamples only the groups holding its own test domains,
    so every class is present in every replicate (fixed class set, no dropout bias)."""
    ku, ki = np.unique(y, return_inverse=True); L = preds.shape[0]; K = len(ku)
    correct = (preds == y[None, :]); bootv = np.zeros((R, L)); pt = np.zeros(L)
    for k in range(K):
        idx = np.where(ki == k)[0]
        gk, gi = np.unique(g[idx], return_inverse=True); G = len(gk)
        tot = np.bincount(gi, minlength=G).astype(float)
        hit = np.stack([np.bincount(gi, weights=correct[l, idx], minlength=G) for l in range(L)], 1)
        pt += hit.sum(0) / tot.sum()
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=R).astype(float)
        bootv += (W @ hit) / (W @ tot)[:, None]
    return pt / K, bootv / K

def summarize(tag, split, task, suffix=""):
    fs = [f for f in sorted(glob.glob(f"results_v2/probe_{tag}/{split}_s*_{task}{suffix}.npz")) if bool("_shuf" in f) == bool(suffix)]
    if not fs: return None
    pts, boots, f1s, nd, nt = [], [], [], [], []
    for f in fs:
        z = np.load(f); p, b = bal_acc_boot(z["preds"], z["y"], groups(split, z["te_idx"]), R)
        pts.append(p); boots.append(b); nd.append(float(z["frac_dropped"])); nt.append(len(z["y"]))
        lab = np.unique(z["y"]); f1s.append([f1_score(z["y"], q, labels=lab, average="macro", zero_division=0) for q in z["preds"]])
    return dict(layers=z["layers"], bal=np.mean(pts, 0), seed_sd=np.std(pts, 0), boot=np.mean(boots, 0),
                f1=np.mean(f1s, 0), frac_dropped=np.mean(nd), n_test=np.mean(nt), pts=np.array(pts))
os.makedirs("tables", exist_ok=True)
models = ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"]
rows, peaks = [], []
for split in ["id80", "id50", "id30", "Hhold", "Thold"]:
    for task in (["C", "A"] if split == "Thold" else ["C", "A", "T"]):
        comp = summarize("composition", split, task)
        shuf = summarize("esm2_t12_35M", split, task, "_shuf") if split in ("Hhold", "id30") else None
        rnd = [summarize(f"esm2_t12_35M_random_s{s}", split, task) for s in range(3)]
        rb = np.mean([r["bal"] for r in rnd], 0)
        for m in models:
            s = summarize(m, split, task)
            if s is None: continue
            q_lo, q_hi = np.percentile(s["boot"], [2.5, 97.5], axis=0)
            lo, hi = q_lo, q_hi   # percentile interval (stratified bootstrap is unbiased)
            bias = s["boot"].mean(0) - s["bal"]
            print(f"{m} {split} {task}: max |bootstrap bias| {np.abs(bias).max():.4f}")
            for i, l in enumerate(s["layers"]):
                rows.append(dict(model=m, split=split, task=task, layer=int(l), bal_acc=s["bal"][i], lo=lo[i], hi=hi[i],
                                 macro_f1=s["f1"][i], seed_sd=s["seed_sd"][i], n_test=s["n_test"], frac_dropped=s["frac_dropped"],
                                 composition=comp["bal"][0] if comp else np.nan,
                                 random_mean=rb[i] if len(rb) == len(s["layers"]) else np.nan,
                                 random_min=(min(r["bal"][i] for r in rnd) if len(rb) == len(s["layers"]) else np.nan), random_max=(max(r["bal"][i] for r in rnd) if len(rb) == len(s["layers"]) else np.nan),
                                 shuffled=shuf["bal"].mean() if shuf else np.nan))
            b = s["boot"]; pk = int(s["bal"].argmax()); d = b[:, [pk]] - b
            onset = int(np.argmax(np.percentile(d, 2.5, axis=0) <= 0))
            peaks.append(dict(model=m, split=split, task=task, peak_layer=int(s["layers"][pk]), peak_bal_acc=s["bal"][pk],
                              earliest_layer_within_CI_of_best=int(s["layers"][onset]), n_layers=len(s["layers"]),
                              rel_depth_peak=pk / (len(s["layers"]) - 1), composition=comp["bal"][0] if comp else np.nan,
                              random_at_peak=rb[pk] if len(rb) == len(s["layers"]) else np.nan))
pd.DataFrame(rows).to_csv("tables/probe_by_layer.csv", index=False)
pk = pd.DataFrame(peaks); pk.to_csv("tables/probe_peaks.csv", index=False)
pd.set_option("display.width", 250); print(pk.round(3).to_string(index=False))

d = pd.read_csv("tables/probe_by_layer.csv")
print("points outside their interval:", int(((d.bal_acc < d.lo) | (d.bal_acc > d.hi)).sum()), "of", len(d))
