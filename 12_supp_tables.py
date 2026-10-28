import glob, os, collections
import numpy as np, pandas as pd
os.makedirs("supp", exist_ok=True)
p4 = pd.read_csv("tables/phase4_selected_layer.csv")
def fmt(d):
    d = d.copy()
    d["95% interval"] = d.apply(lambda r: f"[{r.lo:.3f}, {r.hi:.3f}]", axis=1)
    return d
s1 = fmt(p4[p4.split.isin(["id80", "id50"])])[["model", "split", "task", "test_bal_acc", "95% interval"]]
s2 = fmt(p4)[["model", "split", "task", "selected_layers", "test_bal_acc", "95% interval", "peak_on_test", "optimism"]]
s3 = pd.read_csv("tables/paired_scale_selected.csv")
rows = []
for d in sorted(glob.glob("results_v2/probe_*")):
    n = fits = capped = hi = 0; mg = 0.0; cs = collections.Counter()
    for f in glob.glob(d + "/*.npz"):
        z = np.load(f); n += 1
        it = np.asarray(z["n_iter"]); g = np.asarray(z["max_grad"])
        fits += len(it); capped += int((it >= 2000).sum()); hi += int((g > 1.5e-4).sum()); mg = max(mg, float(g.max()))
        for k in ("C", "chosen_C", "best_C"):
            if k in z.files:
                cs.update(np.asarray(z[k]).ravel().tolist()); break
    rows.append(dict(run=os.path.basename(d), files=n, fits=fits, fits_at_2000_iter_cap=capped,
                     pct_max_grad_gt_1p5e4=round(100 * hi / max(fits, 1), 2), max_grad_overall=mg,
                     chosen_C_counts=str(dict(sorted(cs.items()))) if cs else "n/a"))
s4 = pd.DataFrame(rows)
parts = []
for f in sorted(set(glob.glob("**/leakage_summary*.csv", recursive=True))):
    d = pd.read_csv(f); d.insert(0, "source", f); parts.append(d)
s5 = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"note": ["no leakage_summary*.csv found"]})
out = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
for k, d in out.items():
    d.to_csv(f"supp/Table_{k}.csv", index=False); print(k, d.shape, list(d.columns)[:8])
try:
    with pd.ExcelWriter("supp/Supplementary_Tables.xlsx") as w:
        for k, d in out.items(): d.round(4).to_excel(w, sheet_name=k, index=False)
    print("wrote supp/Supplementary_Tables.xlsx")
except Exception as e:
    print("xlsx skipped:", e)
