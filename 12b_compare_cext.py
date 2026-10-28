import glob, os, numpy as np
from sklearn.metrics import balanced_accuracy_score as bas
rows = []
for s in range(3):
    tag = f"esm2_t12_35M_random_s{s}"
    for f in sorted(glob.glob(f"results_v2_cext/probe_{tag}/*.npz")):
        name = os.path.basename(f); g = f"results_v2/probe_{tag}/{name}"
        if "_shuf" in name or not os.path.exists(g): continue
        a, b = np.load(g), np.load(f)
        acc_a = np.array([bas(a["y"], p) for p in a["preds"]]); acc_b = np.array([bas(b["y"], p) for p in b["preds"]])
        rows.append((tag, name[:-4], acc_a.max(), acc_b.max(), (acc_b - acc_a).max(), float((np.asarray(b["chosen_C"]) > 10).mean())))
print(f"{'run':30s} {'cell':14s} peak_old peak_new max_layer_gain  share_C>10")
for r in rows: print(f"{r[0]:30s} {r[1]:14s} {r[2]:.3f}    {r[3]:.3f}    {r[4]:+.3f}          {r[5]:.2f}")
d = np.array([r[3] - r[2] for r in rows])
print("\ncells:", len(rows), " mean peak change %+.4f  max %+.4f  min %+.4f" % (d.mean(), d.max(), d.min()))
