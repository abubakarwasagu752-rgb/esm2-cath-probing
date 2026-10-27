import glob, sys, collections, numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
cap_iter = 2000
for d in sorted(glob.glob(f"{root}/probe_*")):
    cap, mg, cs, n, bad = [], 0.0, collections.Counter(), 0, 0
    for f in glob.glob(d + "/*.npz"):
        try: z = np.load(f)
        except Exception: bad += 1; continue
        n += 1; mg = max(mg, float(z["max_grad"].max()))
        cap += [(f.split("/")[-1], int(l)) for l, i in zip(z["layers"], z["n_iter"]) if i >= cap_iter]
        if "_shuf" not in f: cs.update(z["chosen_C"].tolist())
    tot = sum(cs.values()) or 1
    print(f"{d.split('/')[-1]}: {n} files, unreadable {bad} | max|grad| {mg:.1e} | capped {len(cap)} | "
          f"C=0.0003: {100*cs.get(0.0003,0)/tot:.0f}% | chosen {dict(sorted(cs.items()))}")
