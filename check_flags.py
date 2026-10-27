import json, glob, os, pandas as pd
cols = ["query","target","fident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
THR = {"id80": .8, "id50": .5, "id30": .3, "Hhold": .3, "Thold": .3}
for d in ["data","data_cap5","data_cap10","data_cap15"]:
    r = json.load(open(f"{d}/splits_report.json"))
    for k in r:
        if k.startswith(("Thold","Hhold")):
            print(d, k, "n_test", r[k]["n_test"], "embargoed", r[k]["n_embargoed_test"], "C4_test", r[k]["test_C4"])
    a = pd.read_csv(f"{d}/allvsall_hits.tsv", sep="\t", header=None, names=cols)
    key = set(zip(a["query"], a.target)) | set(zip(a.target, a["query"]))
    for f in sorted(glob.glob(f"phase1_{d}/hits_*_s*.tsv")):
        sp = os.path.basename(f).split("_")[1]
        h = pd.read_csv(f, sep="\t", header=None, names=cols); s = h[h.evalue <= 1e-3]
        c = s[(s.fident >= THR[sp]) & (s[["qcov","tcov"]].max(axis=1) >= .8)]
        for x in c.itertuples():
            print("FLAG", d, os.path.basename(f), x.query, x.target, round(x.fident,3), "e", f"{x.evalue:.2e}", "bits", x.bits, "in_allvsall", (x.query, x.target) in key)
