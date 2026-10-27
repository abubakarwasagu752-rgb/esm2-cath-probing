#!/usr/bin/env python3
"""Step 2 (v3) - relatedness groups and leakage-free splits.
Rule A(thr): evalue<=1e-3, identity>=thr, coverage>=0.8 of EITHER sequence.
Rule B     : evalue<=1e-3, identity>=0.3, coverage>=0.8 of BOTH sequences.
Groups: id80/id50/id30 = components of rule A (+MMseqs2 clusters); Hhold = rule A@30 + H + clusters;
Thold = rule B@30 + T + clusters. Test-side embargo: test domains with a rule-A hit to train -> -1.
Codes: 0 train, 1 test, -1 embargoed."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import load_cluster_ids, load_domains, ALL_SPLITS

MMSEQS = os.environ.get("MMSEQS", str(Path.home() / "mmseqs/bin/mmseqs"))
EDGE_COLS = ["query","target","fident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
PCT = {"id80": 80, "id50": 50, "id30": 30, "Hhold": 30, "Thold": 30}
COV, EVALUE = 0.8, 1e-3

def run_mmseqs(fasta, workdir, pct, threads):
    prefix, tmp = workdir / f"clu{pct}", workdir / f"tmp{pct}"
    cmd = [MMSEQS, "easy-cluster", str(fasta), str(prefix), str(tmp), "--min-seq-id", str(pct/100),
           "-c", "0.8", "--cov-mode", "0", "-s", "7.5", "--threads", str(threads)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return Path(str(prefix) + "_cluster.tsv")

def to_clusters_tsv(raw, dest):
    cl = pd.read_csv(raw, sep="\t", header=None, names=["cluster_id","domain_id"], dtype=str)
    cl[["domain_id","cluster_id"]].to_csv(dest, sep="\t", index=False)

def load_edges(path, dom):
    h = pd.read_csv(path, sep="\t", header=None, names=EDGE_COLS)
    idx = pd.Series(np.arange(len(dom)), index=dom["domain_id"].values)
    h["qi"], h["ti"] = h["query"].map(idx), h["target"].map(idx)
    if h[["qi","ti"]].isna().any().any(): raise ValueError("hit ids not in domains.tsv")
    if h["fident"].max() > 1.0: raise ValueError("fident must be a fraction")
    h = h[(h["qi"] != h["ti"]) & (h["evalue"] <= EVALUE)].copy()
    h["qi"], h["ti"] = h["qi"].astype(int), h["ti"].astype(int)
    h["cov_max"] = h[["qcov","tcov"]].max(axis=1); h["cov_min"] = h[["qcov","tcov"]].min(axis=1)
    return h

def merge_groups(n, ei, ej, label_arrays):
    rows, cols, off = [np.asarray(ei)], [np.asarray(ej)], n
    for lab in label_arrays:
        f = pd.factorize(lab)[0]
        rows.append(np.arange(n)); cols.append(f + off); off += int(f.max()) + 1
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(off, off))
    _, lab = connected_components(g, directed=False)
    return pd.factorize(lab[:n])[0]

def group_split(groups, test_frac, seed):
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(groups, return_counts=True)
    target, chosen, cur = test_frac * len(groups), [], 0
    for i in rng.permutation(len(labels)):
        if cur >= target: break
        if abs(cur + counts[i] - target) <= abs(cur - target):
            chosen.append(labels[i]); cur += counts[i]
    code = np.zeros(len(groups), dtype=np.int8); code[np.isin(groups, chosen)] = 1
    return code

def crossing(code, ei, ej):
    a, b = code[ei], code[ej]
    return ((a == 1) & (b == 0)) | ((a == 0) & (b == 1))

def embargo_test(code, ei, ej):
    code = code.copy(); cr = crossing(code, ei, ej)
    code[np.unique(np.where(code[ei] == 1, ei, ej)[cr])] = -1
    return code

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-group-frac", type=float, default=0.07)
    ap.add_argument("--skip-mmseqs", action="store_true")
    a = ap.parse_args(); data = Path(a.data); hits_path = data / "allvsall_hits.tsv"
    if not a.skip_mmseqs:
        work = data / "mmseqs"; work.mkdir(exist_ok=True)
        for pct in (80, 50, 30):
            to_clusters_tsv(run_mmseqs(data / "domains.fasta", work, pct, a.threads), data / f"clusters_{pct}.tsv")
    dom = load_domains(data); n = len(dom); assert dom["domain_id"].is_unique
    cl = {p: load_cluster_ids(data, p, dom) for p in (80, 50, 30)}
    for p, ids in cl.items(): print(f"MMseqs2 clusters @ {p}%: {len(np.unique(ids))} for {n} domains")
    edges = load_edges(hits_path, dom); print("hit rows after filter:", len(edges))
    A = {}
    for p in (80, 50, 30):
        e = edges[(edges["fident"] >= p/100) & (edges["cov_max"] >= COV)]
        A[p] = (e["qi"].values, e["ti"].values); print(f"rule A @ {p}%: {len(e)} edges")
    eB = edges[(edges["fident"] >= 0.3) & (edges["cov_min"] >= COV)]; print("rule B @ 30%:", len(eB))
    group_of = {"id80": merge_groups(n, *A[80], [cl[80]]), "id50": merge_groups(n, *A[50], [cl[50]]),
                "id30": merge_groups(n, *A[30], [cl[30]]),
                "Hhold": merge_groups(n, *A[30], [cl[30], dom["H"].values]),
                "Thold": merge_groups(n, eB["qi"].values, eB["ti"].values, [cl[30], dom["T"].values])}
    np.save(data / "groups_id30.npy", group_of["id30"])
    for name in ALL_SPLITS:
        s = np.bincount(group_of[name])
        print(f"  {name:6s} groups {len(s):6d} | largest {s.max():5d} ({s.max()/n:.1%}) | >2%: {(s/n>0.02).sum()}")
        assert s.max()/n <= a.max_group_frac, f"{name}: largest group {s.max()/n:.1%} - chaining"
    part = edges[(edges["fident"] >= 0.8) & (edges["cov_max"] < COV) & (edges["alnlen"] >= 30)]
    pq, pt = part["qi"].values, part["ti"].values
    isC4 = dom["C"].values == "4"
    out, report = {}, {}
    for name in ALL_SPLITS:
        ei, ej = A[PCT[name]]
        for s in a.seeds:
            code0 = group_split(group_of[name], a.test_frac, s); code = embargo_test(code0, ei, ej)
            n_emb = int(((code0 == 1) & (code == -1)).sum()); out[f"{name}_seed{s}"] = code
            tr, te = code == 0, code == 1
            chk = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "n_dropped": int((code == -1).sum()),
                   "n_embargoed_test": n_emb, "test_frac_before_embargo": round(float((code0 == 1).mean()), 4),
                   "group_overlap": len(set(group_of[name][tr]) & set(group_of[name][te])),
                   "overlap_30pct_clusters": len(set(cl[30][tr]) & set(cl[30][te])),
                   "crossing_ruleA_edges": int(crossing(code, ei, ej).sum()),
                   "residual_partial_test_domains": int(np.unique(np.where(code[pq] == 1, pq, pt)[crossing(code, pq, pt)]).size),
                   "train_C4": int((tr & isC4).sum()), "test_C4": int((te & isC4).sum())}
            if name in ("Hhold", "Thold"):
                chk["H_overlap"] = len(set(dom["H"][tr]) & set(dom["H"][te]))
                chk["T_overlap"] = len(set(dom["T"][tr]) & set(dom["T"][te]))
            if name == "Hhold": assert chk["H_overlap"] == 0, (name, s, chk)
            if name == "Thold": assert chk["T_overlap"] == 0, (name, s, chk)
            assert chk["group_overlap"] == 0 and chk["crossing_ruleA_edges"] == 0, (name, s, chk)
            if name != "Thold": assert n_emb == 0, (name, s, chk)
            if name in ("Hhold", "Thold", "id30"): assert chk["overlap_30pct_clusters"] == 0, (name, s, chk)
            report[f"{name}_seed{s}"] = chk
    np.savez_compressed(data / "splits.npz", **out)
    json.dump(report, open(data / "splits_report.json", "w"), indent=2)
    json.dump({"hits_sha256": hashlib.sha256(hits_path.read_bytes()).hexdigest(), "evalue_max": EVALUE,
               "coverage_min": COV, "seeds": a.seeds, "test_frac": a.test_frac}, open(data / "splits_provenance.json", "w"), indent=2)
    pd.set_option("display.width", 250); print("\n" + pd.DataFrame(report).T.to_string())
    print("\nAll leakage assertions passed.")

if __name__ == "__main__":
    main()
