#!/usr/bin/env python3
"""Supplementary Fig. S4 input: examples of train-test leakage under a NAIVE cluster split.

Naive split  = whole 30%-identity easy-cluster clusters (data/clusters_30.tsv, -c 0.8 --cov-mode 0)
               assigned 80/20 to train/test. No rule-A/B edges, no containment merging.
Leaked pair  = test domain with an all-vs-all hit (data/allvsall_hits.tsv) to a TRAIN domain that
               satisfies our rule A at 30%: identity >= 30% and coverage >= 0.8 of either sequence.
Probe        = your 04g_probe_gpu.fit_pred (GPU L-BFGS multinomial LR, balanced weights) on topology (T),
               one layer of ESM-2 650M, trained on the naive split.

Run on a GPU node from ~/esm2_v3 (same slurm header as emb.slurm):
  python 13_leak_examples.py --out leak_ex --layer 28 --seed 0
Outputs in OUT/: summary.txt, examples.tsv (input of 13b_leak_structures.py), pred.csv, alignments.txt
Needs biopython for the alignment text (pip install biopython).
"""
import argparse, importlib.util, sys, random
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from common import load_domains, load_emb, load_cluster_ids, cath_codes
from sklearn.metrics import balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--emb", default="emb/esm2_t33_650M.npy")
ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=28)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--C", type=float, default=0.1)
ap.add_argument("--maxlen", type=int, default=160, help="max domain length for example pairs")
ap.add_argument("--classes", default="2,3", help="CATH classes allowed for examples")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("probe", "04g_probe_gpu.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)

dom = load_domains(a.data); N = len(dom)
codes = cath_codes(dom); yT = codes["T"].astype(np.int64)

# ---- naive split: whole clusters 80/20
cl = load_cluster_ids(a.data, 30, dom)
uc = np.unique(cl); rng = np.random.default_rng(a.seed); rng.shuffle(uc)
test_cl = set(uc[: int(0.2 * len(uc))].tolist())
te = np.array([c in test_cl for c in cl]); tr = ~te
print(f"naive split: train {tr.sum()}, test {te.sum()}", flush=True)

# ---- leaked pairs from all-vs-all hits
cols = ["q", "t", "pident", "aln", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]
h = pd.read_csv(Path(a.data) / "allvsall_hits.tsv", sep="\t", header=None, names=cols)
idx = pd.Series(np.arange(N), index=dom.domain_id)
h["qi"] = h.q.map(idx); h["ti"] = h.t.map(idx)
h = h.dropna(subset=["qi", "ti"]); h["qi"] = h.qi.astype(int); h["ti"] = h.ti.astype(int)
h = h[te[h.qi.values] & tr[h.ti.values]]
h = h[(h.pident >= 0.30) & ((h.qcov >= 0.8) | (h.tcov >= 0.8))]
best = h.sort_values("bits", ascending=False).drop_duplicates("qi")
leaked = np.zeros(N, bool); leaked[best.qi.values] = True
# contained: test domain fully inside a longer training domain (qcov>=0.8, tlen > qlen), hit rule A
cont = np.zeros(N, bool)
cc = h[(h.qcov >= 0.8) & (h.tlen > h.qlen)]; cont[cc.qi.values] = True
n_te = int(te.sum())
lines = [f"test domains: {n_te}",
         f"test domains with a rule-A hit to a training domain (leaked): {int((leaked & te).sum())} ({(leaked & te).sum()/n_te:.1%})",
         f"  of which contained in a longer training domain: {int((cont & te).sum())} ({(cont & te).sum()/n_te:.1%})"]

# ---- probe on the naive split (topology)
arr = load_emb(a.emb)
X = torch.from_numpy(np.ascontiguousarray(arr[:, a.layer, :], dtype=np.float32)).to("cuda")
tri, tei = np.where(tr)[0], np.where(te)[0]
Xtr, Xte = X[torch.as_tensor(tri, device="cuda")], X[torch.as_tensor(tei, device="cuda")]
mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False); sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
pred_te, nit, gn = probe.fit_pred(Xtr, yT[tri], Xte, a.C, 2000)
pred = np.full(N, -1); pred[tei] = pred_te
seen = np.isin(yT[tei], yT[tri])
ok = pred_te == yT[tei]
lk = leaked[tei]
lines += [f"probe: 650M layer {a.layer}, C={a.C}, iterations {nit}, max gradient {gn:.1e}",
          f"topology balanced accuracy on naive-split test set: {balanced_accuracy_score(yT[tei], pred_te):.3f}",
          f"plain accuracy, test domains whose topology occurs in training: {ok[seen].mean():.3f}",
          f"  leaked test domains (n={int((lk & seen).sum())}): {ok[lk & seen].mean():.3f}",
          f"  non-leaked test domains (n={int((~lk & seen).sum())}): {ok[~lk & seen].mean():.3f}"]
print("\n".join(lines)); (out / "summary.txt").write_text("\n".join(lines) + "\n")

# ---- pick examples: same topology, high coverage, short, correctly predicted, three identity bands
c = best.copy()
c["same_T"] = yT[c.qi.values] == yT[c.ti.values]
c["len"] = dom.length.values[c.qi.values]; c["tl"] = dom.length.values[c.ti.values]
c["ok"] = pred[c.qi.values] == yT[c.qi.values]
c = c[c.same_T & c.ok & (c.qcov >= 0.9) & (c.tcov >= 0.9) & (c.len <= a.maxlen) & (c.tl <= a.maxlen)]
_q = dom.domain_id.values[c.qi.values]; _t = dom.domain_id.values[c.ti.values]
_qs = dom.seq.values[c.qi.values]; _ts = dom.seq.values[c.ti.values]; _cls = dom.C.values[c.qi.values]
_HEAD = ("GSSGSSG", "MGSSHHHHHH", "GSHM", "GPLGS", "MHHHHHH"); _TAIL = ("SGPSSG", "HHHHHH")
def _tagged(x): return x.startswith(_HEAD) or x.endswith(_TAIL)
keep = np.array([(q[:4] != t[:4]) and not _tagged(x) and not _tagged(y) and k in a.classes.split(",")
                 for q, t, x, y, k in zip(_q, _t, _qs, _ts, _cls)], dtype=bool)
c = c[keep]
rows = []
for lo, hi in [(0.30, 0.45), (0.45, 0.70), (0.70, 1.01)]:
    cb = c[(c.pident >= lo) & (c.pident < hi)]
    if len(cb): rows.append(cb.sample(1, random_state=a.seed).iloc[0])
ex = pd.DataFrame(rows)
ex["pident"] = ex.pident * 100
for lv in "CATH":
    ex["q" + lv] = dom[lv].values[ex.qi.values]; ex["t" + lv] = dom[lv].values[ex.ti.values]
ex["query"], ex["target"] = ex.q, ex.t
ex[["query", "target", "pident", "qcov", "tcov", "evalue", "bits", "qC", "qA", "qT", "qH", "tC", "tA", "tT", "tH"]].to_csv(out / "examples.tsv", sep="\t", index=False)
seqs = dict(zip(dom.domain_id, dom.seq)); Tname = dom["T"].values
pr = pd.DataFrame({"domain": ex['query'].values, "true_T": Tname[ex.qi.values],
                   "pred_T": [Tname[np.where(yT == pred[i])[0][0]] for i in ex.qi.values]})
pr.to_csv(out / "pred.csv", index=False)
print(ex[["query", "target", "pident", "qcov", "tcov", "evalue", "qT", "tT"]].to_string())

# ---- alignments (Smith-Waterman, BLOSUM62) for the legend / figure
try:
    from Bio.Align import PairwiseAligner, substitution_matrices
    al = PairwiseAligner(); al.mode = "local"; al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    with open(out / "alignments.txt", "w") as f:
        for _, r in ex.iterrows():
            aln = al.align(seqs[r['query']], seqs[r.target])[0]
            f.write(f"# test {r['query']} vs train {r.target}  (MMseqs2 identity {r.pident:.1f}%, E={r.evalue:.1e})\n{aln}\n\n")
except ImportError:
    print("biopython missing: pip install biopython (alignments.txt not written)")

with open(out / "chimerax.cxc", "w") as f:
    f.write("set bgColor white\nlighting soft\n")
    for _, r in ex.iterrows():
        q, t = r["query"], r["target"]
        u = "https://www.cathdb.info/version/v4_3_0/api/rest/id/{}.pdb"
        f.write(f"close session\nopen {u.format(t)} format pdb name train\nopen {u.format(q)} format pdb name test\n"
                f"hide atoms\nshow cartoons\ncolor #1 darkorange target c\ncolor #2 dodgerblue target c\n"
                f"matchmaker #2 to #1\nview\nsave {q}_vs_{t}.png width 1400 supersample 3\n")
print("wrote", out / "chimerax.cxc")
