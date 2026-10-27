#!/usr/bin/env python3
"""Step 2b - independent leakage: fresh MMseqs2 search of test vs train."""
import argparse, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from common import ALL_SPLITS, load_domains, load_split_codes
MMSEQS = os.environ.get("MMSEQS", str(Path.home() / "mmseqs/bin/mmseqs"))
COLS = ["query","target","fident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
THR = {"id80": .8, "id50": .5, "id30": .3, "Hhold": .3, "Thold": .3}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="phase1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2]); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    dom = load_domains(a.data); rows = []
    for sp in ALL_SPLITS:
        for s in a.seeds:
            code = load_split_codes(a.data, sp, s)
            tr, te = np.where(code == 0)[0], np.where(code == 1)[0]
            hp = out / f"hits_{sp}_s{s}.tsv"
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
                for nm, ix in (("q", te), ("t", tr)):
                    with open(f"{tmp}/{nm}.fa", "w") as f:
                        for i in ix: f.write(f">{dom.domain_id.iloc[i]}\n{dom.seq.iloc[i]}\n")
                subprocess.run([MMSEQS, "easy-search", f"{tmp}/q.fa", f"{tmp}/t.fa", str(hp), f"{tmp}/w",
                                "--min-seq-id", "0.0", "-e", "10", "-s", "7.5", "--max-seqs", "500",
                                "--threads", str(a.threads), "--format-output", ",".join(COLS)],
                               check=True, stdout=subprocess.DEVNULL)
            h = pd.read_csv(hp, sep="\t", header=None, names=COLS); sig = h[h.evalue <= 1e-3]
            cmax = sig[["qcov","tcov"]].max(axis=1)
            crit = sig[(sig.fident >= THR[sp]) & (cmax >= .8)]
            part = sig[(sig.fident >= .8) & (cmax < .8) & (sig.alnlen >= 30)]
            nq = len(te)
            r = dict(split=sp, seed=s, n_test=nq, crit_domains=crit["query"].nunique(),
                     partial_domains=part["query"].nunique(), partial_frac=part["query"].nunique()/nq,
                     frac_any_sig_hit=sig["query"].nunique()/nq)
            rows.append(r); print(sp, f"s{s}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("split","seed")}, flush=True)
    pd.DataFrame(rows).to_csv(out / "leakage_summary.csv", index=False)

if __name__ == "__main__":
    main()
