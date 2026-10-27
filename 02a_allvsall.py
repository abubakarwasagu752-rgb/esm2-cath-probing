#!/usr/bin/env python3
"""Step 2a - all-vs-all MMseqs2 search over every domain."""
import argparse, os, subprocess, tempfile
from pathlib import Path
from common import load_domains
COLS = ["query","target","fident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
MMSEQS = os.environ.get("MMSEQS", str(Path.home() / "mmseqs/bin/mmseqs"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    data = Path(a.data); dom = load_domains(data); out = data / "allvsall_hits.tsv"
    subprocess.run([MMSEQS, "version"], check=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as tmp:
        fa = Path(tmp) / "all.fa"
        with open(fa, "w") as f:
            for d, s in zip(dom["domain_id"], dom["seq"]): f.write(f">{d}\n{s}\n")
        subprocess.run([MMSEQS, "easy-search", str(fa), str(fa), str(out), str(Path(tmp) / "w"),
                        "--min-seq-id", "0.3", "-e", "1e-3", "-s", "7.5", "--max-seqs", "1000",
                        "--threads", str(a.threads), "--format-output", ",".join(COLS)],
                       check=True, stdout=subprocess.DEVNULL)
    print("wrote", out)

if __name__ == "__main__":
    main()
