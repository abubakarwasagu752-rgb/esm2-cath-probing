#!/usr/bin/env python3
"""Step 1 - build the CATH v4.3.0 domain dataset (nested caps per superfamily)."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CANON = set("ACDEFGHIKLMNPQRSTVWY")


def read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            c, a, t, h = p[1], p[2], p[3], p[4]
            rows.append((p[0], c, f"{c}.{a}", f"{c}.{a}.{t}", f"{c}.{a}.{t}.{h}"))
    return pd.DataFrame(rows, columns=["domain_id", "C", "A", "T", "H"])


def read_fasta(path):
    seqs, cur, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur, buf = line.split("|")[2].split("/")[0], []
            else:
                buf.append(line)
    if cur is not None:
        seqs[cur] = "".join(buf)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="data/cath-domain-list-v4_3_0.txt")
    ap.add_argument("--fasta", default="data/cath-domain-seqs-v4_3_0.fa")
    ap.add_argument("--caps", type=int, nargs="+", default=[5, 8, 10, 15])
    ap.add_argument("--main-cap", type=int, default=8)
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df = read_list(a.list)
    seqs = read_fasta(a.fasta)
    stats = {"list_rows": len(df), "fasta_seqs": len(seqs)}
    assert df["domain_id"].is_unique
    df["seq"] = df["domain_id"].map(seqs)
    stats["missing_sequence"] = int(df["seq"].isna().sum())
    df = df.dropna(subset=["seq"])
    df["length"] = df["seq"].str.len()
    stats["class_not_1_4"] = int((~df["C"].isin(list("1234"))).sum())
    df = df[df["C"].isin(list("1234"))]
    stats["non_canonical"] = int(df["seq"].map(lambda s: not set(s) <= CANON).sum())
    df = df[df["seq"].map(lambda s: set(s) <= CANON)]
    stats["length_out_of_range"] = int(((df["length"] < a.min_len) | (df["length"] > a.max_len)).sum())
    df = df[(df["length"] >= a.min_len) & (df["length"] <= a.max_len)]
    df = df.sort_values("domain_id").reset_index(drop=True)
    g = df.groupby("seq")["H"].nunique()
    stats["duplicate_seq_groups"] = int((df.groupby("seq").size() > 1).sum())
    stats["duplicate_seq_with_conflicting_H"] = int((g > 1).sum())
    stats["duplicate_rows_removed"] = int(df.duplicated("seq").sum())
    df = df.drop_duplicates("seq", keep="first").reset_index(drop=True)
    stats["eligible_domains"] = len(df)
    stats["eligible_superfamilies"] = int(df["H"].nunique())

    rng = np.random.default_rng(a.seed)
    order = {}
    for h in sorted(df["H"].unique()):
        idx = df.index[df["H"] == h].to_numpy()
        order[h] = rng.permutation(idx)

    for cap in a.caps:
        keep = np.concatenate([v[:cap] for v in order.values()])
        sub = df.loc[np.sort(keep)].reset_index(drop=True)
        out = Path("data" if cap == a.main_cap else f"data_cap{cap}")
        out.mkdir(exist_ok=True)
        sub[["domain_id", "C", "A", "T", "H", "seq", "length"]].to_csv(out / "domains.tsv", sep="\t", index=False)
        with open(out / "domains.fasta", "w") as f:
            for d, s in zip(sub["domain_id"], sub["seq"]):
                f.write(f">{d}\n{s}\n")
        st = dict(stats, cap=cap, n_domains=len(sub), n_C=int(sub["C"].nunique()), n_A=int(sub["A"].nunique()),
                  n_T=int(sub["T"].nunique()), n_H=int(sub["H"].nunique()),
                  class_counts=sub["C"].value_counts().sort_index().to_dict(),
                  median_length_by_class=sub.groupby("C")["length"].median().to_dict(),
                  seed=a.seed, min_len=a.min_len, max_len=a.max_len)
        json.dump(st, open(out / "dataset_stats.json", "w"), indent=2, default=int)
        print(f"cap {cap}: {len(sub)} domains | C {st['n_C']} A {st['n_A']} T {st['n_T']} H {st['n_H']} | "
              f"classes {st['class_counts']} -> {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
