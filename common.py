"""Shared helpers for the ESM-2 / CATH probing pipeline (v3)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

LEVELS = ["C", "A", "T", "H"]
PROBE_TASKS = ["C", "A", "T"]
SPLIT_TASKS = {
    "id80": ["C", "A", "T"], "id50": ["C", "A", "T"], "id30": ["C", "A", "T"],
    "Hhold": ["C", "A", "T"], "Thold": ["C", "A"],
}
ALL_SPLITS = list(SPLIT_TASKS)


def load_domains(data):
    df = pd.read_csv(Path(data) / "domains.tsv", sep="\t", dtype=str)
    df["length"] = df["length"].astype(int)
    return df


def load_cluster_ids(data, pct, domains):
    cl = pd.read_csv(Path(data) / f"clusters_{pct}.tsv", sep="\t", dtype=str)
    m = dict(zip(cl["domain_id"], cl["cluster_id"]))
    ids = domains["domain_id"].map(m)
    if ids.isna().any():
        raise ValueError(f"clusters_{pct}.tsv does not cover all domains")
    return pd.factorize(ids)[0]


def load_split_codes(data, split, seed):
    """int8 per domain: 0 = train, 1 = test, -1 = embargoed."""
    with np.load(Path(data) / "splits.npz") as z:
        return z[f"{split}_seed{seed}"]


def load_emb(path):
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected (N, layers, D), got {arr.shape}")
    return arr


def cath_codes(domains):
    return {lv: pd.factorize(domains[lv])[0] for lv in LEVELS}


def cath_distance_matrix(codes, idx):
    c, a, t, h = (codes[k][idx] for k in LEVELS)
    d = np.full((len(idx), len(idx)), 4, dtype=np.int8)
    d[c[:, None] == c[None, :]] = 3
    d[a[:, None] == a[None, :]] = 2
    d[t[:, None] == t[None, :]] = 1
    d[h[:, None] == h[None, :]] = 0
    return d
