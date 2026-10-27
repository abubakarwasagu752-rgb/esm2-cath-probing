#!/usr/bin/env python3
"""GPU multinomial logistic regression, balanced class weights, L2, train-only standardisation.
C tuned on a group-held-out part of the training set. Saves test predictions + convergence info."""
import argparse
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from common import ALL_SPLITS, SPLIT_TASKS, load_domains, load_emb, load_split_codes
DEV = "cuda"

def fit_lr(X, yi, K, C, max_iter):
    n, d = X.shape
    cnt = torch.bincount(yi, minlength=K).float(); sw = (n / (K * cnt))[yi]
    W = torch.zeros(d, K, device=DEV, requires_grad=True); b = torch.zeros(K, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-4, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")
    lam = 0.5 / (C * n)
    def closure():
        opt.zero_grad()
        loss = (sw * torch.nn.functional.cross_entropy(X @ W + b, yi, reduction="none")).sum() / n + lam * (W * W).sum()
        loss.backward(); return loss
    opt.step(closure)
    gn = max(W.grad.abs().max().item(), b.grad.abs().max().item())
    return W.detach(), b.detach(), int(opt.state[opt._params[0]]["n_iter"]), gn

def fit_pred(Xa, ya, Xb, C, max_iter):
    cls, yi = np.unique(ya, return_inverse=True)
    W, b, nit, gn = fit_lr(Xa, torch.as_tensor(yi.ravel(), device=DEV), len(cls), C, max_iter)
    return cls[(Xb @ W + b).argmax(1).cpu().numpy()].astype(np.int32), nit, gn

def run_layer(arr, layer, tr, te, y, g, grid, tune, shuffle, seed, c_fixed, max_iter):
    X = torch.from_numpy(np.ascontiguousarray(arr[:, layer, :], dtype=np.float32)).to(DEV)
    Xtr, Xte = X[torch.as_tensor(tr, device=DEV)], X[torch.as_tensor(te, device=DEV)]
    ytr = y[tr].copy()
    if shuffle: ytr = np.random.default_rng(seed).permutation(ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False); sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    best = c_fixed
    if tune == "val":
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(np.zeros(len(tr)), groups=g[tr]))
        ta, tb = torch.as_tensor(a, device=DEV), torch.as_tensor(b, device=DEV); bs = -1.0
        for C in grid:
            p, _, _ = fit_pred(Xtr[ta], ytr[a], Xtr[tb], C, max_iter)
            s = balanced_accuracy_score(ytr[b], p)
            if s > bs: bs, best = s, C
    pred, nit, gn = fit_pred(Xtr, ytr, Xte, best, max_iter)
    return pred, best, nit, gn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS); ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--tasks", nargs="+", default=None); ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--tune", choices=["val","none"], default="val")
    ap.add_argument("--C-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--C", type=float, default=1.0); ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--shuffle-control", action="store_true")
    a = ap.parse_args()
    dom = load_domains(a.data); g30 = np.load(f"{a.data}/groups_id30.npy"); assert len(g30) == len(dom)
    arr = load_emb(a.emb); layers = a.layers if a.layers is not None else list(range(arr.shape[1]))
    outdir = Path(a.out) / f"probe_{a.tag}"; outdir.mkdir(parents=True, exist_ok=True)
    suf = "_shuf" if a.shuffle_control else ""
    for split in a.splits:
        for seed in a.seeds:
            code = load_split_codes(a.data, split, seed); tr, te_all = np.where(code == 0)[0], np.where(code == 1)[0]
            for task in (a.tasks or SPLIT_TASKS[split]):
                if task not in SPLIT_TASKS[split]: continue
                dest = outdir / f"{split}_s{seed}_{task}{suf}.npz"
                if dest.exists(): print("skip", dest.name, flush=True); continue
                classes, y = np.unique(dom[task].to_numpy(), return_inverse=True); y = y.ravel()
                keep = np.isin(y[te_all], np.unique(y[tr])); te = te_all[keep]
                print(f"{a.tag} {split} s{seed} {task}: train={len(tr)} test={len(te)} dropped={(~keep).sum()} classes={len(np.unique(y[tr]))}", flush=True)
                res = [run_layer(arr, l, tr, te, y, g30, a.C_grid, a.tune, a.shuffle_control, seed, a.C, a.max_iter) for l in layers]
                np.savez_compressed(dest, preds=np.stack([r[0] for r in res]), chosen_C=np.array([r[1] for r in res]),
                                    n_iter=np.array([r[2] for r in res]), max_grad=np.array([r[3] for r in res]),
                                    layers=np.array(layers), y=y[te], g=g30[te], te_idx=te, classes=classes,
                                    frac_dropped=float((~keep).mean()))
    print("done")

if __name__ == "__main__":
    main()
