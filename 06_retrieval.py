import argparse, numpy as np, torch
from pathlib import Path
from common import load_domains, load_emb, load_split_codes, cath_codes
ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--splits", nargs="+", default=["id30", "Hhold", "Thold"]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--out", default="results_v2")
a = ap.parse_args()
dev = "cuda"; dom = load_domains("data"); codes = cath_codes(dom); arr = load_emb(a.emb); L = arr.shape[1]
out = Path(a.out) / f"retrieval_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
for split in a.splits:
    for seed in a.seeds:
        dest = out / f"{split}_s{seed}.npz"
        if dest.exists(): print("skip", dest.name, flush=True); continue
        code = load_split_codes("data", split, seed); tr = np.where(code == 0)[0]; te = np.where(code == 1)[0]
        nn = np.zeros((L, len(te)), dtype=np.int64)
        for l in range(L):
            X = torch.from_numpy(np.ascontiguousarray(arr[:, l, :], dtype=np.float32)).to(dev)
            Xtr, Xte = X[torch.as_tensor(tr, device=dev)], X[torch.as_tensor(te, device=dev)]
            mu, sd = Xtr.mean(0), Xtr.std(0, unbiased=False); sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            Xtr = Xtr / Xtr.norm(dim=1, keepdim=True).clamp_min(1e-8); Xte = Xte / Xte.norm(dim=1, keepdim=True).clamp_min(1e-8)
            for s0 in range(0, len(te), 1000):
                nn[l, s0:s0 + 1000] = tr[(Xte[s0:s0 + 1000] @ Xtr.T).argmax(1).cpu().numpy()]
        np.savez_compressed(dest, nn=nn, te_idx=te, tr_idx=tr)
        print(a.tag, split, seed, "done", flush=True)
