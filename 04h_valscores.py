import argparse, importlib.util, numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import torch
from common import ALL_SPLITS, SPLIT_TASKS, load_domains, load_emb, load_split_codes
spec = importlib.util.spec_from_file_location("p04", "04g_probe_gpu.py"); p04 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p04)
ap = argparse.ArgumentParser(); ap.add_argument("--emb", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="results_v2"); a = ap.parse_args()
G = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0]
dom = load_domains("data"); g30 = np.load("data/groups_id30.npy"); arr = load_emb(a.emb); L = arr.shape[1]; dev = "cuda"
out = Path(a.out) / f"val_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
for split in ALL_SPLITS:
    grp = {"Hhold": dom["H"].values, "Thold": dom["T"].values}.get(split, g30)
    for seed in range(3):
        code = load_split_codes("data", split, seed); tr = np.where(code == 0)[0]
        for task in SPLIT_TASKS[split]:
            dest = out / f"{split}_s{seed}_{task}.npy"
            if dest.exists(): continue
            _, y = np.unique(dom[task].to_numpy(), return_inverse=True); y = y.ravel()
            f, v = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(np.zeros(len(tr)), groups=grp[tr]))
            f, v = tr[f], tr[v]; keep = np.isin(y[v], np.unique(y[f])); v = v[keep]   # unseen-label val domains dropped, as for test
            sc = np.zeros(L)
            for l in range(L):
                X = torch.from_numpy(np.ascontiguousarray(arr[:, l, :], dtype=np.float32)).to(dev)
                Xf, Xv = X[torch.as_tensor(f, device=dev)], X[torch.as_tensor(v, device=dev)]
                mu, sd = Xf.mean(0), Xf.std(0, unbiased=False); sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
                Xf, Xv = (Xf - mu) / sd, (Xv - mu) / sd
                sc[l] = max(balanced_accuracy_score(y[v], p04.fit_pred(Xf, y[f], Xv, C, 2000)[0]) for C in G)
            np.save(dest, sc); print(a.tag, split, seed, task, "best val layer", int(sc.argmax()), flush=True)
