import glob, sys, numpy as np, warnings
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
tag = sys.argv[1]
for split in ["id80","id50","id30","Hhold","Thold"]:
    for task in "CAT":
        fs = [f for f in sorted(glob.glob(f"results/probe_{tag}/{split}_s*_{task}.npz")) if "_shuf" not in f]
        if not fs: continue
        c = np.mean([[balanced_accuracy_score(z["y"], p) for p in z["preds"]] for z in map(np.load, fs)], 0)
        print(f"{split:5s} {task}: " + " ".join(f"{v:.2f}" for v in c) + f"   (best layer {int(c.argmax())})")
