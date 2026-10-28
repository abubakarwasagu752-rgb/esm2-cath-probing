import ast
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"axes.grid": False, "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8})
models = ["esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"]
MC = dict(zip(models, ["#4C78A8", "#F58518", "#54A24B"])); MN = dict(zip(models, ["ESM-2 35M", "ESM-2 150M", "ESM-2 650M"]))
try:  # reuse the palette and names from 09_figures.py when they can be read as literals
    for node in ast.parse(open("09_figures.py").read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in ("MC", "MN"):
            v = ast.literal_eval(node.value)
            if all(m in v for m in models): (MC if node.targets[0].id == "MC" else MN).update(v)
except Exception as e:
    print("palette fallback:", e)
d = pd.read_csv("tables/phase4_selected_layer.csv")
try:
    pk = pd.read_csv("tables/probe_peaks.csv"); comp = pk[pk.model == "composition"]
except Exception: comp = pd.DataFrame()
order = ["id80", "id50", "id30", "Hhold", "Thold"]; TN = {"C": "Class", "A": "Architecture", "T": "Topology"}; w = .26
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
for j, t in enumerate("CAT"):
    for k, m in enumerate(models):
        x, v, lo, hi = [], [], [], []
        for i, s in enumerate(order):
            r = d[(d.model == m) & (d.task == t) & (d.split == s)]
            if r.empty: continue
            r = r.iloc[0]; x.append(i + (k - 1) * w); v.append(r.test_bal_acc); lo.append(r.test_bal_acc - r.lo); hi.append(r.hi - r.test_bal_acc)
        ax[j].bar(x, v, w, color=MC[m], yerr=[lo, hi], error_kw=dict(lw=.6, capsize=1.2, ecolor="#333"))
    if not comp.empty:
        for i, s in enumerate(order):
            r = comp[(comp.task == t) & (comp.split == s)]
            if len(r): ax[j].hlines(r.peak_bal_acc.iloc[0], i - .42, i + .42, color="k", ls="-.", lw=.9)
    ax[j].set_xticks(range(5)); ax[j].set_xticklabels(order, fontsize=7); ax[j].set_title(TN[t] + " (selected layer)", fontsize=8); ax[j].set_ylim(0, 1)
    if j == 0: ax[j].set_ylabel("balanced accuracy")
h = [Patch(color=MC[m], label=MN[m]) for m in models]
if not comp.empty: h.append(plt.Line2D([0], [0], color="k", ls="-.", lw=.9, label="composition + length"))
fig.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.855, 0.80))
fig.tight_layout(rect=(0, 0, 0.85, 1))
for ext in ("png", "pdf", "svg"): fig.savefig(f"figures/fig3_split_gap.{ext}", bbox_inches="tight")
fig.savefig("figures/fig3_split_gap.tif", dpi=350, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
print("wrote figures/fig3_split_gap.[png|pdf|svg|tif]; composition lines:", not comp.empty)
