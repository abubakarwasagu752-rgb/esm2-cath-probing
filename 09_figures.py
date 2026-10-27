import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- save an SVG copy of every figure next to its PNG/PDF
_orig_savefig = matplotlib.figure.Figure.savefig
def _savefig_all(self, fname, *a, **k):
    _orig_savefig(self, fname, *a, **k)
    if str(fname).endswith(".png"):
        k2 = {k_: v for k_, v in k.items() if k_ != "dpi"}
        _orig_savefig(self, str(fname)[:-4] + ".svg", **k2)
        w, h = self.get_size_inches(); self.set_size_inches(7.0, 7.0 * h / w)
        _orig_savefig(self, str(fname)[:-4] + ".tif", dpi=350, **{k_: v for k_, v in k2.items()}, pil_kwargs={"compression": "tiff_lzw"})
        self.set_size_inches(w, h)
matplotlib.figure.Figure.savefig = _savefig_all
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "figure.dpi": 300, "savefig.dpi": 300})
LS = {"id80": ":", "id50": "-.", "id30": "-", "Hhold": "--", "Thold": (0, (5, 1, 1, 1))}
SC = {"id80": "#56B4E9", "id50": "#0072B2", "id30": "#009E73", "Hhold": "#D55E00", "Thold": "#CC79A7"}
MC = {"esm2_t12_35M": "#0072B2", "esm2_t30_150M": "#009E73", "esm2_t33_650M": "#D55E00"}
MN = {"esm2_t12_35M": "ESM-2 35M", "esm2_t30_150M": "ESM-2 150M", "esm2_t33_650M": "ESM-2 650M"}
TN = {"C": "Class", "A": "Architecture", "T": "Topology"}
P = pd.read_csv("tables/probe_by_layer.csv"); K = pd.read_csv("tables/probe_peaks.csv"); Rt = pd.read_csv("tables/retrieval_by_layer.csv")
models = list(MN)
# 1 probe by split
fig, ax = plt.subplots(3, 3, figsize=(11, 8.5))
for i, m in enumerate(models):
    for j, t in enumerate("CAT"):
        for sp in SC:
            d = P[(P.model == m) & (P.task == t) & (P.split == sp)]
            if d.empty: continue
            ax[i, j].plot(d.layer, d.bal_acc, color=SC[sp], ls=LS[sp], lw=1.4, label=sp); ax[i, j].fill_between(d.layer, d.lo, d.hi, color=SC[sp], alpha=.15, lw=0)
        ax[i, j].set_title(f"{MN[m]} - {TN[t]}", fontsize=9)
        if i == 2: ax[i, j].set_xlabel("layer")
        if j == 0: ax[i, j].set_ylabel("balanced accuracy")
ax[0, 0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig1_probe_splits.png"); fig.savefig("figures/fig1_probe_splits.pdf"); plt.close(fig)
# 2 baselines (Hhold)
fig, ax = plt.subplots(3, 3, figsize=(11, 8.5))
for i_, m in enumerate(models):
    for j_, t in enumerate("CAT"):
        d = P[(P.model == m) & (P.task == t) & (P.split == "Hhold")]; a = ax[i_, j_]
        a.plot(d.layer, d.bal_acc, color=MC[m], lw=1.6, label=MN[m]); a.fill_between(d.layer, d.lo, d.hi, color=MC[m], alpha=.15, lw=0)
        a.axhline(d.composition.iloc[0], color="k", ls="-.", lw=1, label="composition+length")
        if d.random_mean.notna().any():
            a.plot(d.layer, d.random_mean, color="#666", ls="--", lw=1.1, label="random-init (35M arch.)"); a.fill_between(d.layer, d.random_min, d.random_max, color="#666", alpha=.15, lw=0)
        if d.shuffled.notna().any() and t != "T": a.axhline(d.shuffled.iloc[0], color="#999", ls=":", lw=1, label="shuffled labels (35M)")
        a.set_title(f"{MN[m]} - {TN[t]} (Hhold)", fontsize=9)
        if i_ == 2: a.set_xlabel("layer")
        if j_ == 0: a.set_ylabel("balanced accuracy")
ax[0, 0].legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(0.12, 0.38))
fig.tight_layout(); fig.savefig("figures/fig2_baselines.png"); fig.savefig("figures/fig2_baselines.pdf"); plt.close(fig)
# 3 split gap
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6)); order = ["id80", "id50", "id30", "Hhold", "Thold"]; w = .26
for j_, t in enumerate("CAT"):
    for k, m in enumerate(models):
        v = [K[(K.model == m) & (K.task == t) & (K.split == s_)].peak_bal_acc.mean() for s_ in order]
        ax[j_].bar(np.arange(5) + (k - 1) * w, v, w, color=MC[m], label=MN[m])
    ax[j_].set_xticks(range(5)); ax[j_].set_xticklabels(order); ax[j_].set_title(TN[t] + " (peak over layers)"); ax[j_].set_ylabel("balanced accuracy")
h, l_ = ax[2].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 0.86, 1))
fig.legend(h, l_, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.865, 0.80))
fig.savefig("figures/fig3_split_gap.png", bbox_inches="tight"); fig.savefig("figures/fig3_split_gap.pdf", bbox_inches="tight"); plt.close(fig)
# 4 retrieval
fig, ax = plt.subplots(3, 3, figsize=(11, 8.5))
for i, m in enumerate(models):
    for j, t in enumerate("CAT"):
        for sp in ["id30", "Hhold", "Thold"]:
            d = Rt[(Rt.tag == m) & (Rt.split == sp) & (Rt.metric == t)]
            if d.empty or d.value.isna().all(): continue
            ax[i, j].plot(d.layer, d.value, color=SC[sp], ls=LS[sp], lw=1.4, label=sp); ax[i, j].fill_between(d.layer, d.lo, d.hi, color=SC[sp], alpha=.15, lw=0)
        c = Rt[(Rt.tag == "composition") & (Rt.split == "Hhold") & (Rt.metric == t)]
        if not c.empty and c.value.notna().any(): ax[i, j].axhline(c.value.iloc[0], color="k", ls="-.", lw=1)
        ax[i, j].set_title(f"{MN[m]} - 1-NN {TN[t]}", fontsize=9)
        if i == 2: ax[i, j].set_xlabel("layer")
        if j == 0: ax[i, j].set_ylabel("top-1 match")
ax[0, 0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig4_retrieval.png"); fig.savefig("figures/fig4_retrieval.pdf"); plt.close(fig)
# 5 RSA
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m in models:
    d = pd.read_csv(f"results_v2/rsa_{m}_Hhold.csv").groupby("layer").spearman.agg(["mean", "min", "max"]); x = d.index / d.index.max()
    ax.plot(x, d["mean"], color=MC[m], label=MN[m]); ax.fill_between(x, d["min"], d["max"], color=MC[m], alpha=.15, lw=0)
ax.axhline(pd.read_csv("results_v2/rsa_composition_Hhold.csv").spearman.mean(), color="k", ls="-.", lw=1, label="composition+length")
r = np.mean([pd.read_csv(f"results_v2/rsa_esm2_t12_35M_random_s{s_}_Hhold.csv").spearman.mean() for s_ in range(3)]); ax.axhline(r, color="#666", ls="--", lw=1, label="random-init (35M)")
ax.set_xlabel("relative depth"); ax.set_ylabel("Spearman with CATH tree distance")
ax.set_title("RSA, Hhold test side\n(band = range over 5 draws)", fontsize=9, loc="left", pad=8)
ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, borderaxespad=0.0)
fig.tight_layout(); fig.savefig("figures/fig5_rsa.png", bbox_inches="tight"); fig.savefig("figures/fig5_rsa.pdf", bbox_inches="tight"); plt.close(fig)
# 6 CKA
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for k, m in enumerate(models):
    M = np.load(f"results_v2/cka_{m}__{m}.npy"); im = ax[k].imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis"); ax[k].set_title(MN[m] + " layer-by-layer CKA", fontsize=8); ax[k].set_xlabel("layer"); ax[k].set_ylabel("layer"); ax[k].grid(False)
    ax[3].plot(np.arange(M.shape[0]) / (M.shape[0] - 1), M[:, -1], color=MC[m], label=MN[m])
fig.colorbar(im, ax=ax[2], fraction=.046); ax[3].set_xlabel("relative depth"); ax[3].set_ylabel("CKA with last layer"); ax[3].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figures/fig6_cka.png"); fig.savefig("figures/fig6_cka.pdf"); plt.close(fig)
# 7 relative depth overlay
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for a, (sp, t) in zip(ax, [("Hhold", "A"), ("Hhold", "T"), ("id30", "A"), ("id30", "T")]):
    for m in models:
        d = P[(P.model == m) & (P.task == t) & (P.split == sp)]; x = d.layer / d.layer.max()
        a.plot(x, d.bal_acc, color=MC[m], label=MN[m]); a.fill_between(x, d.lo, d.hi, color=MC[m], alpha=.12, lw=0)
    a.set_title(f"{TN[t]}, {sp}", fontsize=9); a.set_xlabel("relative depth"); a.set_ylabel("balanced accuracy")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figures/fig7_reldepth.png"); fig.savefig("figures/fig7_reldepth.pdf"); plt.close(fig)
print("figures written")
