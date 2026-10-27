import sys, numpy as np, torch, transformers
from transformers import AutoTokenizer, EsmModel
from common import load_domains
emb_path, name = sys.argv[1], sys.argv[2]
dom = load_domains("data"); n = len(dom); E = np.load(emb_path, mmap_mode="r"); ok = True
def check(t, c, info=""):
    global ok; print(("PASS  " if c else "FAIL  ") + t, info, flush=True); ok = ok and bool(c)
print("torch", torch.__version__, "| transformers", transformers.__version__)
model = EsmModel.from_pretrained(name, add_pooling_layer=False).eval().cuda()
L, D = model.config.num_hidden_layers, model.config.hidden_size
tok = AutoTokenizer.from_pretrained(name)
check("shape", E.shape == (n, L + 1, D), str(E.shape)); check("dtype float16", E.dtype == np.float16)
check("right padding", tok.padding_side == "right"); check("max length <= 1022", dom.length.max() <= 1022)
bad = zero = 0; amax = 0.0
for s in range(0, n, 2000):
    c = np.asarray(E[s:s+2000]).astype(np.float32)
    bad += int((~np.isfinite(c)).sum()); zero += int((np.abs(c).reshape(len(c), -1).max(1) == 0).sum()); amax = max(amax, float(np.abs(c).max()))
check("finite", bad == 0, str(bad)); check("no empty rows", zero == 0, str(zero)); check("float16 headroom", amax < 30000, f"{amax:.1f}")
cap = {}
hook = model.encoder.layer[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("raw", (o[0] if isinstance(o, tuple) else o).detach()))
enc = tok([dom.seq.iloc[0]], return_tensors="pt").to("cuda")
with torch.inference_mode():
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states; ln_raw = model.encoder.emb_layer_norm_after(cap["raw"])
hook.remove()
check("len(hidden_states)==L+1", len(hs) == L + 1)
check("last == last_hidden_state", torch.allclose(hs[-1], out.last_hidden_state, atol=1e-5))
check("last == LayerNorm(raw last block)", torch.allclose(ln_raw, hs[-1], atol=1e-4))
check("LayerNorm really applied", not torch.allclose(cap["raw"], hs[-1], atol=1e-3))
rng = np.random.default_rng(0); by = dom.length.values.argsort()
idx = list(by[:3]) + list(by[-3:]) + list(np.where(dom.C.values == "4")[0][:3]) + list(rng.choice(n, 6, replace=False))
wc, wr = 1.0, 0.0
with torch.inference_mode():
    for i in idx:
        h = model(**tok([dom.seq.iloc[i]], return_tensors="pt").to("cuda"), output_hidden_states=True).hidden_states
        ref = torch.stack([x[0, 1:-1].float().mean(0) for x in h]).cpu().numpy(); got = np.asarray(E[i]).astype(np.float32)
        cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-12)
        wc = min(wc, cos.min()); wr = max(wr, (np.abs(ref - got).max(1) / (np.abs(ref).max(1) + 1e-12)).max())
check(f"{len(idx)} recomputed rows match", wc >= 0.9999 and wr <= 0.01, f"cos {wc:.6f} rel {wr:.4f}")
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED"); sys.exit(0 if ok else 1)
