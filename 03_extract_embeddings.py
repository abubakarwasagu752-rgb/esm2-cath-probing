#!/usr/bin/env python3
"""Mean-pooled hidden states from every layer. Output (N, L+1, D) float16, rows in domains.tsv order.
Index L = last block output AFTER the final LayerNorm (a hidden state, not logits).
Pooling: real residues only (<cls>, <eos>, padding excluded), float32."""
import argparse
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, EsmConfig, EsmModel
from common import load_domains

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--data", default="data"); ap.add_argument("--out", required=True)
    ap.add_argument("--random-init", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    a = ap.parse_args()
    dev = "cuda"
    seqs = load_domains(a.data)["seq"].tolist(); n = len(seqs)
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.random_init:
        torch.manual_seed(a.seed)
        model = EsmModel(EsmConfig.from_pretrained(a.model), add_pooling_layer=False)
    else:
        model = EsmModel.from_pretrained(a.model, add_pooling_layer=False)
    model.eval().to(dev)
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    print(f"{a.model} random_init={a.random_init} layers={L} dim={D} N={n}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(n, L + 1, D))
    order = np.argsort([len(s) for s in seqs]); i = 0
    with torch.inference_mode():
        while i < n:
            j, longest = i, 0
            while j < n:
                longest = max(longest, len(seqs[order[j]]) + 2)
                if (j - i + 1) * longest > a.batch_tokens and j > i: break
                j += 1
            idx = order[i:j]
            enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states
            attn = enc["attention_mask"].clone(); lengths = attn.sum(1)
            attn[:, 0] = 0; attn[torch.arange(len(idx)), lengths - 1] = 0
            m = attn.unsqueeze(-1).float(); denom = m.sum(1).clamp(min=1)
            pooled = torch.stack([(h.float() * m).sum(1) / denom for h in hs], dim=1)
            out[idx] = pooled.cpu().numpy().astype(np.float16)
            i = j
    out.flush(); print("saved", a.out, out.shape)

if __name__ == "__main__":
    main()
