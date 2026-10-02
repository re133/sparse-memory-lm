"""Post-training quantisation of the memory value table only (no training, checkpoint is read-only).

  python scripts/quantize_table_eval.py runs/s1b/B-1M-s0 --data wikipedia

The value table is replaced IN MEMORY by its quantised-and-dequantised version; every other parameter stays
in fp32. For each scheme the full validation set is evaluated with smlm.train.evaluate (same code path as
in training). The checkpoint file is hashed before and after to show it was not modified.

Schemes (one scale per table row = per 384-value entry, stored as fp16):
  fp32      original (reference)
  bf16      plain cast (reference: what bf16 inference would use)
  int8/4/3/2  "Q_0"-style (as llama.cpp Q4_0/Q8_0, but per row): codes c in [-2^(b-1), 2^(b-1)-1],
            d = (value with the largest |w| in the row) / -2^(b-1), w ~ c * d. Uses all 2^b codes,
            the largest-magnitude value is represented exactly.
  ternary   BitNet-b1.58-style absmean: s = mean |w| of the row, c = clamp(round(w / s), -1, 1), w ~ c * s.
            log2(3) = 1.58 bit of information; packed as 5 values per byte = 1.6 bit.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smlm.data import load_meta  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.train import evaluate  # noqa: E402

SCALE_BYTES = 2          # fp16 scale per row


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def quantize(w, scheme):
    """w: (rows, dim) float32 -> dequantised float32 of the same shape."""
    if scheme == "fp32":
        return w.clone()
    if scheme == "bf16":
        return w.to(torch.bfloat16).float()
    if scheme == "ternary":
        s = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
        s = s.half().float()                                  # the scale is stored in fp16
        return torch.clamp(torch.round(w / s), -1, 1) * s
    bits = int(scheme.replace("int", ""))
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    idx = w.abs().argmax(dim=1, keepdim=True)
    m = w.gather(1, idx)                                       # signed value of largest magnitude
    d = (m / lo).half().float()                                # fp16 scale; maps m to code lo exactly
    d = torch.where(d == 0, torch.ones_like(d), d)
    return torch.clamp(torch.round(w / d), lo, hi) * d


def bits_per_value(scheme):
    return {"fp32": 32, "bf16": 16, "int8": 8, "int4": 4, "int3": 3, "int2": 2, "ternary": 1.6}[scheme]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default="wikipedia")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None, help="json output (default: report/quant_<run>.json)")
    ap.add_argument("--schemes", default="fp32,bf16,int8,int4,int3,int2,ternary")
    args = ap.parse_args()

    ck_path = os.path.join(args.run_dir, "model.pt")
    st0 = os.stat(ck_path)
    h0 = sha256(ck_path)
    ck = torch.load(ck_path, map_location="cpu")
    cfg = ModelConfig(**ck["model_config"])
    model = Transformer(cfg)
    model.load_state_dict(ck["state_dict"])
    del ck
    model.to(args.device).eval()
    mems = model.memory_layers()
    table = mems[0].values.weight
    assert all(m.values.weight is table for m in mems), "expects one shared table (B-1M)"
    rows, dim = table.shape
    orig = table.detach().clone()
    meta = load_meta(args.data)
    n_words = meta["splits"]["validation"]["n_words"]

    results = []
    for scheme in args.schemes.split(","):
        t0 = time.time()
        q = torch.empty_like(orig)
        for a in range(0, rows, 1 << 16):                     # chunked to bound temporary memory
            q[a:a + (1 << 16)] = quantize(orig[a:a + (1 << 16)], scheme)
        rel_err = float((q - orig).norm() / orig.norm())
        with torch.no_grad():
            table.copy_(q)
        del q
        ev = evaluate(model, "validation", cfg.max_seq_len, n_words, batch=args.batch, dataset=args.data)
        bpv = bits_per_value(scheme)
        size_mb = rows * dim * bpv / 8 / 1e6 + (rows * SCALE_BYTES / 1e6 if scheme.startswith(("int", "tern")) else 0)
        results.append({"scheme": scheme, "bits_per_value": bpv, "table_mb": size_mb, "val_loss": ev["loss"],
                        "val_ppl": ev["ppl"], "rel_l2_error": rel_err, "eval_tokens": ev["n_tokens"],
                        "seconds": round(time.time() - t0, 1)})
        print(f"{scheme:8s} {bpv:>4} bit  {size_mb:8.1f} MB  val_ppl {ev['ppl']:.4f}  rel_err {rel_err:.4f}", flush=True)
    with torch.no_grad():
        table.copy_(orig)                                      # leave the in-memory model as loaded

    base = next(r for r in results if r["scheme"] == "fp32")
    for r in results:
        r["ppl_delta"] = r["val_ppl"] - base["val_ppl"]
        r["ppl_delta_pct"] = 100 * (r["val_ppl"] / base["val_ppl"] - 1)

    st1 = os.stat(ck_path)
    h1 = sha256(ck_path)
    out = {"run": args.run_dir, "data": args.data, "rows": rows, "dim": dim,
           "checkpoint": {"sha256_before": h0, "sha256_after": h1, "unchanged": h0 == h1 and st0.st_mtime == st1.st_mtime,
                          "mtime": st0.st_mtime},
           "scale": "one fp16 scale per row (entry of %d values)" % dim, "results": results}
    path = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report",
                                    f"quant_{os.path.basename(os.path.normpath(args.run_dir))}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("checkpoint unchanged:", out["checkpoint"]["unchanged"], "->", path)


if __name__ == "__main__":
    main()
