"""Step 4 evaluation (criteria fixed in REPORT.md, "Step 4", before the run): E-1M (Engram n-gram tables) against
B-1M-sparse (product keys), both trained at home with the same arguments.

  python scripts/engram_eval.py     -> report/engram/summary.json, markdown on stdout

  * main verdict: mean val PPL of E-1M s0/s1 against B-1M-sparse s0/s1 (runs/hampter): "E better" E <= 0.99 B,
    "on par" within +-1 %, "B better" E >= 1.01 B
  * against A (runs/s1b/A-s0, runs/hampter/A-s1): improvement in %, equivalent dense size with the interpolation of
    scripts/dense_equiv.py
  * optimizer control: E-1M-dense-s0 <= 0.99 x E-1M-s0 -> lazy Adam puts the n-gram table at a disadvantage, the main
    verdict is then also given with the dense value (one seed, hint only)
  * reported without a verdict: WikiText-103 val PPL, training speed and peak memory, table values read per token,
    share of table rows read on the validation set (computed from the hash, which only depends on the tokens)
"""
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dense_equiv import DENSE, SEED_SPREAD, fit, fit_inverse, interp, load  # noqa: E402
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402
from smlm.engram import NgramHash  # noqa: E402

E = {"E-1M-s0": "runs/engram/E-1M-s0", "E-1M-s1": "runs/engram/E-1M-s1"}
E_DENSE = ("E-1M-dense-s0", "runs/engram/E-1M-dense-s0")
B = {"B-1M-sparse-s0": "runs/hampter/B-1M-sparse-s0", "B-1M-sparse-s1": "runs/hampter/B-1M-sparse-s1"}
A = {"A-s0": "runs/s1b/A-s0", "A-s1": "runs/hampter/A-s1"}
OUT = os.path.join(ROOT, "report", "engram")
SEQ = 1024


def info(rel):
    return json.load(open(os.path.join(ROOT, rel, "run-info.json")))


def verdict(e, b):
    r = e / b
    return r, ("E better" if r <= 0.99 else "B better" if r >= 1.01 else "on par")


def n_eq(ppl, pts, f):
    """Equivalent dense size as in dense_equiv.py: interpolation, range with +-SEED_SPREAD, widened to the fit."""
    neq, status = interp(ppl, pts)
    cands = []
    for sb in (-1, 1):
        for sn in (-1, 1):
            v, _ = interp(ppl * (1 + sb * SEED_SPREAD), [(n, p * (1 + sn * SEED_SPREAD)) for n, p in pts])
            cands.append(v)
    ok = [c for c in cands if c]
    lo, hi = (min(ok), max(ok)) if ok else (None, None)
    nfit = fit_inverse(ppl, f)
    if neq and math.isfinite(nfit):
        lo, hi = min(lo, nfit), max(hi, nfit)
    return {"n_eq": neq, "status": status, "range": [lo, hi], "n_eq_fit": nfit}


@torch.no_grad()
def val_row_share(cfg):
    """Share of the table rows that are read on the validation set, windows as in smlm.train.evaluate (pad history
    at every window start). The rows only depend on the tokens, so the share is the same for every Engram module."""
    h = NgramHash(cfg["vocab_size"], cfg["eng_orders"], cfg["eng_heads"], cfg["eng_rows"], cfg["eng_hash_seed"])
    tok = torch.from_numpy(np.asarray(load_split("validation", "wikipedia"), dtype=np.int64))
    n = tok.numel() - 1
    full = n // SEQ
    chunks = list(tok[:full * SEQ].view(full, SEQ).split(64)) + ([tok[full * SEQ:n][None]] if n > full * SEQ else [])
    seen = torch.zeros(h.n_lookups * h.rows, dtype=torch.bool)
    for x in chunks:
        c = torch.cat([torch.full((x.shape[0], h.history), h.pad, dtype=torch.long), h.canon[x]], 1)
        seen[h(c).reshape(-1)] = True
    per_head = seen.view(h.n_lookups, h.rows).float().mean(1)
    return {"share": float(seen.float().mean()), "per_order": {f"order_{o}": float(per_head[i * h.heads:(i + 1)
            * h.heads].mean()) for i, o in enumerate(h.orders)}, "val_tokens": n}


def main():
    runs = {**E, E_DENSE[0]: E_DENSE[1], **B, **A}
    res = {}
    for name, rel in runs.items():
        i = info(rel)
        assert i["status"] == "done", name
        r = i["results"]
        res[name] = {"run": rel, "val_ppl": r["val_ppl"], "wikitext103_ppl": r["val2_ppl"],
                     "train_tok_s_median": r["train_tok_s_median"], "peak_train_vram_gib": r["peak_train_vram_gib"],
                     "table_params": i["params"].get("memory_values", 0), "dense_body": i["params"]["dense_body"],
                     "macs_per_token": i["macs_per_token"]["total"]}
    mean = lambda d: float(np.mean([res[k]["val_ppl"] for k in d]))  # noqa: E731
    e, b, a = mean(E), mean(B), mean(A)
    r, v = verdict(e, b)
    out = {"runs": res, "mean_val_ppl": {"E-1M": e, "B-1M-sparse": b, "A": a},
           "main": {"ratio_E_over_B": r, "verdict": v,
                    "per_seed_ratio": {f"s{s}": res[f"E-1M-s{s}"]["val_ppl"] / res[f"B-1M-sparse-s{s}"]["val_ppl"]
                                       for s in (0, 1)}},
           "against_A": {"E_improvement_pct": 100 * (a - e) / a, "B_improvement_pct": 100 * (a - b) / a}}
    dense = [d for d in (load(rel) for _, rel in DENSE) if d]
    pts = sorted((d["n"], d["ppl"]) for d in dense)
    f = fit(pts)
    out["against_A"]["E_1M_mean"] = n_eq(e, pts, f)
    out["against_A"]["B_1M_sparse_mean"] = n_eq(b, pts, f)
    out["against_A"]["E_1M_dense_s0"] = n_eq(res[E_DENSE[0]]["val_ppl"], pts, f)
    out["against_A"]["share_of_B_gain_pct"] = {"E-1M": 100 * (a - e) / (a - b),
                                               "E-1M-dense-s0": 100 * (a - res[E_DENSE[0]]["val_ppl"]) / (a - b)}
    out["against_A"]["note"] = "dense points as in report/dense_equiv.json (A-s0, D-50M ... D-400M, seed 0)"
    d, s0 = res[E_DENSE[0]]["val_ppl"], res["E-1M-s0"]["val_ppl"]
    ctl = {"ratio_dense_over_s0": d / s0, "lazy_adam_disadvantage": d / s0 <= 0.99}
    if ctl["lazy_adam_disadvantage"]:
        rd, vd = verdict(d, b)
        ctl["verdict_with_dense_value"] = {"ratio_over_B_mean": rd, "verdict": vd, "note": "one seed, hint only"}
    out["optimizer_control"] = ctl
    cfg = info(E["E-1M-s0"])["model_config"]
    out["table_values_read_per_token"] = {
        "E-1M": len(cfg["eng_layers"]) * len(cfg["eng_orders"]) * cfg["eng_heads"] * cfg["eng_head_dim"],
        "B-1M": 3 * 4 * 32 * 384}
    out["val_row_share"] = {"E-1M": val_row_share(cfg)}
    b_usage = [float(np.genfromtxt(os.path.join(ROOT, rel, "metrics.csv"), delimiter=",", names=True)
                     ["mem_val_usage"][-1]) for rel in B.values()]
    out["val_row_share"]["B-1M-sparse"] = {"share": float(np.mean(b_usage)), "source": "metrics.csv mem_val_usage"}
    # course of val PPL at equal tokens (not a criterion)
    curves = {k: np.genfromtxt(os.path.join(ROOT, rel, "metrics.csv"), delimiter=",", names=True)
              for k, rel in runs.items()}
    out["course"] = {}
    for t in (50e6, 100e6, 200e6, 300e6, 400e6, 500e6):
        row = {}
        for k, m in curves.items():
            i = int(np.argmin(abs(m["tokens"] - t)))
            assert abs(m["tokens"][i] - t) < 6e6, (k, t)
            row[k] = float(m["val_ppl"][i])
        out["course"][f"{t / 1e6:.0f}M"] = row
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "summary.json"), out)

    print("| Run | Val PPL | WikiText-103 | tok/s | peak GiB |\n|---|---:|---:|---:|---:|")
    for name, x in res.items():
        print(f"| {name} | {x['val_ppl']:.3f} | {x['wikitext103_ppl']:.2f} | {x['train_tok_s_median']:,.0f} | "
              f"{x['peak_train_vram_gib']:.2f} |")
    print(f"\nmean E {e:.3f}  B {b:.3f}  A {a:.3f}  E/B {r:.4f} -> {v}")
    print("against A:", {k: (round(x, 2) if isinstance(x, float) else x) for k, x in out["against_A"].items()})
    print("control:", ctl)
    print("values/token:", out["table_values_read_per_token"], "val rows:", out["val_row_share"])
    print("\n| Tokens | " + " | ".join(runs) + " |\n|---|" + "---:|" * len(runs))
    for t, row in out["course"].items():
        print(f"| {t} | " + " | ".join(f"{v:.2f}" for v in row.values()) + " |")


if __name__ == "__main__":
    main()
