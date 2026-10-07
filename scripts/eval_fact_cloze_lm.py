"""Step 6 fact test: score the models trained from scratch on data/lm_fact_cloze.jsonl (make_fact_cloze_lm.py).

  python scripts/eval_fact_cloze_lm.py B-16M          -> report/facts_lm/B-16M.json
  python scripts/eval_fact_cloze_lm.py --summary      -> report/facts_lm/summary.json

Greedy continuation of the prompt with KV cache, same correctness rule as step 3 (scripts/eval_fact_cloze.py): the
continuation, without leading spaces, starts with the answer and the next character is not a letter or digit.
Generation stops as soon as the item is decided (wrong prefix, or two characters past the answer), at most 16 tokens.
Product-key models run with the 4-bit inference table (B-16M only fits that way on 16 GB; for B-1M the 4-bit table
changes val PPL by +0.13 %), dense models as trained, everything under bf16 autocast.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from smlm.atomic import write_json  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402

ITEMS = os.path.join(ROOT, "data", "lm_fact_cloze.jsonl")
OUT = os.path.join(ROOT, "report", "facts_lm")
TABLES = os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M"))
RUNS = {"A": "s1b/A-s0", "B-1M": "cloud/B-1M-s0", "B-4M": "cloud/B-4M-s0", "B-16M": "cloud/B-16M-s0",
        "D-50M": "cloud_dense/D-50M-s0", "D-100M": "cloud_dense/D-100M-s0", "D-200M": "cloud_dense/D-200M-s0",
        "D-400M": "cloud_dense/D-400M-s0"}
EOT, MAX_NEW, FIFTHS = 50256, 16, 5
BOOT = 10000


def correct(cont, ans):
    """Same rule as scripts/eval_fact_cloze.py (copied: that script needs transformers)."""
    c = cont.lstrip()
    if not c.startswith(ans):
        return False
    rest = c[len(ans):]
    if not rest:
        return True
    if rest[0].isalnum():
        return False
    return not (ans[-1:].isdigit() and (rest[0] == "_" or (rest[0] in ".," and rest[1:2].isdigit())))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - w), 2), round(100 * (c + w), 2)]


def load(name):
    if name == "B-16M":
        from demo_generate import load as load_tables
        return load_tables(TABLES, "vram", 0)                          # 4-bit table in VRAM, decode graphs
    ck = torch.load(os.path.join(ROOT, "runs", RUNS[name], "model.pt"), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    if cfg.mem_layers:
        cfg.mem_impl = "triton"
    model = Transformer(cfg).cuda()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if cfg.mem_layers:
        model.set_memory_inference_table("q4")
        model.set_memory_decode_graphs(True)
    return model


@torch.no_grad()
def continuation(model, enc, prompt, ans):
    ids = enc.encode(prompt)
    caches = [dict() for _ in range(model.cfg.n_layers)]
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(torch.tensor([ids], device="cuda"), kv_caches=caches, pos0=0)
        for i in range(MAX_NEW):
            tok = int(logits[0, -1].float().argmax())
            if tok == EOT:
                break
            out.append(tok)
            c = enc.decode(out).lstrip()
            if not (ans.startswith(c) or c.startswith(ans)) or len(c) >= len(ans) + 2:
                break
            logits = model(torch.tensor([[tok]], device="cuda"), kv_caches=caches, pos0=len(ids) + i)
    return enc.decode(out)


def score(name):
    items = [json.loads(line) for line in open(ITEMS)]
    enc = tiktoken.get_encoding("gpt2")
    model = load(name)
    t0 = time.time()
    res = []
    for it in items:
        cont = continuation(model, enc, it["prompt"], it["answer"])
        res.append({"id": it["id"], "ok": correct(cont, it["answer"]), "out": cont[:60]})
    out = {"model": name, "run": RUNS[name], "items": len(items), "seconds": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name()}
    for split in ("seen", "unseen"):
        ok = [r["ok"] for r, it in zip(res, items) if it["split"] == split]
        out[split] = {"n": len(ok), "acc_pct": round(100 * sum(ok) / len(ok), 2), "ci95": wilson(sum(ok), len(ok))}
    out["results"] = res
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, name + ".json"), out)
    print(name, {k: out[k] for k in ("seen", "unseen", "seconds")}, flush=True)


def gap_draws(ok, seen, idx_s, idx_u):
    """Bootstrap draws of acc(seen) - acc(unseen) in pp, items resampled within each split (same draws for all
    models, so differences between models are paired)."""
    s, u = ok[seen], ok[~seen]
    return 100 * (s[idx_s].mean(1) - u[idx_u].mean(1))


def summary():
    items = [json.loads(line) for line in open(ITEMS)]
    seen = np.array([it["split"] == "seen" for it in items])
    step = np.array([it["train_step"] for it in items])
    kinds = np.array([it["type"] for it in items])
    rng = np.random.default_rng(0)
    idx_s = rng.integers(0, seen.sum(), (BOOT, seen.sum()))
    idx_u = rng.integers(0, (~seen).sum(), (BOOT, (~seen).sum()))
    ok, out = {}, {"items": len(items), "bootstrap": BOOT, "models": {}}
    for name in RUNS:
        p = os.path.join(OUT, name + ".json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        assert [x["id"] for x in r["results"]] == [it["id"] for it in items]
        ok[name] = np.array([x["ok"] for x in r["results"]], dtype=np.float64)
        d = gap_draws(ok[name], seen, idx_s, idx_u)
        m = {"seen_pct": r["seen"]["acc_pct"], "unseen_pct": r["unseen"]["acc_pct"],
             "gap_pp": round(float(100 * (ok[name][seen].mean() - ok[name][~seen].mean())), 2),
             "gap_ci95": [round(float(np.percentile(d, q)), 2) for q in (2.5, 97.5)],
             "by_type": {}, "seen_by_training_fifth": []}
        for k in ("name", "date/year", "number"):
            m["by_type"][k] = {s: round(100 * float(ok[name][(kinds == k) & (seen == (s == "seen"))].mean()), 2)
                               for s in ("seen", "unseen")}
        edges = np.linspace(0, step.max() + 1, FIFTHS + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = seen & (step >= a) & (step < b)
            m["seen_by_training_fifth"].append(round(100 * float(ok[name][sel].mean()), 2))
        out["models"][name] = m
    if "B-16M" in ok:
        for ref in ("D-100M", "D-200M"):
            if ref in ok:
                d = gap_draws(ok["B-16M"], seen, idx_s, idx_u) - gap_draws(ok[ref], seen, idx_s, idx_u)
                g = out["models"]["B-16M"]["gap_pp"] - out["models"][ref]["gap_pp"]
                out[f"gap_B-16M_minus_{ref}"] = {"pp": round(g, 2),
                                                 "ci95": [round(float(np.percentile(d, q)), 2) for q in (2.5, 97.5)]}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    for name, m in out["models"].items():
        print(name, m["seen_pct"], m["unseen_pct"], m["gap_pp"], m["gap_ci95"], m["seen_by_training_fifth"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=list(RUNS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else score(a.model)
