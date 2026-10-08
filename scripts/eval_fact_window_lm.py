"""Second addendum to step 6: the fact test in the exact training window (REPORT.md).

  python scripts/eval_fact_window_lm.py B-16M        -> report/facts_lm/window_B-16M.json
  python scripts/eval_fact_window_lm.py --summary    -> report/facts_lm/window_summary.json

Items and scoring as scripts/eval_fact_logprob_lm.py (log-probability of the article's own answer tokens, teacher
forced, bf16 autocast, product-key models with the 4-bit table), with one prompt instead of two:
  window   seen items: the training window in which the answer's first token was a target, from the window start up to
           the answer, so exactly what the model had in context when it learned the fact (training cuts the train split
           into windows of 1,024 + 1 tokens on a fixed grid; checked for every seen item against
           smlm.data.TrainStream.batch(train_step)). Unseen items: the same grid on the validation split, so both
           prompts have the same length distribution. The answer must lie inside the same window. Items whose answer
           already occurs in the prompt are left out (copying, not memory); same rule for both splits.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_fact_cloze_lm import ITEMS, OUT, RUNS, load  # noqa: E402
from eval_fact_logprob_lm import BOOT, answer_tokens, ci, score_pairs  # noqa: E402
from smlm.atomic import write_json  # noqa: E402
from smlm.data import TrainStream, load_split  # noqa: E402

SEQ, BATCH_SEQS, DATA_SEED = 1024, 32, 1234           # as every model of steps 1-3 was trained


def build(enc, check=True):
    """-> cases [{id, split, train_step, window: (prompt ids, answer ids) or None}], number of checked seen items."""
    items = [json.loads(line) for line in open(ITEMS)]
    toks = {"seen": np.asarray(load_split("train", "wikipedia")),
            "unseen": np.asarray(load_split("validation", "wikipedia"))}
    stream = TrainStream(SEQ, BATCH_SEQS, DATA_SEED, split="train", dataset="wikipedia") if check else None
    cases, checked = [], 0
    for it in items:
        t, g = toks[it["split"]], it["answer_token"]
        start = (g - 1) // SEQ * SEQ                     # window [start, start + SEQ]: g is one of its targets
        prompt = [int(x) for x in t[start:g]]
        ans = answer_tokens(enc, t, g, it["answer"])
        ok = ans is not None and g + len(ans) - 1 <= start + SEQ and it["answer"] not in enc.decode(prompt)
        if check and it["split"] == "seen":
            window = t[start:start + SEQ + 1]
            assert any(np.array_equal(row, window) for row in stream.batch(it["train_step"])), it["id"]
            checked += 1
        cases.append({"id": it["id"], "split": it["split"], "train_step": it["train_step"],
                      "window": (prompt, ans) if ok else None})
    return cases, checked


def score(name):
    enc = tiktoken.get_encoding("gpt2")
    cases, checked = build(enc)
    model = load(name)
    t0 = time.time()
    sel = [c for c in cases if c["window"] is not None]
    sc = score_pairs(model, [c["window"] for c in sel])
    res = {"model": name, "run": RUNS[name], "seen_windows_checked": checked, "gpu": torch.cuda.get_device_name(),
           "window": {c["id"]: {"logp": round(lp, 4), "hit": hit, "prompt_tokens": len(c["window"][0])}
                      for c, (lp, hit) in zip(sel, sc)},
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, f"window_{name}.json"), res)
    for s in ("seen", "unseen"):
        v = [r["logp"] for i, r in res["window"].items() if i.startswith(s + "-")]
        print(name, s, len(v), round(float(np.mean(v)), 3), flush=True)


def summary():
    enc = tiktoken.get_encoding("gpt2")
    cases, _ = build(enc, check=False)
    data = {n: json.load(open(p)) for n in RUNS if os.path.exists(p := os.path.join(OUT, f"window_{n}.json"))}
    context = {}
    p = os.path.join(OUT, "logprob_summary.json")
    if os.path.exists(p):
        context = json.load(open(p))["context"]["models"]
    use = [c for c in cases if c["window"] is not None]
    ids = [c["id"] for c in use]
    split = np.array([c["split"] for c in use])
    step = np.array([c["train_step"] for c in use])
    plen = np.array([len(c["window"][0]) for c in use])
    seen, unseen = np.flatnonzero(split == "seen"), np.flatnonzero(split == "unseen")
    edges = np.linspace(0, step.max() + 1, 6)
    early = np.flatnonzero((split == "seen") & (step < edges[1]))
    late = np.flatnonzero((split == "seen") & (step >= edges[4]))
    rng = np.random.default_rng(0)
    draws = {k: rng.integers(0, len(v), (BOOT, len(v))) for k, v in
             (("seen", seen), ("unseen", unseen), ("early", early), ("late", late))}
    out = {"bootstrap": BOOT, "items": {"seen": len(seen), "unseen": len(unseen), "early": len(early),
                                        "late": len(late)},
           "prompt_tokens_mean": {"seen": round(float(plen[seen].mean()), 1),
                                  "unseen": round(float(plen[unseen].mean()), 1)},
           "models": {}}
    gaps, recs = {}, {}
    for name, r in data.items():
        lp = np.array([r["window"][i]["logp"] for i in ids])
        hit = np.array([r["window"][i]["hit"] for i in ids], dtype=float)
        gaps[name] = lp[seen][draws["seen"]].mean(1) - lp[unseen][draws["unseen"]].mean(1)
        recs[name] = lp[late][draws["late"]].mean(1) - lp[early][draws["early"]].mean(1)
        fifths = [round(float(lp[(split == "seen") & (step >= a) & (step < b)].mean()), 3)
                  for a, b in zip(edges[:-1], edges[1:])]
        m = {"logp_seen": round(float(lp[seen].mean()), 3), "logp_unseen": round(float(lp[unseen].mean()), 3),
             "gap_seen_minus_unseen": round(float(lp[seen].mean() - lp[unseen].mean()), 3), "gap_ci95": ci(gaps[name]),
             "recency_late_minus_early": round(float(lp[late].mean() - lp[early].mean()), 3),
             "recency_ci95": ci(recs[name]), "logp_seen_by_fifth": fifths,
             "hit_seen_pct": round(100 * float(hit[seen].mean()), 2),
             "hit_unseen_pct": round(100 * float(hit[unseen].mean()), 2)}
        if name in context:                              # the addendum's article-context prompt, for comparison
            m["context_prompt"] = {k: context[name][k] for k in ("gap_seen_minus_unseen", "gap_ci95",
                                                                 "recency_late_minus_early", "recency_ci95")}
        out["models"][name] = m
    if "B-16M" in data:
        for ref in ("D-100M", "D-200M"):
            if ref in data:
                b, d = out["models"]["B-16M"], out["models"][ref]
                out[f"B-16M_minus_{ref}"] = {
                    "gap": round(b["gap_seen_minus_unseen"] - d["gap_seen_minus_unseen"], 3),
                    "gap_ci95": ci(gaps["B-16M"] - gaps[ref]),
                    "recency": round(b["recency_late_minus_early"] - d["recency_late_minus_early"], 3),
                    "recency_ci95": ci(recs["B-16M"] - recs[ref])}
    write_json(os.path.join(OUT, "window_summary.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    for name, m in out["models"].items():
        print(name, m["gap_seen_minus_unseen"], m["gap_ci95"], m["recency_late_minus_early"], m["recency_ci95"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=list(RUNS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else score(a.model)
