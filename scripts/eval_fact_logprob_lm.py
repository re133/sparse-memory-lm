"""Addendum to step 6: the fact test made more sensitive. Instead of "greedy hit or not", the log-probability of the
whole answer, and besides the short cloze prompt the original article text before the fact as the prompt.

  python scripts/eval_fact_logprob_lm.py B-16M        -> report/facts_lm/logprob_B-16M.json
  python scripts/eval_fact_logprob_lm.py --summary    -> report/facts_lm/logprob_summary.json

Items: data/lm_fact_cloze.jsonl (scripts/make_fact_cloze_lm.py), the same 2 x 1,382 as in step 6.
  cloze    prompt = title + blank line + sentence up to the fact (step 6); answer = GPT-2 tokens of " " + answer
  context  prompt = the article's own tokens from its start up to the answer, at most CONTEXT tokens (for seen items
           this is the text the model read right before the fact in training, unless the training window started
           later); answer = the next tokens of the article itself until they spell the answer. Items whose answer
           already occurs in that context are left out (copying, not memory); same rule for both splits.
Score per item: sum of log p(answer tokens), teacher forced, bf16 autocast, product-key models with the 4-bit table.
Also: "hit" = every answer token is the argmax (greedy along the article's own tokenization).
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
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402

EOT, CONTEXT, BATCH, BOOT = 50256, 1000, 8, 10000


def answer_tokens(enc, tokens, g, ans):
    """The article's own tokens from position g on until their text covers the answer."""
    out = []
    while len(enc.decode(out).lstrip()) < len(ans) and len(out) < 16:
        out.append(int(tokens[g + len(out)]))
    return out if enc.decode(out).lstrip().startswith(ans) else None


def build(enc):
    items = [json.loads(line) for line in open(ITEMS)]
    toks = {"seen": np.asarray(load_split("train", "wikipedia")), "unseen": np.asarray(load_split("validation",
                                                                                                     "wikipedia"))}
    starts = {}
    for s, t in toks.items():
        cuts = np.flatnonzero(t == EOT)
        starts[s] = np.r_[0, cuts[:-1] + 1]
    cases = []
    for it in items:
        t, g = toks[it["split"]], it["answer_token"]          # unseen: answer_token indexes validation.bin
        a0 = int(starts[it["split"]][it["article"]])
        ctx = [int(x) for x in t[max(a0, g - CONTEXT):g]]
        ans_ctx = answer_tokens(enc, t, g, it["answer"])
        copy = it["answer"] in enc.decode(ctx)
        cases.append({"id": it["id"], "split": it["split"], "type": it["type"], "train_step": it["train_step"],
                      "cloze": (enc.encode(it["prompt"]), enc.encode(" " + it["answer"])),
                      "context": None if (ans_ctx is None or copy) else (ctx, ans_ctx)})
    return cases


@torch.no_grad()
def score_pairs(model, pairs, drop_last=False):
    """pairs: list of (prompt ids, answer ids) -> (sum log p, all argmax) per pair, right-padded batches.
    drop_last: don't feed the answer's last token (no scored position reads it; the model is causal), so that a whole
    training window of 1,024 inputs + 1 target fits the model's 1,024 positions."""
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    res = [None] * len(pairs)
    for a in range(0, len(order), BATCH):
        idx = order[a:a + BATCH]
        seqs = [(pairs[i][0] + pairs[i][1])[:-1] if drop_last else pairs[i][0] + pairs[i][1] for i in idx]
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda())
        for r, i in enumerate(idx):
            p, ans = pairs[i]
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(ans), device="cuda")
            lp = torch.log_softmax(logits[r, pos].float(), -1)
            tgt = torch.tensor(ans, device="cuda")
            res[i] = (float(lp.gather(1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
    return res


def score(name):
    enc = tiktoken.get_encoding("gpt2")
    cases = build(enc)
    model = load(name)
    t0 = time.time()
    res = {"model": name, "run": RUNS[name], "context_tokens": CONTEXT, "gpu": torch.cuda.get_device_name()}
    for kind in ("cloze", "context"):
        sel = [i for i, c in enumerate(cases) if c[kind] is not None]
        sc = score_pairs(model, [cases[i][kind] for i in sel])
        res[kind] = {cases[i]["id"]: {"logp": round(lp, 4), "hit": hit} for i, (lp, hit) in zip(sel, sc)}
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, f"logprob_{name}.json"), res)
    for kind in ("cloze", "context"):
        for s in ("seen", "unseen"):
            v = [r["logp"] for i, r in res[kind].items() if i.startswith(s + "-")]
            print(name, kind, s, len(v), round(float(np.mean(v)), 3), flush=True)


def ci(d):
    return [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]


def summary():
    enc = tiktoken.get_encoding("gpt2")
    cases = build(enc)
    out = {"bootstrap": BOOT, "models": {}}
    data = {}
    for name in RUNS:
        p = os.path.join(OUT, f"logprob_{name}.json")
        if os.path.exists(p):
            data[name] = json.load(open(p))
    rng = np.random.default_rng(0)
    for kind in ("cloze", "context"):
        ids = [c["id"] for c in cases if c[kind] is not None]
        split = np.array([c["split"] for c in cases if c[kind] is not None])
        step = np.array([c["train_step"] for c in cases if c[kind] is not None])
        seen, unseen = np.flatnonzero(split == "seen"), np.flatnonzero(split == "unseen")
        edges = np.linspace(0, step.max() + 1, 6)
        early = np.flatnonzero((split == "seen") & (step < edges[1]))
        late = np.flatnonzero((split == "seen") & (step >= edges[4]))
        draws = {k: rng.integers(0, len(v), (BOOT, len(v))) for k, v in
                 (("seen", seen), ("unseen", unseen), ("early", early), ("late", late))}
        out[kind] = {"items": {"seen": len(seen), "unseen": len(unseen), "early": len(early), "late": len(late)},
                     "models": {}}
        gaps, recs = {}, {}
        for name, r in data.items():
            lp = np.array([r[kind][i]["logp"] for i in ids])
            hit = np.array([r[kind][i]["hit"] for i in ids], dtype=float)
            gaps[name] = lp[seen][draws["seen"]].mean(1) - lp[unseen][draws["unseen"]].mean(1)
            recs[name] = lp[late][draws["late"]].mean(1) - lp[early][draws["early"]].mean(1)
            fifths = [round(float(lp[(split == "seen") & (step >= a) & (step < b)].mean()), 3)
                      for a, b in zip(edges[:-1], edges[1:])]
            out[kind]["models"][name] = {
                "logp_seen": round(float(lp[seen].mean()), 3), "logp_unseen": round(float(lp[unseen].mean()), 3),
                "gap_seen_minus_unseen": round(float(lp[seen].mean() - lp[unseen].mean()), 3), "gap_ci95": ci(gaps[name]),
                "recency_late_minus_early": round(float(lp[late].mean() - lp[early].mean()), 3),
                "recency_ci95": ci(recs[name]), "logp_seen_by_fifth": fifths,
                "hit_seen_pct": round(100 * float(hit[seen].mean()), 2),
                "hit_unseen_pct": round(100 * float(hit[unseen].mean()), 2)}
        if "B-16M" in data:
            for ref in ("D-100M", "D-200M"):
                if ref in data:
                    out[kind][f"B-16M_minus_{ref}"] = {
                        "gap": round(float(out[kind]["models"]["B-16M"]["gap_seen_minus_unseen"]
                                           - out[kind]["models"][ref]["gap_seen_minus_unseen"]), 3),
                        "gap_ci95": ci(gaps["B-16M"] - gaps[ref]),
                        "recency": round(float(out[kind]["models"]["B-16M"]["recency_late_minus_early"]
                                               - out[kind]["models"][ref]["recency_late_minus_early"]), 3),
                        "recency_ci95": ci(recs["B-16M"] - recs[ref])}
    write_json(os.path.join(OUT, "logprob_summary.json"), out)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in out.items()
                      if isinstance(v, dict)}, indent=1))
    for kind in ("cloze", "context"):
        for name, m in out[kind]["models"].items():
            print(kind, name, m["gap_seen_minus_unseen"], m["gap_ci95"], m["recency_late_minus_early"],
                  m["recency_ci95"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", choices=list(RUNS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else score(a.model)
