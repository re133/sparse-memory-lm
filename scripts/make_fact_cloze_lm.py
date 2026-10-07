"""Step 6 fact test on the models trained from scratch: cloze items from Wikipedia articles they saw in training
(once) and from articles they never saw. Same item rules as step 3 (scripts/make_fact_cloze.py).

  python scripts/make_fact_cloze_lm.py --out data/lm_fact_cloze.jsonl

  * seen: items from training articles whose answer was a prediction target during training. Every model of the
    500 M-token Wikipedia runs (A, B-*, D-*) used the same windows in the same order (data seed 1234, 32 x 1024
    tokens per step, 15,258 steps), so "seen" and the training step are the same for all of them.
  * unseen: items from the validation articles (random articles of the same dump, never trained on).
One item per article, articles in a fixed hash order. Target mix 40 % names, 30 % dates/years, 30 % numbers; the
1,917 validation articles run out of numbers first, so the seen split takes exactly the per-type counts the unseen
split reached (same mix on both sides).
"""
import argparse
import json
import os
import sys

import numpy as np
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_fact_cloze import candidates, h  # noqa: E402
from smlm.data import load_split  # noqa: E402

EOT = 50256
SEQ, BATCH, STEPS, DATA_SEED = 1024, 32, 15258, 1234


def seen_step(n_tokens):
    """Training step in which each window was used (-1: never), as smlm.data.TrainStream for epoch 0."""
    n_windows = (n_tokens - 1) // SEQ
    perm = np.random.default_rng([DATA_SEED, 0]).permutation(n_windows)
    step = np.full(n_windows, -1, dtype=np.int64)
    step[perm[:STEPS * BATCH]] = np.arange(STEPS * BATCH) // BATCH
    return step


def answer_token(tok, ids, text, title, item):
    """Token position of the answer's first token in the article (the answer follows the prompt's sentence prefix)."""
    prefix = item["prompt"][len(title) + 2:]
    a = text.find(prefix)
    while a >= 0 and not text[a + len(prefix):].lstrip(" ").startswith(item["answer"]):
        a = text.find(prefix, a + 1)
    if a < 0:
        return None
    char = a + len(prefix)
    ends = np.cumsum([len(tok.decode_single_token_bytes(t)) for t in ids])
    byte = len(text[:char].encode())
    return int(np.searchsorted(ends, byte, side="right"))


def build(split, want, tok, step_of_window):
    t = load_split("train" if split == "seen" else "validation", "wikipedia")
    cuts = np.flatnonzero(np.asarray(t) == EOT)
    starts = np.r_[0, cuts[:-1] + 1]
    order = sorted(range(len(cuts)), key=lambda k: h(split, k))
    items, got = [], {k: 0 for k in want}
    for k in order:
        if all(got[x] >= want[x] for x in want):
            break
        ids = np.asarray(t[starts[k]:cuts[k]]).tolist()
        text = tok.decode(ids)
        title = text.split("\n\n", 1)[0]
        cands = sorted((c for c in candidates(title, text) if got[c["type"]] < want[c["type"]]),
                       key=lambda c: h(k, c["answer"], c["prompt"]))
        for c in cands:
            pos = answer_token(tok, ids, text, title, c)
            if pos is None:
                continue
            g = int(starts[k]) + pos
            step = -1
            if split == "seen":
                w = (g - 1) // SEQ                   # window in which the answer's first token is a target
                step = int(step_of_window[w]) if w < len(step_of_window) else -1
                if step < 0 or (g - 1) % SEQ > SEQ - 8:      # unused window, or the answer runs into the next
                    continue
            got[c["type"]] += 1
            items.append({"id": f"{split}-{k}", "split": split, "article": int(k), "title": title,
                          "answer_token": g, "train_step": step, **c})
            break
    return items, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "lm_fact_cloze.jsonl"))
    args = ap.parse_args()
    tok = tiktoken.get_encoding("gpt2")
    step_of_window = seen_step(len(load_split("train", "wikipedia")))
    want = {"name": int(0.4 * args.n), "date/year": int(0.3 * args.n)}
    want["number"] = args.n - sum(want.values())
    unseen, got = build("unseen", want, tok, step_of_window)
    seen, got_seen = build("seen", got, tok, step_of_window)
    assert got_seen == got, (got_seen, got)
    print("unseen", len(unseen), got, "seen", len(seen), got_seen, flush=True)
    all_items = seen + unseen
    with open(args.out, "w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
