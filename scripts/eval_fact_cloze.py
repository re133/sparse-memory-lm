"""Step 3 fact test: score Qwen (optionally with add-ons) on data/qwen_fact_cloze.jsonl (scripts/make_fact_cloze.py).

  .venv-qwen/bin/python scripts/eval_fact_cloze.py --out report/qwen/facts_Q.json
  .venv-qwen/bin/python scripts/eval_fact_cloze.py --addons runs/qwen/QT-s0/addons.pt --out report/qwen/facts_QT.json

Greedy continuation of the prompt (max 16 new tokens, plain text, no chat template); correct = the continuation,
without leading spaces, starts with the answer and the next character is not a letter or digit (exact match,
first attempt). Accuracy per split (train / heldout) and type, 95 % Wilson interval; per-item results are saved so
that two models can be compared item by item (paired) with scripts/compare_fact_cloze.py.
"""
import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_qwen_general import load  # noqa: E402


def correct(cont, ans):
    c = cont.lstrip()
    return c.startswith(ans) and (len(c) == len(ans) or not c[len(ans)].isalnum())


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - w, c + w]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/home/leon/smlm-models/Qwen3.5-0.8B")
    ap.add_argument("--addons", default=None)
    ap.add_argument("--items", default=os.path.join(ROOT, "data", "qwen_fact_cloze.jsonl"))
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = load(args.model_dir, args.addons)
    items = [json.loads(line) for line in open(args.items)]
    per = []
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        cont = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        per.append({"id": it["id"], "split": it["split"], "type": it["type"], "answer": it["answer"],
                    "output": cont, "correct": correct(cont, it["answer"])})
    summary = {}
    for split in ("train", "heldout"):
        for typ in (None, "name", "date/year", "number"):
            sel = [p for p in per if p["split"] == split and (typ is None or p["type"] == typ)]
            k = sum(p["correct"] for p in sel)
            summary[f"{split}/{typ or 'all'}"] = {"n": len(sel), "correct": k, "acc": k / max(1, len(sel)),
                                                 "wilson95": wilson(k, len(sel))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"addons": args.addons, "summary": summary, "items": per}, open(args.out, "w"), indent=1,
              ensure_ascii=False)
    for k, v in summary.items():
        print(f"{k:20s} {v['correct']:4d}/{v['n']:4d} = {100 * v['acc']:5.1f} %  "
              f"[{100 * v['wilson95'][0]:.1f}, {100 * v['wilson95'][1]:.1f}]")


if __name__ == "__main__":
    main()
