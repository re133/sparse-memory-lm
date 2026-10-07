"""Where does Q+T keep what it learned? The trained add-on with its value table as trained (T), zeroed (Z) or
re-drawn from the initial distribution (R); Q = all gates at zero (bit-for-bit Qwen alone).
REPORT.md, "Addendum to step 3: where does Q+T keep what it learned?".

  python scripts/qwen_table_ablation.py [--addons runs/qwen_cloud/QT-s0/addons.pt] [--skip_facts]

PPL (and knowledge-token PPL) on mem_probe, val_new, val_known, val_known_same as in train_qwen_memory.py; fact test
as in eval_fact_cloze.py (T, Z, R). Results go to report/qwen/table_ablation.json after every variant.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_fact_cloze import correct, wilson  # noqa: E402
from eval_qwen_general import load  # noqa: E402
from smlm.atomic import write_json  # noqa: E402
from train_qwen_memory import evaluate, knowledge_mask, knowledge_token_ids, load_tokens  # noqa: E402

SETS = ["mem_probe", "val_new", "val_known", "val_known_same"]


@torch.no_grad()
def facts(model, tok, items, max_new=16):
    model.eval()                                             # evaluate() leaves the model in train mode (BatchNorm!)
    per = []
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        cont = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        per.append({"id": it["id"], "split": it["split"], "correct": correct(cont, it["answer"]), "output": cont})
    summary = {}
    for split in ("train", "heldout"):
        sel = [p["correct"] for p in per if p["split"] == split]
        summary[split] = {"n": len(sel), "correct": sum(sel), "acc": sum(sel) / len(sel), "wilson95": wilson(sum(sel), len(sel))}
    return summary, per


def paired_diff(a, b, n_boot=20000, seed=7):
    """Accuracy difference a - b over the same items with a paired bootstrap 95 % interval (percentage points)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return {"diff_pp": 100 * float(a.mean() - b.mean()), "ci95_pp": [100 * float(np.percentile(d, 2.5)),
                                                                    100 * float(np.percentile(d, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("QWEN_DIR", os.path.join(ROOT, "models", "Qwen3.5-0.8B")))
    ap.add_argument("--data_dir", default=os.environ.get("QWEN_DATA", os.path.join(ROOT, "data", "qwen_wiki")))
    ap.add_argument("--addons", default=os.path.join(ROOT, "runs", "qwen_cloud", "QT-s0", "addons.pt"))
    ap.add_argument("--items", default=os.path.join(ROOT, "data", "qwen_fact_cloze.jsonl"))
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_windows", type=int, default=None)
    ap.add_argument("--skip_facts", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "report", "qwen", "table_ablation.json"))
    args = ap.parse_args()
    model, tok = load(args.model_dir, args.addons)
    table = model.addons[0].body.values.weight                  # shared by all add-on blocks
    assert all(a.body.values.weight is table for a in model.addons)
    gates = [a.gate.detach().clone() for a in model.addons]
    kid = knowledge_token_ids(tok)
    sets = {k: load_tokens(args.data_dir, k) for k in SETS if os.path.exists(os.path.join(args.data_dir, k + ".bin"))}
    masks = {k: knowledge_mask(v, kid) for k, v in sets.items()}
    items = [json.loads(line) for line in open(args.items)]
    res = {"addons": os.path.relpath(args.addons, ROOT), "gpu": torch.cuda.get_device_name(0),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gates": [float(g) for g in gates], "variants": {}}
    per_items = {}

    def measure(name, do_facts):
        t0 = time.time()
        out = {k: evaluate(model, v, args.seq_len, args.max_windows, mask=masks[k]) for k, v in sets.items()}
        model.eval()
        print(name, {k: round(v["ppl"], 3) for k, v in out.items()}, f"({time.time() - t0:.0f} s)", flush=True)
        if do_facts and not args.skip_facts:
            out["facts"], per_items[name] = facts(model, tok, items)
            print(name, "facts", {k: f"{v['correct']}/{v['n']}" for k, v in out["facts"].items()}, flush=True)
        out["seconds"] = round(time.time() - t0, 1)
        res["variants"][name] = out
        write_json(args.out, res, indent=1)

    with torch.no_grad():
        for a in model.addons:                                  # Q: gates 0 -> exactly Qwen alone
            a.gate.zero_()
        measure("Q", do_facts=False)
        for a, g in zip(model.addons, gates):
            a.gate.copy_(g)
        measure("T", do_facts=True)
        trained = table.detach().cpu()
        table.zero_()
        measure("Z", do_facts=True)
        g = torch.Generator(device="cuda").manual_seed(0)
        table.normal_(std=table.shape[1] ** -0.5, generator=g)  # ProductKeyMemory.reset_parameters
        measure("R", do_facts=True)
        table.copy_(trained.cuda())

    # shares of the gain over Q that go away without the table, on log PPL
    V = res["variants"]
    res["share_lost_without_table"] = {
        k: (math.log(V["Z"][k]["ppl"]) - math.log(V["T"][k]["ppl"])) / (math.log(V["Q"][k]["ppl"]) - math.log(V["T"][k]["ppl"]))
        for k in sets}
    if per_items:
        def acc(name, split):
            return [p["correct"] for p in per_items[name] if p["split"] == split]
        res["facts_T_minus_Z"] = {s: paired_diff(acc("T", s), acc("Z", s)) for s in ("train", "heldout")}
        res["facts_T_minus_R"] = {s: paired_diff(acc("T", s), acc("R", s)) for s in ("train", "heldout")}
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(args.out, res, indent=1)
    print(json.dumps({k: res[k] for k in res if k.startswith(("share", "facts_"))}, indent=1))


if __name__ == "__main__":
    main()
