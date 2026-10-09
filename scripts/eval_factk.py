"""FACTK evaluation (criteria fixed in REPORT.md, "Step 10", before the runs): how well each model knows the made-up
people of scripts/make_factk.py, by how often it saw them in training (k = 0, 1, 2, 4, ..., 64).

  python scripts/eval_factk.py B-1M [--run DIR] [--device cpu]   -> report/factk/eval_B-1M.json
  python scripts/eval_factk.py --summary                         -> report/factk/summary.json

Per person and attribute (year, city, profession; every value one GPT-2 token) one prompt in the training format,
  <|endoftext|>Name\\n\\nName was born in | grew up in | worked as a
and one forward pass: the model's probability of the true value, normalised over all values of that attribute
(118 years, 107 cities, 48 professions), and whether it is the most likely of them. Score of a person = mean over its
three attributes. Product-key models with the full fp32 table (no 4-bit table), bf16 autocast on the GPU as in
validation.

Summary: lift(k) = mean score at level k minus mean score at k = 0 (people the model never saw: its prior), 95%
bootstrap over people (paired across models, the same resampled people for all). Pooled lift = mean of lift(k) over
k = 1 ... 64, every level weighted equally.
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
from smlm.atomic import write_json  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402

OUT = os.path.join(ROOT, "report", "factk")
RUNS = {"B-1M": "runs/factk/B-1M-factk-s0", "D-50M": "runs/factk/D-50M-factk-s0",
        "D-100M": "runs/factk/D-100M-factk-s0"}
ATTRS = ("year", "city", "job")
EOT, BATCH, BOOT = 50256, 64, 10000


def load(run, device):
    ck = torch.load(os.path.join(ROOT, run, "model.pt"), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    if cfg.mem_layers:
        cfg.mem_impl = "triton" if device == "cuda" else "torch"
    model = Transformer(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval()


@torch.no_grad()
def score(name, run, device):
    enc = tiktoken.get_encoding("gpt2")
    P = json.load(open(os.path.join(OUT, "persons.json")))
    cand = {a: torch.tensor([enc.encode(" " + v)[0] for v in P["values"][a]], device=device) for a in ATTRS}
    items = []
    for pr in P["people"]:
        for a in ATTRS:
            ids = [EOT] + enc.encode(f"{pr['name']}\n\n{pr['name']}{P['cues'][a]}")
            items.append((pr["id"], a, ids, P["values"][a].index(pr[a])))
    model = load(run, device)
    t0 = time.time()
    res = {}
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        L = max(len(items[i][2]) for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(items[i][2])] = torch.tensor(items[i][2])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device))
        for r, i in enumerate(idx):
            pid, a, ids, true = items[i]
            lp = torch.log_softmax(logits[r, len(ids) - 1].float()[cand[a]], -1)
            res.setdefault(str(pid), {})[a] = {"p": round(float(lp[true].exp()), 6),
                                              "hit": bool(int(lp.argmax()) == true)}
    out = {"model": name, "run": run, "device": device, "seconds": round(time.time() - t0, 1), "people": res}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, f"eval_{name}.json"), out)
    lv = {p["id"]: p["k"] for p in P["people"]}
    for k in P["levels"]:
        s = [np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in lv if lv[i] == k]
        print(name, f"k={k:2d}", f"score {np.mean(s):.4f}", flush=True)


def ci(d):
    return [round(float(np.percentile(d, q)), 5) for q in (2.5, 97.5)]


def summary():
    P = json.load(open(os.path.join(OUT, "persons.json")))
    levels = P["levels"]
    data = {n: json.load(open(p)) for n in RUNS if os.path.exists(p := os.path.join(OUT, f"eval_{n}.json"))}
    ids = {k: [p["id"] for p in P["people"] if p["k"] == k] for k in levels}
    rng = np.random.default_rng(0)
    draws = {k: rng.integers(0, len(ids[k]), (BOOT, len(ids[k]))) for k in levels}
    score, hit, boot = {}, {}, {}
    for n, d in data.items():
        score[n] = {k: np.array([np.mean([d["people"][str(i)][a]["p"] for a in ATTRS]) for i in ids[k]])
                    for k in levels}
        hit[n] = {k: float(np.mean([d["people"][str(i)][a]["hit"] for i in ids[k] for a in ATTRS])) for k in levels}
        boot[n] = {k: score[n][k][draws[k]].mean(1) for k in levels}
    out = {"bootstrap": BOOT, "levels": levels, "models": {}}
    pooled = {}
    for n in data:
        lift = {k: boot[n][k] - boot[n][0] for k in levels if k}
        pooled[n] = np.mean([lift[k] for k in lift], axis=0)
        first = next((k for k in levels if k and ci(lift[k])[0] > 0), None)
        out["models"][n] = {
            "score": {k: round(float(score[n][k].mean()), 5) for k in levels},
            "accuracy": {k: round(hit[n][k], 4) for k in levels},
            "lift": {k: round(float(score[n][k].mean() - score[n][0].mean()), 5) for k in levels if k},
            "lift_ci95": {k: ci(lift[k]) for k in lift},
            "pooled_lift": round(float(pooled[n].mean()), 5), "pooled_lift_ci95": ci(pooled[n]),
            "first_k_lift_above_0": first}
        # recency (no verdict): k = 1 people by the step of their one occurrence, first against second half
        step1 = {p["id"]: p["occurrences"][0]["step"] for p in P["people"] if p["k"] == 1}
        half = max(step1.values()) / 2
        s1 = {i: float(np.mean([data[n]["people"][str(i)][a]["p"] for a in ATTRS])) for i in step1}
        base = float(score[n][0].mean())
        out["models"][n]["k1_lift_by_half"] = {
            "early": round(float(np.mean([v for i, v in s1.items() if step1[i] < half])) - base, 5),
            "late": round(float(np.mean([v for i, v in s1.items() if step1[i] >= half])) - base, 5)}
    if data:
        best = {n: ci(boot[n][64] - boot[n][0])[0] for n in data}
        out["gate_testable"] = any(v > 0 for v in best.values())
    if "B-1M" in data:
        for ref in ("D-50M", "D-100M"):
            if ref in data:
                d = pooled["B-1M"] - pooled[ref]
                out[f"B-1M_minus_{ref}"] = {"pooled_lift": round(float(d.mean()), 5), "ci95": ci(d)}
        diffs = [out.get(f"B-1M_minus_{r}") for r in ("D-50M", "D-100M")]
        if all(diffs) and out.get("gate_testable"):
            lo = [x["ci95"][0] for x in diffs]
            hi = [x["ci95"][1] for x in diffs]
            out["verdict"] = ("the table learns repeated facts better" if min(lo) > 0 else
                              "dense learns repeated facts better" if max(hi) < 0 else "no clear difference")
        elif all(diffs):
            out["verdict"] = "not testable at this size (no model above its k = 0 baseline at k = 64)"
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    for n, m in out["models"].items():
        print(n, "pooled", m["pooled_lift"], m["pooled_lift_ci95"], "first k", m["first_k_lift_above_0"])
        print("  lift", m["lift"])
        print("  acc ", m["accuracy"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--run", help="run directory (default: RUNS[model])")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else score(a.model, a.run or RUNS[a.model], a.device)
