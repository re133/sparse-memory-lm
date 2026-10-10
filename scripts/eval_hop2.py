"""HOP2 evaluation (criteria fixed in REPORT.md, "Step 11", before the runs): does a model chain two made-up facts it
learned separately ("A's mother was B", "B was born in 1912") when asked "A's mother was born in"?

  python scripts/eval_hop2.py B-1M [--run DIR] [--device cpu] [--host_table]   -> report/hop2/eval_B-1M.json
  python scripts/eval_hop2.py --summary                                          -> report/hop2/summary.json

Pairs and conditions: scripts/make_hop2.py (report/hop2/pairs.json). One forward pass per prompt, in the training
format, bf16 autocast on the GPU as in validation, product-key models with the full fp32 table:
- two-hop question (every condition), `<|endoftext|>A\\n\\nA's mother was born in` / `grew up in` / `worked as a`:
  the probability of B's true value, normalised over all values of the attribute (118 years, 107 cities, 48
  professions), whether it is the most likely of them, and the probability of the value of another pair's B of the
  same condition (the false bridge). Score of a pair = mean over the three attributes.
- single step B (H2, and NULL as its baseline), `<|endoftext|>B\\n\\nB was born in` ...: as in FACTK.
- single step relation (H2, SAME, NULL), `<|endoftext|>A\\n\\nA's mother was`: log probability of B's name tokens,
  and whether greedy decoding writes exactly B's name (every name token the most likely one).

Summary: lift(condition) = mean score of the condition minus mean score of NULL, 95% bootstrap over pairs (each
condition resampled on its own, paired across models: the same resampled pairs for every model).
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
from eval_factk import load  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

OUT = os.path.join(ROOT, "report", "hop2")
RUNS = {"B-1M": "runs/hop2/B-1M-hop2-s0", "D-50M": "runs/hop2/D-50M-hop2-s0", "D-100M": "runs/hop2/D-100M-hop2-s0",
        "B-16M": "runs/hop2/B-16M-hop2-s0"}
ORIG_PPL = {"B-1M": 21.837, "D-50M": 22.452, "D-100M": 20.270, "B-16M": 19.960}   # the same runs without the pairs
ATTRS = ("year", "city", "job")
EOT, BATCH, BOOT, GATE_ACC = 50256, 64, 10000, 0.90


def false_partner(pairs):
    """For every pair another pair of the same condition (a fixed derangement): its B's values are the false bridge."""
    rng = np.random.default_rng(1)
    out = {}
    for cond in {p["cond"] for p in pairs}:
        ids = [p["id"] for p in pairs if p["cond"] == cond]
        perm = rng.permutation(len(ids))
        for k, i in enumerate(perm):
            out[ids[i]] = ids[perm[(k + 1) % len(perm)]]
    return out


def forward(model, seqs, device):
    """Logits (float32, on CPU) for a list of token lists, right-padded, in batches."""
    res = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        L = max(len(seqs[i]) for i in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device))
        for r, i in enumerate(idx):
            res[i] = logits[r, :len(seqs[i])].float().cpu()
    return res


@torch.no_grad()
def score(name, run, device, host_table=False):
    enc = tiktoken.get_encoding("gpt2")
    P = json.load(open(os.path.join(OUT, "pairs.json")))
    pairs = P["pairs"]
    by_id = {p["id"]: p for p in pairs}
    vals = {a: P["values"][a] for a in ATTRS}
    cand = {a: torch.tensor([enc.encode(" " + v)[0] for v in vals[a]]) for a in ATTRS}
    partner = false_partner(pairs)
    model = load(run, device, host_table)
    t0 = time.time()
    res = {str(p["id"]): {"two_hop": {}, "b": {}, "rel": {}} for p in pairs}
    # value questions: two-hop for every pair, single step B for H2 and NULL
    items = []
    for p in pairs:
        for a in ATTRS:
            items.append(("two_hop", p, a, [EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{P['cues2'][a]}")))
            if p["cond"] in ("H2", "NULL"):
                items.append(("b", p, a, [EOT] + enc.encode(f"{p['B']}\n\n{p['B']}{P['cues'][a]}")))
    logits = forward(model, [it[3] for it in items], device)
    for (kind, p, a, ids), lg in zip(items, logits):
        lp = torch.log_softmax(lg[len(ids) - 1][cand[a]], -1)
        true = vals[a].index(p[a])
        r = {"p": round(float(lp[true].exp()), 6), "hit": bool(int(lp.argmax()) == true)}
        if kind == "two_hop":
            r["p_false"] = round(float(lp[vals[a].index(by_id[partner[p["id"]]][a])].exp()), 6)
        res[str(p["id"])][kind][a] = r
    # relation: B's name after "A's mother was", teacher-forced
    rel = [p for p in pairs if p["cond"] in ("H2", "SAME", "NULL")]
    seqs, spans = [], []
    for p in rel:
        pre = [EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{P['rel']}")
        b_ids = enc.encode(" " + p["B"])
        seqs.append(pre + b_ids)
        spans.append((len(pre), b_ids))
    for p, lg, (n_pre, b_ids) in zip(rel, forward(model, seqs, device), spans):
        lsm = torch.log_softmax(lg[n_pre - 1:n_pre - 1 + len(b_ids)], -1)
        tgt = torch.tensor(b_ids)
        res[str(p["id"])]["rel"] = {"logp": round(float(lsm[torch.arange(len(b_ids)), tgt].sum()), 4),
                                    "greedy": bool((lsm.argmax(-1) == tgt).all())}
    out = {"model": name, "run": run, "device": device, "seconds": round(time.time() - t0, 1), "pairs": res}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, f"eval_{name}.json"), out)
    for cond in P["conditions"]:
        ids = [str(p["id"]) for p in pairs if p["cond"] == cond]
        s = np.mean([np.mean([res[i]["two_hop"][a]["p"] for a in ATTRS]) for i in ids])
        acc = np.mean([res[i]["two_hop"][a]["hit"] for i in ids for a in ATTRS])
        extra = ""
        if cond in ("H2", "NULL"):
            extra += f"  B single step acc {np.mean([res[i]['b'][a]['hit'] for i in ids for a in ATTRS]):.3f}"
        if cond in ("H2", "SAME", "NULL"):
            extra += f"  relation greedy {np.mean([res[i]['rel']['greedy'] for i in ids]):.3f}"
        print(f"{name} {cond:6s} two-hop score {s:.4f} acc {acc:.3f}{extra}", flush=True)


def ci(d):
    return [round(float(np.percentile(d, q)), 5) for q in (2.5, 97.5)]


def summary():
    P = json.load(open(os.path.join(OUT, "pairs.json")))
    conds = P["conditions"]
    ids = {c: [str(p["id"]) for p in P["pairs"] if p["cond"] == c] for c in conds}
    data = {n: json.load(open(f)) for n in RUNS if os.path.exists(f := os.path.join(OUT, f"eval_{n}.json"))}
    rng = np.random.default_rng(0)
    draws = {c: rng.integers(0, len(ids[c]), (BOOT, len(ids[c]))) for c in conds}
    out = {"bootstrap": BOOT, "conditions": conds, "models": {}}
    boot = {}
    for n, d in data.items():
        R = d["pairs"]
        sc = {c: np.array([np.mean([R[i]["two_hop"][a]["p"] for a in ATTRS]) for i in ids[c]]) for c in conds}
        boot[n] = {c: sc[c][draws[c]].mean(1) for c in conds}
        lift = {c: boot[n][c] - boot[n]["NULL"] for c in conds if c != "NULL"}
        acc = lambda c, k: float(np.mean([R[i][k][a]["hit"] for i in ids[c] for a in ATTRS]))
        m = {"score": {c: round(float(sc[c].mean()), 5) for c in conds},
             "accuracy": {c: round(acc(c, "two_hop"), 4) for c in conds},
             "lift": {c: round(float(sc[c].mean() - sc["NULL"].mean()), 5) for c in lift},
             "lift_ci95": {c: ci(v) for c, v in lift.items()},
             "same_minus_h2_ci95": ci(boot[n]["SAME"] - boot[n]["H2"]),
             "false_bridge_h2": {"p_true": round(float(np.mean([R[i]["two_hop"][a]["p"] for i in ids["H2"]
                                                                for a in ATTRS])), 5),
                                 "p_false": round(float(np.mean([R[i]["two_hop"][a]["p_false"] for i in ids["H2"]
                                                                 for a in ATTRS])), 5)},
             "b_single_step_acc": {c: round(acc(c, "b"), 4) for c in ("H2", "NULL")},
             "relation_greedy": {c: round(float(np.mean([R[i]["rel"]["greedy"] for i in ids[c]])), 4)
                                 for c in ("H2", "SAME", "NULL")},
             "relation_logp": {c: round(float(np.mean([R[i]["rel"]["logp"] for i in ids[c]])), 3)
                               for c in ("H2", "SAME", "NULL")}}
        m["gate_single_steps"] = (m["b_single_step_acc"]["H2"] >= GATE_ACC and m["relation_greedy"]["H2"] >= GATE_ACC)
        m["gate_format"] = m["lift_ci95"]["DIRECT"][0] > 0
        m["testable"] = m["gate_single_steps"] and m["gate_format"]
        m["verdict"] = (("chains" if m["lift_ci95"]["H2"][0] > 0 else "does not chain") if m["testable"] else
                        "not testable: " + ("the single facts don't sit" if not m["gate_single_steps"] else
                                            "the question format doesn't work"))
        try:
            ppl = json.load(open(os.path.join(ROOT, d["run"], "run-info.json")))["results"]["val_ppl"]
            m["val_ppl"] = round(ppl, 3)
            m["val_ppl_vs_original"] = f"{100 * (ppl / ORIG_PPL[n] - 1):+.2f}%" if n in ORIG_PPL else None
        except (OSError, KeyError, ValueError):
            pass
        out["models"][n] = m
    h2 = {n: boot[n]["H2"] - boot[n]["NULL"] for n in data}
    for a, b in (("B-1M", "D-50M"), ("B-1M", "D-100M"), ("B-16M", "B-1M"), ("B-16M", "D-100M")):
        if a in data and b in data:
            out[f"H2_{a}_minus_{b}"] = {"lift": round(float((h2[a] - h2[b]).mean()), 5), "ci95": ci(h2[a] - h2[b]),
                                        "same_lift_ci95": ci((boot[a]["SAME"] - boot[a]["NULL"])
                                                             - (boot[b]["SAME"] - boot[b]["NULL"]))}
    if all(n in data for n in ("B-1M", "D-50M", "D-100M")):
        if all(out["models"][n]["testable"] for n in ("B-1M", "D-50M", "D-100M")):
            lo = [out[f"H2_B-1M_minus_{r}"]["ci95"][0] for r in ("D-50M", "D-100M")]
            hi = [out[f"H2_B-1M_minus_{r}"]["ci95"][1] for r in ("D-50M", "D-100M")]
            out["verdict"] = ("the table chains better" if min(lo) > 0 else "dense chains better" if max(hi) < 0
                              else "no clear difference")
        else:
            out["verdict"] = "not testable (a gate failed for at least one model)"
    if "B-16M" in data and "B-1M" in data:
        if out["models"]["B-16M"]["testable"] and out["models"]["B-1M"]["testable"]:
            c = out["H2_B-16M_minus_B-1M"]["ci95"]
            out["verdict_bigger_table"] = ("the bigger table chains better" if c[0] > 0 else
                                           "the bigger table chains worse" if c[1] < 0 else "no clear difference")
        else:
            out["verdict_bigger_table"] = "not testable (a gate failed for B-16M or B-1M)"
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    for n, m in out["models"].items():
        print(n, m["verdict"], "| lift", m["lift"], "| ci", m["lift_ci95"], "| acc", m["accuracy"])
        print("   B single", m["b_single_step_acc"], "relation greedy", m["relation_greedy"], "false bridge",
              m["false_bridge_h2"], "PPL", m.get("val_ppl"), m.get("val_ppl_vs_original"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--run", help="run directory (default: RUNS[model])")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--host_table", action="store_true", help="keep the fp32 table in RAM (B-16M on a 16 GB card)")
    a = ap.parse_args()
    summary() if a.summary else score(a.model, a.run or RUNS[a.model], a.device, a.host_table)
