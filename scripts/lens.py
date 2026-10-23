"""Step 19 (LENS): at which layer does the bridge B become readable? Logit lens on B-1M-hop2 (step 11).

  python scripts/lens.py [--device cpu]     -> report/lens/summary.json

Step 11's pairs (report/hop2/pairs.json), condition H2 (A's mother is B and B's facts were learned in separate
articles) against NULL (never linked), 200 pairs each. Two prompts per pair in the training format:
  bridge   `<|endoftext|>A\\n\\nA's mother was`        next token: the first token of B's name
  two-hop  `<|endoftext|>A\\n\\nA's mother was born in` (also grew up in / worked as a), last position
At every probe point (the embedding, the residual stream after each block, and the input of each memory read, i.e.
after the attention of blocks 2, 6, 10) the last position goes through the final norm and the output layer. Hit: B's
first token is the most likely among the distinct first tokens of the 200 B names of the condition.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_factk import load  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

RUN = "runs/hop2/B-1M-hop2-s0"
OUT = os.path.join(ROOT, "report", "lens")
EOT, BATCH, BOOT = 50256, 32, 10000


@torch.no_grad()
def lens(model, seqs, device):
    """{probe: (n, vocab) log-probs at each sequence's last position} via the final norm and the output layer."""
    probes = {}
    hooks = [model.layers[0].register_forward_pre_hook(lambda m, a: probes.__setitem__("emb", a[0]))]
    for i, blk in enumerate(model.layers):
        hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: probes.__setitem__(f"block{i}", o)))
        if blk.is_memory:
            hooks.append(blk.ffn_norm.register_forward_pre_hook(
                lambda m, a, i=i: probes.__setitem__(f"mem{i}_in", a[0])))
    out = {}
    try:
        for b in range(0, len(seqs), BATCH):
            part = seqs[b:b + BATCH]
            L = max(len(s) for s in part)
            x = torch.zeros(len(part), L, dtype=torch.long)
            for r, s in enumerate(part):
                x[r, :len(s)] = torch.tensor(s)
            probes.clear()
            model(x.to(device))
            last = torch.tensor([len(s) - 1 for s in part])
            for k, h in probes.items():
                h = h[torch.arange(len(part)), last.to(h.device)]
                lp = torch.log_softmax(model.lm_head(model.norm(h)).float(), -1).cpu()
                out.setdefault(k, []).append(lp)
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v) for k, v in out.items()}


def wilson(k, n):
    z = 1.96
    p, d = k / n, 1 + z * z / n
    c, w = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - w, 4), round(c + w, 4)]


def main(device):
    enc = tiktoken.get_encoding("gpt2")
    P = json.load(open(os.path.join(ROOT, "report", "hop2", "pairs.json")))
    model = load(RUN, device)
    order = None
    res = {}
    for cond in ("H2", "NULL"):
        group = [p for p in P["pairs"] if p["cond"] == cond]
        first = [enc.encode(" " + p["B"])[0] for p in group]
        cand = sorted(set(first))
        pos = torch.tensor([cand.index(t) for t in first])
        cand_t = torch.tensor(cand)
        prompts = {"bridge": [[EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{P['rel']}") for p in group]}
        for a, cue in P["cues2"].items():
            prompts[f"two_hop_{a}"] = [[EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{cue}") for p in group]
        res[cond] = {"pairs": len(group), "distinct_first_tokens": len(cand), "chance": round(1 / len(cand), 4)}
        for name, seqs in prompts.items():
            lp = lens(model, seqs, device)
            if order is None:
                order = list(lp)
            r = {}
            for k in order:
                sub = lp[k][:, cand_t]
                hit = (sub.argmax(1) == pos).numpy()
                p_true = torch.softmax(sub, 1)[torch.arange(len(group)), pos].numpy()
                r[k] = {"hit": round(float(hit.mean()), 4), "hit_ci95": wilson(int(hit.sum()), len(hit)),
                        "p_norm": round(float(p_true.mean()), 4),
                        "rank_in_vocab_median": int(np.median((lp[k] > lp[k][torch.arange(len(group)),
                                                                              cand_t[pos]][:, None]).sum(1).numpy())),
                        "hits": hit.astype(int).tolist()}
            res[cond][name] = r
            print(cond, name, " ".join(f"{k}:{r[k]['hit']:.2f}" for k in order), flush=True)
    h2, null = res["H2"]["bridge"], res["NULL"]["bridge"]
    final = order[-1]
    gate = h2[final]["hit"] >= 0.9
    k10 = "mem10_in"
    above_null = h2[k10]["hit_ci95"][0] > null[k10]["hit_ci95"][1]
    verdict = ("not testable (the model doesn't name B at the bridge)" if not gate else
               "B is readable before the layer-10 read" if h2[k10]["hit"] >= 0.5 and above_null else
               "B forms only at or after the layer-10 read")
    first50 = next((k for k in order if h2[k]["hit"] >= 0.5), None)
    summary = {"run": RUN, "device": device, "probes": order, "verdict": verdict, "gate_final_hit": h2[final]["hit"],
               "bridge_hit_at_mem10_in": h2[k10]["hit"], "bridge_hit_at_mem10_in_ci95": h2[k10]["hit_ci95"],
               "null_hit_at_mem10_in_ci95": null[k10]["hit_ci95"], "first_probe_with_half_hits": first50,
               "conditions": res}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "summary.json"), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    main(ap.parse_args().device)
