"""HOP2 part B (REPORT.md, "Step 11", part B; criteria fixed before measuring): which table rows does B-1M read at
the bridge and at the end of the two-hop question, compared with the rows it reads for B itself? No training.

  python scripts/hop2_reads.py B-1M                                     -> report/hop2/reads_B-1M.json
  python scripts/hop2_reads.py B-1M-orig --run runs/hampter/B-1M-sparse-s0   (null test: never saw the pairs)
  python scripts/hop2_reads.py --step12 B-1M-r2 | B-1M | B-1M-orig --run ...  -> report/hop2d/ (step 12, LOOP)

Step 12: results are keyed by read site ("2", "6", "10", and "10r2" for layer 10's second read); the Bonferroni
interval covers all sites of the model (four for B-1M-r2), and the value control is paired with the original B-1M at
the same site, round 2 with its layer 10. Verdict for round 2 (REPORT.md, "Step 12"): "round 2 reads B's values" only
if its value control is above the null model's and its answer signal has an interval above 0. Without a verdict:
round 2 at the end of the two-hop question against round 1 at the end of B's own question.

Rows of one read: every memory layer (2, 6, 10) reads heads x knn rows of the shared table at every position, with
softmax weights per head; the weight of a row is the sum over heads / heads (the rows of one read sum to 1, a row
picked by two heads counts once with both weights). Overlap of two reads = sum over rows of min(weight, weight), 0 to 1.

Prompts (training format, as in scripts/eval_hop2.py), 200 pairs per condition:
- T(A, a)  `<|endoftext|>A\\n\\nA's mother was born in` (a = year; `grew up in`, `worked as a` for city, profession)
- R(A)     `<|endoftext|>A\\n\\nA's mother was` (the relation; the model writes B's name next)
- S(B, a)  `<|endoftext|>B\\n\\nB was born in` (B's own question, which every model gets right in H2)
Positions are aligned from the end: -1 is the last token (the answer position), -2 and -3 the other cue tokens, -4
the subject (" mother" in T, the last token of B's name in S; the same position for all three attributes of S).

Measures, each against all 200 B of the condition (pair i's true B against the 199 others):
- signal of pair i = overlap with its own B minus the mean overlap with the other 199, and the rank of its own B among
  the 200 (1 = most overlap; chance: mean rank 100.5, rank 1 in 0.5%). 95% bootstrap over pairs, and a Bonferroni
  interval over the three layers (98.3%), on which the verdicts rest.
- answer: T(A, a) at -1 against S(B, a) at -1, signal averaged over the three attributes.
- bridge: R(A) at its last token against S(B, year) at -4 (B's name).
- profile (without a verdict): T at -4..-1 against S at the same offsets.
- value control (gate of the answer verdict): the end of a DIRECT question (`A's mother was born in`, trained with
  its value; every model gets it right) against the end of S for the H2 B: same value against different value. Shows
  whether the rows that produce a value are shared across contexts at all; if not, an answer position without B's
  rows would mean nothing.
- SAME (without a verdict): answer, bridge and profile for the SAME pairs.
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
import eval_hop2 as E  # noqa: E402
from eval_hop2 import ATTRS, EOT  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

BATCH, BOOT = 64, 10000
OFFSETS = (-4, -3, -2, -1)
BONF = 100 * 0.05 / 3 / 2                       # two-sided, three read sites: percentiles 0.833 / 99.167 (set in main)


def record(model, seqs, offsets, device):
    """Rows read by every memory layer at the given offsets from the end of every sequence:
    idx (n, layers, offsets, heads*knn) int64, w the same shape float32 (softmax weight / heads)."""
    mems = model.memory_layers()
    for m in mems:
        m.record = True
    hk = mems[0].heads * mems[0].knn
    idx = torch.zeros(len(seqs), len(mems), len(offsets), hk, dtype=torch.long)
    w = torch.zeros(len(seqs), len(mems), len(offsets), hk)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), BATCH):
        sel = order[b:b + BATCH]
        L = max(len(seqs[i]) for i in sel)
        x = torch.zeros(len(sel), L, dtype=torch.long)
        for r, i in enumerate(sel):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])   # right padding: causal, BN in eval mode, so harmless
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model(x.to(device))
        for k, m in enumerate(mems):
            li = m.last_indices.view(len(sel), L, hk).cpu()
            lw = m.last_weights.view(len(sel), L, hk).float().cpu() / mems[0].heads
            for r, i in enumerate(sel):
                pos = [len(seqs[i]) + o for o in offsets]
                idx[i, k], w[i, k] = li[r, pos], lw[r, pos]
    return idx, w


def dedup(idx, w):
    """(n, S) slots -> the same shape with every row once (weights summed), padded with row -1 and weight 0."""
    oi, ow = torch.full_like(idx, -1), torch.zeros_like(w)
    for k in range(len(idx)):
        u, inv = torch.unique(idx[k], return_inverse=True)
        oi[k, :len(u)] = u
        ow[k, :len(u)] = torch.zeros(len(u)).scatter_add_(0, inv, w[k])
    return oi, ow


def overlap(x, y, device):
    """x, y: (idx, w) deduplicated, (n, S) and (m, S) -> (n, m) sum over rows of min(weight in x, weight in y)."""
    xi, xw, yi, yw = (t.to(device) for t in (*x, *y))
    out = torch.empty(len(xi), len(yi))
    for a in range(0, len(xi), 16):
        eq = xi[a:a + 16, None, :, None] == yi[None, :, None, :]
        mn = torch.minimum(xw[a:a + 16, None, :, None], yw[None, :, None, :])
        out[a:a + 16] = (mn * eq).sum((-1, -2)).cpu()   # padding: weight 0 on both sides, adds nothing
    return out.numpy()


def own_vs_others(O):
    """O (n, n), row i against its own B at column i: signal = own minus mean of the others, rank of own, and the
    mean overlap with the others (the level the signal sits on)."""
    n = len(O)
    d = np.diag(O)
    others = (O.sum(1) - d) / (n - 1)
    rank = 1 + (O > d[:, None]).sum(1) + 0.5 * ((O == d[:, None]).sum(1) - 1)
    return d - others, rank, others


def stats(v, rank=None, others=None):
    v = np.asarray(v)
    b = v[np.random.default_rng(0).integers(0, len(v), (BOOT, len(v)))].mean(1)
    out = {"mean": round(float(v.mean()), 6), "ci95": [round(float(x), 6) for x in np.percentile(b, [2.5, 97.5])],
           "ci_bonf": [round(float(x), 6) for x in np.percentile(b, [BONF, 100 - BONF])], "n": len(v)}
    if rank is not None:
        rank = np.asarray(rank)
        out.update({"mean_rank": round(float(rank.mean()), 2), "rank1": round(float((rank == 1).mean()), 4)})
    if others is not None:
        out["overlap_others"] = round(float(np.mean(others)), 5)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--step12", action="store_true", help="step 12 (LOOP): report/hop2d, runs/hop2d")
    a = ap.parse_args()
    if a.step12:
        E.step12()
    OUT = E.OUT
    run = a.run or E.RUNS[a.name]
    enc = tiktoken.get_encoding("gpt2")
    P = json.load(open(os.path.join(OUT, "pairs.json")))
    group = {c: [p for p in P["pairs"] if p["cond"] == c] for c in ("H2", "SAME", "DIRECT")}
    model = load(run, a.device)
    layers = [x[1:] for x in model.memory_sites()]     # read sites: "2", "6", "10" (+ "10r2" with a second read)
    global BONF
    BONF = 100 * 0.05 / len(layers) / 2
    t0 = time.time()

    def T(p, at):
        return [EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{P['cues2'][at]}")

    def S(p, at):
        return [EOT] + enc.encode(f"{p['B']}\n\n{p['B']}{P['cues'][at]}")

    def R(p):
        return [EOT] + enc.encode(f"{p['A']}\n\n{p['A']}{P['rel']}")

    for p in group["H2"][:3]:                           # the alignment the offsets assume
        for at in ATTRS:
            assert enc.decode(T(p, at)[-4:-3]) == " mother" and S(p, at)[:-3] == S(p, "year")[:-3]
            assert T(p, at)[-3:] == S(p, at)[-3:]
        assert R(p) == T(p, "year")[:-2]

    def reads(seqs, offsets):
        idx, w = record(model, seqs, offsets, a.device)
        return {(k, o): dedup(idx[:, k, j], w[:, k, j]) for k in range(len(layers)) for j, o in enumerate(offsets)}

    def sub(r, rows):
        return {key: (v[0][rows], v[1][rows]) for key, v in r.items()}

    res = {"model": a.name, "run": run, "device": a.device, "layers": layers, "bonferroni_sites": len(layers)}
    for cond in ("H2", "SAME"):
        g = group[cond]
        n = len(g)
        t_r = reads([T(p, at) for at in ATTRS for p in g], OFFSETS)    # row index: attribute * n + pair
        s_r = reads([S(p, at) for at in ATTRS for p in g], OFFSETS)
        r_r = reads([R(p) for p in g], (-1,))
        out = {"profile": {}, "bridge": {}}
        for o in OFFSETS:
            out["profile"][str(o)] = {}
            for k, layer in enumerate(layers):
                sig, rk, oth = [], [], []
                for j in range(len(ATTRS)):
                    rows = slice(j * n, (j + 1) * n)
                    s, r, x = own_vs_others(overlap(sub(t_r, rows)[(k, o)], sub(s_r, rows)[(k, o)], a.device))
                    sig.append(s)
                    rk.append(r)
                    oth.append(x)
                out["profile"][str(o)][str(layer)] = stats(np.mean(sig, 0), np.concatenate(rk), oth)
        out["answer"] = out["profile"]["-1"]
        if "10r2" in layers:                            # round 2 of the question against round 1 of B's question
            k2, k1 = layers.index("10r2"), layers.index("10")
            sig, rk, oth = [], [], []
            for j in range(len(ATTRS)):
                rows = slice(j * n, (j + 1) * n)
                s, r, x = own_vs_others(overlap(sub(t_r, rows)[(k2, -1)], sub(s_r, rows)[(k1, -1)], a.device))
                sig.append(s)
                rk.append(r)
                oth.append(x)
            out["answer_r2_vs_10"] = stats(np.mean(sig, 0), np.concatenate(rk), oth)
        for k, layer in enumerate(layers):
            s, r, x = own_vs_others(overlap(r_r[(k, -1)], sub(s_r, slice(0, n))[(k, -4)], a.device))
            out["bridge"][str(layer)] = stats(s, r, x)
        res[cond] = out
    # value control: end of the DIRECT question against the end of S for the H2 B, same value against different
    d, h = group["DIRECT"], group["H2"]
    d_r = reads([T(p, at) for at in ATTRS for p in d], (-1,))
    s_r = reads([S(p, at) for at in ATTRS for p in h], (-1,))
    res["value_control"], vc = {}, {}
    for k, layer in enumerate(layers):
        per = [[] for _ in d]
        for j, at in enumerate(ATTRS):
            O = overlap(sub(d_r, slice(j * len(d), (j + 1) * len(d)))[(k, -1)],
                        sub(s_r, slice(j * len(h), (j + 1) * len(h)))[(k, -1)], a.device)
            hv = np.array([p[at] for p in h])
            for i, p in enumerate(d):
                same = hv == p[at]
                if same.any():
                    per[i].append(O[i, same].mean() - O[i, ~same].mean())
        vc[str(layer)] = [float(np.mean(x)) for x in per if x]    # same DIRECT pairs in every model
        res["value_control"][str(layer)] = {**stats(vc[str(layer)]), "per_pair": [round(x, 6) for x in vc[str(layer)]]}
    # verdicts (criteria in REPORT.md, step 11, part B)
    h2 = res["H2"]
    res["verdict_bridge"] = ("B's rows are read at the bridge" if any(h2["bridge"][str(l)]["ci_bonf"][0] > 0
                                                                       for l in layers)
                             else "no B rows at the bridge")
    # gate: the value control above that of the original B-1M (never saw the pairs), paired over the same DIRECT pairs
    null = os.path.join(OUT, "reads_B-1M-orig.json")
    if a.name != "B-1M-orig" and os.path.exists(null):
        N = json.load(open(null))["value_control"]
        res["value_control_minus_null"] = {      # round 2 has no counterpart in the original: its layer 10
            str(l): stats(np.array(vc[str(l)]) - np.array(N[str(l).replace("r2", "")]["per_pair"])) for l in layers}
        gate = any(res["value_control_minus_null"][str(l)]["ci_bonf"][0] > 0 for l in layers)
    else:
        gate = any(res["value_control"][str(l)]["ci_bonf"][0] > 0 for l in layers)
    if not gate:
        res["verdict_answer"] = "not testable at the answer position (rows don't carry a value across contexts)"
    else:
        res["verdict_answer"] = ("B's rows are read at the answer position"
                                 if any(h2["answer"][str(l)]["ci_bonf"][0] > 0 for l in layers)
                                 else "no trace of B at the answer position")
    if "10r2" in layers and "value_control_minus_null" in res:
        g2 = res["value_control_minus_null"]["10r2"]["ci_bonf"][0] > 0
        res["verdict_round2"] = ("not testable for round 2 (its rows don't carry a value across contexts)" if not g2
                                 else "round 2 reads B's values" if h2["answer"]["10r2"]["ci_bonf"][0] > 0
                                 else "round 2 reads no B values")
    res["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, f"reads_{a.name}.json"), res)
    for cond in ("H2", "SAME"):
        for part in ("bridge", "answer"):
            print(a.name, cond, part, {l: (v["mean"], v["ci_bonf"], v["rank1"]) for l, v in res[cond][part].items()})
        for o in OFFSETS[:-1]:
            print(a.name, cond, "offset", o, {l: (v["mean"], v["ci_bonf"]) for l, v in res[cond]["profile"][str(o)].items()})
    print(a.name, "value control", {l: (v["mean"], v["ci_bonf"], v["n"]) for l, v in res["value_control"].items()})
    if "value_control_minus_null" in res:
        print(a.name, "value control minus null", {l: (v["mean"], v["ci_bonf"])
                                                    for l, v in res["value_control_minus_null"].items()})
    print(a.name, "overlap with the others (answer, bridge)", {l: (res["H2"]["answer"][l]["overlap_others"],
                                                               res["H2"]["bridge"][l]["overlap_others"]) for l in
                                                           res["H2"]["answer"]})
    print(a.name, res["verdict_bridge"], "|", res["verdict_answer"], "|", res.get("verdict_round2", "-"),
          f"| {res['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
