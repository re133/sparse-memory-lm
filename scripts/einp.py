"""Step 14 (EINP): write new facts straight into the table of a finished model instead of training it.

  python scripts/einp.py split            -> report/einp/groups.json (no model)
  python scripts/einp.py null             -> report/einp/null.json   (unmodified model: groups, overlap, val PPL)
  python scripts/einp.py dev              -> report/einp/dev.json    (step count / step size grid on the dev people)
  python scripts/einp.py test             -> report/einp/test.json   (the chosen setting on the test people)
  python scripts/einp.py summary          -> report/einp/summary.json

Model: B-16M-factk (runs/factk/B-16M-factk-s0). It learned the made-up people of step 10 by training (k = 1 ... 64
repetitions); the 300 k = 0 people it never saw. Those are split, seeded, into dev (choose the setting), test (the
measurement) and unwritten (the control: scoring is normalised over the candidate values, so a write could raise a
value for everyone).

Writing a person: their article in the training format of step 10 (`<|endoftext|>Name\\n\\n` + the three sentences in
a seeded random order) goes through the model once per step. The loss is the cross entropy of the three value tokens
only (not of the made-up name). Only the table rows read in that pass change, by plain gradient descent on those
rows: value += -lr * gradient. No optimizer state, nothing else in the model changes. People are written one after
another into the same table.

The table stays in RAM (fp32, step 8), the rest on the GPU, bf16 autocast as in step 10's evaluation, so the rows a
prompt reads while writing are the rows it reads while being scored. Writes go into the RAM copy that load_state_dict
filled, never into the checkpoint file; `restore()` puts back the original rows between settings.
"""
import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402

OUT = os.path.join(ROOT, "report", "einp")
FACTK = os.path.join(ROOT, "report", "factk")
RUN = "runs/factk/B-16M-factk-s0"
ATTRS = ("year", "city", "job")
EOT, SEQ, BATCH, BOOT = 50256, 1024, 64, 10000
SEED = 20261022
GROUPS = {"dev": 50, "test": 150, "unwritten": 100}
KEEP = (16, 32, 64)                         # trained people whose facts must survive the writes
STEPS, LRS = (1, 4, 16), (0.1, 1.0, 10.0, 100.0, 1000.0)
DEV_WINDOWS = 64                            # val windows for choosing the setting (the main check uses all)
DEV_PPL_LIMIT = 0.01                        # a setting may raise the dev val PPL by at most 1%


# ---------------------------------------------------------------------------------------------------- people
def persons():
    return json.load(open(os.path.join(FACTK, "persons.json")))


def article(enc, P, pr, rng):
    """Token ids of a person's article and, per attribute, the position of its value token and its rank (0 = first
    sentence, so the scoring prompt is an exact prefix of the article). Positions as in make_factk.py: the tokens of
    the text up to the cue, so BPE merges are those of the whole article."""
    attrs = [str(a) for a in rng.permutation(list(ATTRS))]
    sentences = [f"{pr['name']}{P['cues'][a]} {pr[a]}." for a in attrs]
    ids = [EOT] + enc.encode(f"{pr['name']}\n\n" + " ".join(sentences))
    pos = {}
    for j, a in enumerate(attrs):
        pre = [EOT] + enc.encode(f"{pr['name']}\n\n" + " ".join(sentences[:j] + [f"{pr['name']}{P['cues'][a]}"]))
        val = enc.encode(" " + pr[a])
        assert len(val) == 1 and ids[:len(pre)] == pre and ids[len(pre)] == val[0]
        pos[a] = {"pos": len(pre), "rank": j}
    return ids, pos


def split():
    P, enc = persons(), tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(SEED)
    zero = [p["id"] for p in P["people"] if p["k"] == 0]
    order = [int(i) for i in rng.permutation(zero)]
    groups, a = {}, 0
    for g, n in GROUPS.items():
        groups[g], a = order[a:a + n], a + n
    by_id = {p["id"]: p for p in P["people"]}
    arts = {}
    for i in groups["dev"] + groups["test"]:
        ids, pos = article(enc, P, by_id[i], rng)
        arts[str(i)] = {"ids": ids, "values": pos}
    write_order = [int(i) for i in rng.permutation(groups["test"])]
    out = {"seed": SEED, "groups": groups, "test_write_order": write_order, "articles": arts}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "groups.json"), out)
    print({g: len(v) for g, v in groups.items()}, "articles", len(arts))


# ---------------------------------------------------------------------------------------------------- writing
class TableWriter:
    """Gradient descent on the table rows one forward pass reads, nothing else. While active, every memory layer reads
    its rows through a small leaf tensor (a copy of those rows) instead of the table, so backward fills exactly the
    rows read; step() adds -lr * gradient to them in the table and remembers their original values for restore()."""

    def __init__(self, model, layers=None, top=None, idf=None):
        self.mems = model.memory_layers()
        self.layers = layers                    # indices into memory_layers() whose reads are written (None: all)
        # top: write only the `top` rows with the highest tf * idf among those with a gradient (sparse memory
        # finetuning, Lin et al. 2025): tf = how often this pass reads the row before its last value token, idf =
        # a CPU tensor over all rows (rare in background text = high)
        self.top, self.idf, self.limit = top, idf, None
        self.table = self.mems[0].values.weight
        assert all(m.values.weight is self.table for m in self.mems), "one shared table"
        assert self.table.device.type == "cpu" and self.table.dtype == torch.float32
        self.saved = torch.zeros(self.table.shape[0], dtype=torch.bool)
        self.orig_rows, self.orig_vals = [], []
        self.parts = []

    def _read(self, mem, indices, weights):
        rows, inv = torch.unique(indices.reshape(-1), return_inverse=True)
        sub = self.table.detach().index_select(0, rows.cpu()).to(indices.device)
        if self.layers is None or self.mems.index(mem) in self.layers:
            sub.requires_grad_(True)
            tf = None
            if self.top is not None:
                seen = indices.reshape(-1, indices.shape[-1])[:self.limit]   # positions that get a gradient
                tf = torch.zeros(len(rows), device=indices.device).index_add_(
                    0, torch.searchsorted(rows, seen.reshape(-1)), torch.ones(seen.numel(), device=indices.device))
                tf = tf.cpu()
            self.parts.append((rows.cpu(), sub, tf))
        return F.embedding_bag(inv.view_as(indices), sub, per_sample_weights=weights.to(sub.dtype), mode="sum")

    def __enter__(self):
        self.parts = []
        for m in self.mems:
            m.read_values = types.MethodType(lambda mem, i, w: self._read(mem, i, w), m)
        return self

    def __exit__(self, *exc):
        for m in self.mems:
            del m.read_values

    @torch.no_grad()
    def step(self, lr):
        rows = torch.cat([r for r, _, _ in self.parts])
        grad = torch.cat([s.grad.float().cpu() if s.grad is not None else torch.zeros_like(s, device="cpu")
                          for _, s, _ in self.parts])
        rows, inv = torch.unique(rows, return_inverse=True)
        g = torch.zeros(len(rows), grad.shape[1]).index_add_(0, inv, grad)
        nz = g.abs().sum(1) > 0                  # rows read only after the last value token get no gradient
        if self.top is not None:
            tf = torch.zeros(len(rows)).index_add_(0, inv, torch.cat([t for _, _, t in self.parts]))
            score = torch.where(nz, tf * self.idf[rows], torch.tensor(-1.0))
            keep = torch.zeros(len(rows), dtype=torch.bool)
            keep[score.topk(min(self.top, int(nz.sum()))).indices] = True
            nz = nz & keep
        rows, g = rows[nz], g[nz]
        new = rows[~self.saved[rows]]
        if len(new):
            self.orig_rows.append(new)
            self.orig_vals.append(self.table.detach().index_select(0, new).clone())
            self.saved[new] = True
        self.table.detach().index_add_(0, rows, g, alpha=-lr)
        self.parts = []
        return len(rows)

    @torch.no_grad()
    def restore(self):
        for r, v in zip(self.orig_rows, self.orig_vals):
            self.table.detach().index_copy_(0, r, v)
        self.saved.zero_()
        self.orig_rows, self.orig_vals = [], []


def write_person(model, writer, art, steps, lr, device):
    x = torch.tensor([art["ids"]], device=device)
    pos = [v["pos"] for v in art["values"].values()]
    tgt = torch.tensor([art["ids"][p] for p in pos], device=device)
    rows = 0
    writer.limit = max(pos)                  # positions before the last value token (the ones with a gradient)
    for _ in range(steps):
        with writer, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits[0, [p - 1 for p in pos]].float(), tgt)
        loss.backward()
        rows = writer.step(lr)
    return rows


# ---------------------------------------------------------------------------------------------------- measuring
def load_model(device):
    from eval_factk import load
    model = load(RUN, device, host_table=device == "cuda")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def score(model, P, ids, device):
    """Step 10's measure for the given people: probability of the true value, normalised over all values of the
    attribute, and whether it is the most likely one. Returns {id: {attr: {"p", "hit"}}}."""
    enc = tiktoken.get_encoding("gpt2")
    by_id = {p["id"]: p for p in P["people"]}
    cand = {a: torch.tensor([enc.encode(" " + v)[0] for v in P["values"][a]], device=device) for a in ATTRS}
    items = []
    for i in ids:
        pr = by_id[i]
        for a in ATTRS:
            items.append((i, a, [EOT] + enc.encode(f"{pr['name']}\n\n{pr['name']}{P['cues'][a]}"),
                          P["values"][a].index(pr[a])))
    order = sorted(range(len(items)), key=lambda k: len(items[k][2]))
    res = {}
    for b in range(0, len(order), BATCH):
        idx = order[b:b + BATCH]
        L = max(len(items[k][2]) for k in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, k in enumerate(idx):
            x[r, :len(items[k][2])] = torch.tensor(items[k][2])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device))
        for r, k in enumerate(idx):
            i, a, t, true = items[k]
            lp = torch.log_softmax(logits[r, len(t) - 1].float()[cand[a]], -1)
            res.setdefault(i, {})[a] = {"p": float(lp[true].exp()), "hit": bool(int(lp.argmax()) == true)}
    return res


@torch.no_grad()
def val_ppl(model, device, windows=None):
    tok = torch.from_numpy(np.asarray(load_split("validation", "wikipedia_factk"), dtype=np.int64))
    full = (tok.numel() - 1) // SEQ if windows is None else windows
    nll, n = 0.0, 0
    for a in range(0, full, 8):
        b = min(full, a + 8)
        x = tok[a * SEQ:b * SEQ].view(-1, SEQ).to(device)
        y = tok[a * SEQ + 1:b * SEQ + 1].view(-1, SEQ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        nll += float(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        n += y.numel()
    return math.exp(nll / n)


@torch.no_grad()
def reads(model, seqs, at, device):
    """Rows and weights each memory layer reads at position at[k] of seqs[k]: list over layers of (rows, weights),
    rows (N, heads*knn) int64, weights (N, heads*knn) fp32 (each head's weights sum to 1)."""
    mems = model.memory_layers()
    for m in mems:
        m.record = True
    out = [([], []) for _ in mems]
    for b in range(0, len(seqs), BATCH):
        part = seqs[b:b + BATCH]
        L = max(len(s) for s in part)
        x = torch.zeros(len(part), L, dtype=torch.long)
        for r, s in enumerate(part):
            x[r, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            model(x.to(device))
        flat = torch.tensor([r * L + at[b + r] for r in range(len(part))], device=device)
        for li, m in enumerate(mems):
            out[li][0].append(m.last_indices.reshape(len(part) * L, -1)[flat].cpu())
            out[li][1].append(m.last_weights.reshape(len(part) * L, -1)[flat].float().cpu())
    for m in mems:
        m.record = False
    return [(torch.cat(r), torch.cat(w)) for r, w in out]


def overlap(r1, w1, r2, w2):
    """Shared read weight of two reads of one layer: sum over common rows of the smaller weight, divided by the
    number of heads (so 1 = the same rows with the same weights, 0 = no common row)."""
    a, b = {}, {}
    for r, w in zip(r1.tolist(), w1.tolist()):
        a[r] = a.get(r, 0.0) + w
    for r, w in zip(r2.tolist(), w2.tolist()):
        b[r] = b.get(r, 0.0) + w
    return sum(min(a[r], b[r]) for r in a.keys() & b.keys()) / (sum(a.values()))


def covered(r, w, written):
    """Share of a read's weight that falls on rows the writes touched."""
    return float(sum(x for row, x in zip(r.tolist(), w.tolist()) if row in written) / sum(w.tolist()))


def prompts(P, ids):
    enc = tiktoken.get_encoding("gpt2")
    by_id = {p["id"]: p for p in P["people"]}
    seqs, keys = [], []
    for i in ids:
        for a in ATTRS:
            seqs.append([EOT] + enc.encode(f"{by_id[i]['name']}\n\n{by_id[i]['name']}{P['cues'][a]}"))
            keys.append((i, a))
    return seqs, keys


def mean_score(res, ids):
    return float(np.mean([np.mean([res[i][a]["p"] for a in ATTRS]) for i in ids]))


# ---------------------------------------------------------------------------------------------------- stages
def null(device):
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    keep = [p["id"] for p in P["people"] if p["k"] in KEEP]
    model = load_model(device)
    t0 = time.time()
    ids = G["groups"]["dev"] + G["groups"]["test"] + G["groups"]["unwritten"] + keep
    res = score(model, P, ids, device)
    out = {"score": {g: mean_score(res, v) for g, v in G["groups"].items()},
           "score_keep": {k: mean_score(res, [p["id"] for p in P["people"] if p["k"] == k]) for k in KEEP},
           "people": {str(i): res[i] for i in ids}}
    # overlap of the rows read where the value is predicted: between different people (same cue) and, per person,
    # between the scoring prompt and the same attribute's position in the article that will be written
    rng = np.random.default_rng(SEED + 1)
    lv = {p["id"]: p["k"] for p in P["people"]}
    ov = {}
    for name, grp in (("k0", G["groups"]["test"] + G["groups"]["unwritten"]),
                      ("k64", [i for i in lv if lv[i] == 64])):
        seqs, keys = prompts(P, grp)
        rd = reads(model, seqs, [len(s) - 1 for s in seqs], device)
        pairs = []
        for _ in range(2000):
            a, b = rng.choice(len(grp), 2, replace=False)
            at = int(rng.integers(3))
            pairs.append((3 * a + at, 3 * b + at))
        ov[name] = [float(np.mean([overlap(r[p], w[p], r[q], w[q]) for p, q in pairs])) for r, w in rd]
    out["overlap_between_people"] = ov
    seqs, keys = prompts(P, G["groups"]["test"])
    rd_prompt = reads(model, seqs, [len(s) - 1 for s in seqs], device)
    art = [G["articles"][str(i)] for i, _ in keys]
    rd_art = reads(model, [x["ids"] for x in art], [x["values"][a]["pos"] - 1 for x, (_, a) in zip(art, keys)],
                   device)
    by_rank = {}
    for k, (i, a) in enumerate(keys):
        rank = G["articles"][str(i)]["values"][a]["rank"]
        by_rank.setdefault(rank, []).append(np.mean([overlap(rp[k], wp[k], ra[k], wa[k])
                                                      for (rp, wp), (ra, wa) in zip(rd_prompt, rd_art)]))
    out["overlap_prompt_vs_article"] = {str(r): float(np.mean(v)) for r, v in sorted(by_rank.items())}
    out["val_ppl"] = val_ppl(model, device)
    out["val_ppl_dev"] = val_ppl(model, device, DEV_WINDOWS)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, "null.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "people"}, indent=1))


def dev(device):
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    N = json.load(open(os.path.join(OUT, "null.json")))
    model = load_model(device)
    writer = TableWriter(model)
    grid = []
    for steps in STEPS:
        for lr in LRS:
            t0 = time.time()
            rows = [write_person(model, writer, G["articles"][str(i)], steps, lr, device) for i in G["groups"]["dev"]]
            res = score(model, P, G["groups"]["dev"], device)
            ppl = val_ppl(model, device, DEV_WINDOWS)
            writer.restore()
            grid.append({"steps": steps, "lr": lr, "score": mean_score(res, G["groups"]["dev"]),
                         "accuracy": float(np.mean([res[i][a]["hit"] for i in G["groups"]["dev"] for a in ATTRS])),
                         "val_ppl_dev": ppl, "val_ppl_change": ppl / N["val_ppl_dev"] - 1,
                         "rows_per_step": float(np.mean(rows)), "seconds": round(time.time() - t0, 1)})
            print(grid[-1], flush=True)
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    ok = [g for g in grid if g["val_ppl_change"] <= DEV_PPL_LIMIT] or grid
    best = max(ok, key=lambda g: (round(g["score"], 6), -g["steps"], -g["lr"]))
    write_json(os.path.join(OUT, "dev.json"), {"grid": grid, "chosen": {"steps": best["steps"], "lr": best["lr"]},
                                               "rule": f"highest dev score with dev val PPL +{DEV_PPL_LIMIT:.0%} or "
                                                       f"less; ties: fewer steps, then smaller lr"})
    print("chosen", best)


def test(device):
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    chosen = json.load(open(os.path.join(OUT, "dev.json")))["chosen"]
    keep = [p["id"] for p in P["people"] if p["k"] in KEEP]
    model = load_model(device)
    writer = TableWriter(model)
    t0 = time.time()
    for i in G["test_write_order"]:
        write_person(model, writer, G["articles"][str(i)], chosen["steps"], chosen["lr"], device)
    t_write = time.time() - t0
    written = set(torch.cat(writer.orig_rows).tolist()) if writer.orig_rows else set()
    ids = G["groups"]["dev"] + G["groups"]["test"] + G["groups"]["unwritten"] + keep
    res = score(model, P, ids, device)
    seqs, keys = prompts(P, G["groups"]["test"])
    rd = reads(model, seqs, [len(s) - 1 for s in seqs], device)
    cov = {}
    for k, (i, a) in enumerate(keys):
        rank = G["articles"][str(i)]["values"][a]["rank"]
        cov.setdefault(str(rank), []).append(float(np.mean([covered(r[k], w[k], written) for r, w in rd])))
    out = {"chosen": chosen, "write_seconds": round(t_write, 1), "rows_written": len(written),
           "val_ppl": val_ppl(model, device), "coverage_by_rank": {r: float(np.mean(v)) for r, v in cov.items()},
           "people": {str(i): res[i] for i in ids}}
    writer.restore()
    out["val_ppl_after_restore_dev"] = val_ppl(model, device, DEV_WINDOWS)
    write_json(os.path.join(OUT, "test.json"), out)
    print({k: v for k, v in out.items() if k != "people"})


# ---------------------------------------------------------------------------------------------------- summary
def ci(d):
    return [round(float(np.percentile(d, q)), 5) for q in (2.5, 97.5)]


def summary():
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    N, T = (json.load(open(os.path.join(OUT, f"{n}.json"))) for n in ("null", "test"))
    S = json.load(open(os.path.join(FACTK, "summary.json")))["models"]["B-16M"]
    rng = np.random.default_rng(0)
    lv = {p["id"]: p["k"] for p in P["people"]}
    keep = [i for i in lv if lv[i] in KEEP]

    def per(res, ids):
        return np.array([np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in ids])

    def acc(res, ids):
        return float(np.mean([res[str(i)][a]["hit"] for i in ids for a in ATTRS]))

    # lift: change of each person's score (after minus before, the null test is "before"), written test people
    # against unwritten people. Paired per person because the groups differ by chance before any write (null test)
    wr, un = per(T["people"], G["groups"]["test"]), per(T["people"], G["groups"]["unwritten"])
    dw = wr - per(N["people"], G["groups"]["test"])
    du = un - per(N["people"], G["groups"]["unwritten"])
    lift = dw[rng.integers(0, len(dw), (BOOT, len(dw)))].mean(1) - du[rng.integers(0, len(du), (BOOT, len(du)))].mean(1)
    # what the null test showed: the unpaired comparison of the groups on the unchanged model
    wr0, un0 = per(N["people"], G["groups"]["test"]), per(N["people"], G["groups"]["unwritten"])
    lift0 = (wr0[rng.integers(0, len(wr0), (BOOT, len(wr0)))].mean(1)
             - un0[rng.integers(0, len(un0), (BOOT, len(un0)))].mean(1))
    # side effects, paired over the same people
    kb, ka = per(N["people"], keep), per(T["people"], keep)
    dk = (ka - kb)[rng.integers(0, len(keep), (BOOT, len(keep)))].mean(1)
    dun = du[rng.integers(0, len(du), (BOOT, len(du)))].mean(1)
    ppl_change = T["val_ppl"] / N["val_ppl"] - 1
    accuracy = acc(T["people"], G["groups"]["test"])
    # step-10 repetitions that give B-16M-factk the same mean score (log-linear between levels, no verdict)
    lev = [int(k) for k in S["score"]]
    sc = [S["score"][str(k)] for k in lev]
    m = float(wr.mean())
    reps = "<1" if m < sc[1] else ">64" if m > sc[-1] else next(
        round(2 ** (math.log2(a) + (m - x) / (y - x) * (math.log2(b) - math.log2(a))), 1)
        for a, b, x, y in zip(lev[1:], lev[2:], sc[1:], sc[2:]) if x <= m <= y)
    # by the position of the attribute's sentence in the written article (0: the prompt is a prefix of it)
    rank = {}
    for i in G["groups"]["test"]:
        for a in ATTRS:
            r = G["articles"][str(i)]["values"][a]["rank"]
            rank.setdefault(str(r), []).append(T["people"][str(i)][a]["p"])
    order = G["test_write_order"]
    first, last = per(T["people"], order[:50]), per(T["people"], order[-50:])
    side_ok = ppl_change <= 0.01 and float(dk.mean()) >= -0.02
    sticks = ci(lift)[0] > 0
    verdict = ("doesn't stick" if not sticks else "sticks only weakly" if accuracy < 0.5 else
               "writing facts into the table works" if side_ok else "works, with side effects")
    out = {"null_unpaired_lift": round(float(lift0.mean()), 5), "null_unpaired_lift_ci95": ci(lift0),
           "written_score": round(m, 5), "unwritten_score": round(float(un.mean()), 5),
           "lift": round(float(lift.mean()), 5), "lift_ci95": ci(lift), "written_accuracy": round(accuracy, 4),
           "unwritten_accuracy": round(acc(T["people"], G["groups"]["unwritten"]), 4),
           "written_change": round(float(dw.mean()), 5),
           "unwritten_change": round(float(du.mean()), 5), "unwritten_change_ci95": ci(dun),
           "keep_change": round(float(dk.mean()), 5), "keep_change_ci95": ci(dk),
           "val_ppl_before": round(N["val_ppl"], 4), "val_ppl_after": round(T["val_ppl"], 4),
           "val_ppl_change": round(ppl_change, 5), "side_effects_ok": side_ok,
           "verdict": verdict,
           "same_score_as_training_repetitions": reps,
           "score_by_sentence_rank": {r: round(float(np.mean(v)), 5) for r, v in sorted(rank.items())},
           "first_vs_last_50_written": [round(float(first.mean()), 5), round(float(last.mean()), 5)],
           "chosen": T["chosen"], "rows_written": T["rows_written"], "coverage_by_rank": T["coverage_by_rank"],
           "overlap_between_people": N["overlap_between_people"],
           "overlap_prompt_vs_article": N["overlap_prompt_vs_article"]}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "null", "dev", "test", "summary"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"split": split, "summary": summary}.get(a.stage, lambda: globals()[a.stage](a.device))()
