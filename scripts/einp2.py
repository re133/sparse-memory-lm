"""Step 15 (EINP2): write new facts into the table without pushing out the ones the model knows.

  python scripts/einp2.py split     -> report/einp2/groups.json (150 fresh dev people, known people split; CPU)
  python scripts/einp2.py pre       -> report/einp2/pre.json    (dev people before writing, shared-row check)
  python scripts/einp2.py dev       -> report/einp2/dev.json    (grid per variant, choice under the known budget)
  python scripts/einp2.py test      -> report/einp2/test.json   (chosen setting per variant on the test people)
  python scripts/einp2.py summary   -> report/einp2/summary.json

Step 14 (scripts/einp.py) wrote 150 new people into every table row their articles read and pushed out the people
the model had learned in training. Two variants, otherwise step 14's writing, model, test people and order:
  A  all rows read (step 14's rule), now with the setting chosen under a budget for the known people
  B  only the rows memory layer 10 reads, with the gradient of that read only (layers 2 and 6 read nearly the same
     rows at the cue for every person, layer 10 rows of its own)
The setting of each variant is chosen on 150 fresh made-up people (new names, the same value lists, none of them in
the training text or in step 10's people), so the dev writes are as many as the test writes. The 900 known people
(seen 16, 32 or 64 times) are split, seeded, into 300 for choosing and 600 for the measurement.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import tiktoken
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from einp import (ATTRS, BOOT, DEV_WINDOWS, EOT, KEEP, LRS, STEPS, TableWriter, article, ci, load_model,  # noqa: E402
                  mean_score, persons, prompts, reads, score, val_ppl, write_person)
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402

OUT = os.path.join(ROOT, "report", "einp2")
E14 = os.path.join(ROOT, "report", "einp")
SEED = 20261023
N_DEV, KNOWN_DEV = 150, 300
VARIANTS = {"A": None, "B": [2]}           # memory_layers() indices written: None = all, [2] = layer 10
KNOWN_LIMIT, PPL_LIMIT = -0.02, 0.01


def everyone():
    """step 10's people plus the fresh dev people (ids from 10000), one dict as persons.json."""
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    P["people"] = P["people"] + G["fresh"]
    return P


def split():
    import make_factk as MF
    enc = tiktoken.get_encoding("gpt2")
    P = persons()
    rng = np.random.default_rng(SEED)
    taken = {p["name"] for p in P["people"]}
    names = [n for n in MF.make_names(rng, 3 * N_DEV) if n not in taken]
    forms = {n: (tuple(enc.encode(n)), tuple(enc.encode(" " + n))) for n in names}
    hit = MF.occurs(np.asarray(load_split("train", "wikipedia")), [s for f in forms.values() for s in f])
    names = [n for n in names if not (set(forms[n]) & hit)][:N_DEV]
    assert len(names) == N_DEV
    fresh = [{"id": 10000 + i, "name": n, "k": -1, **{a: str(rng.choice(P["values"][a])) for a in ATTRS}}
             for i, n in enumerate(names)]
    Pf = dict(P, people=fresh)
    arts = {}
    for pr in fresh:
        ids, pos = article(enc, Pf, pr, rng)
        arts[str(pr["id"])] = {"ids": ids, "values": pos}
    known = [int(i) for i in rng.permutation([p["id"] for p in P["people"] if p["k"] in KEEP])]
    out = {"seed": SEED, "fresh": fresh, "dev": [p["id"] for p in fresh], "dev_articles": arts,
           "known_dev": sorted(known[:KNOWN_DEV]), "known_test": sorted(known[KNOWN_DEV:]),
           "names_dropped_in_training_text": len([n for n in forms if set(forms[n]) & hit])}
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "groups.json"), out)
    print(N_DEV, "fresh people,", len(out["known_dev"]), "/", len(out["known_test"]), "known,",
          out["names_dropped_in_training_text"], "names dropped")


@torch.no_grad()
def article_rows(model, arts, layer):
    """Rows a memory layer (index into memory_layers(), None = all) reads at the positions that get a gradient when
    an article is written (every position before its last value token)."""
    mems = model.memory_layers()
    for m in mems:
        m.record = True
    rows = set()
    for a in arts:
        last = max(v["pos"] for v in a["values"].values())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(torch.tensor([a["ids"]], device="cuda"))
        for li, m in enumerate(mems):
            if layer is None or li == layer:
                rows.update(m.last_indices.reshape(len(a["ids"]), -1)[:last].reshape(-1).tolist())
    for m in mems:
        m.record = False
    return rows


def pre(device):
    P, G = everyone(), json.load(open(os.path.join(OUT, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    model = load_model(device)
    t0 = time.time()
    out = {"dev_before": {str(k): v for k, v in score(model, P, G["dev"], device).items()}}
    # shared rows (no writes): how much of the known people's read weight at the cue, per layer, falls on the rows
    # each variant would write for the test people
    test_arts = [G14["articles"][str(i)] for i in G14["groups"]["test"]]
    seqs, _ = prompts(P, G["known_test"])
    rd = reads(model, seqs, [len(s) - 1 for s in seqs], device)
    share = {}
    for v, layers in VARIANTS.items():
        rows = article_rows(model, test_arts, None if layers is None else layers[0])
        share[v] = {"rows": len(rows), "known_read_weight_on_them": [
            float(np.mean([sum(x for r, x in zip(rr.tolist(), ww.tolist()) if r in rows) / sum(ww.tolist())
                           for rr, ww in zip(R, W)])) for R, W in rd]}
    out["shared_rows"] = share
    out["seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(OUT, "pre.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "dev_before"}, indent=1))


def known_change(res, before, ids):
    return float(np.mean([np.mean([res[i][a]["p"] for a in ATTRS])
                          - np.mean([before[str(i)][a]["p"] for a in ATTRS]) for i in ids]))


def dev(device):
    P, G = everyone(), json.load(open(os.path.join(OUT, "groups.json")))
    N = json.load(open(os.path.join(E14, "null.json")))
    model = load_model(device)
    out = {}
    for v, layers in VARIANTS.items():
        writer = TableWriter(model, layers)
        grid = []
        for steps in STEPS:
            for lr in LRS:
                t0 = time.time()
                for i in G["dev"]:
                    write_person(model, writer, G["dev_articles"][str(i)], steps, lr, device)
                rows = int(writer.saved.sum())
                res = score(model, P, G["dev"] + G["known_dev"], device)
                ppl = val_ppl(model, device, DEV_WINDOWS)
                writer.restore()
                grid.append({"steps": steps, "lr": lr, "score": mean_score(res, G["dev"]),
                             "accuracy": float(np.mean([res[i][a]["hit"] for i in G["dev"] for a in ATTRS])),
                             "known_change": known_change(res, N["people"], G["known_dev"]),
                             "val_ppl_change": ppl / N["val_ppl_dev"] - 1, "rows": rows,
                             "seconds": round(time.time() - t0, 1)})
                print(v, grid[-1], flush=True)
        ok = [g for g in grid if g["known_change"] >= KNOWN_LIMIT and g["val_ppl_change"] <= PPL_LIMIT]
        best = (max(ok, key=lambda g: (round(g["score"], 6), -g["steps"], -g["lr"])) if ok else
                max(grid, key=lambda g: (g["known_change"], g["score"])))
        out[v] = {"grid": grid, "chosen": {"steps": best["steps"], "lr": best["lr"]}, "within_budget": bool(ok)}
        print(v, "chosen", best, "within budget" if ok else "NONE within budget", flush=True)
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    out["rule"] = (f"per variant: highest dev score with known_dev change >= {KNOWN_LIMIT} and dev val PPL "
                   f"+{PPL_LIMIT:.0%} or less (ties: fewer steps, then smaller lr); if none: smallest known loss")
    write_json(os.path.join(OUT, "dev.json"), out)


def test(device):
    P, G = everyone(), json.load(open(os.path.join(OUT, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    D = json.load(open(os.path.join(OUT, "dev.json")))
    model = load_model(device)
    ids = G14["groups"]["test"] + G14["groups"]["unwritten"] + G["known_test"]
    out = {}
    for v, layers in VARIANTS.items():
        writer = TableWriter(model, layers)
        c = D[v]["chosen"]
        t0 = time.time()
        for i in G14["test_write_order"]:
            write_person(model, writer, G14["articles"][str(i)], c["steps"], c["lr"], device)
        t_write = time.time() - t0
        res = score(model, P, ids, device)
        out[v] = {"chosen": c, "write_seconds": round(t_write, 1), "rows_written": int(writer.saved.sum()),
                  "val_ppl": val_ppl(model, device), "people": {str(i): res[i] for i in ids}}
        writer.restore()
        print(v, {k: x for k, x in out[v].items() if k != "people"}, flush=True)
    N = json.load(open(os.path.join(E14, "null.json")))
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    write_json(os.path.join(OUT, "test.json"), out)


def summary():
    P, G = persons(), json.load(open(os.path.join(OUT, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    N, T, D = (json.load(open(os.path.join(d, f))) for d, f in ((E14, "null.json"), (OUT, "test.json"),
                                                               (OUT, "dev.json")))
    S = json.load(open(os.path.join(E14, "..", "factk", "summary.json")))["models"]["B-16M"]
    wr_ids, un_ids, kn_ids = G14["groups"]["test"], G14["groups"]["unwritten"], G["known_test"]
    rng = np.random.default_rng(0)
    dw_draw = rng.integers(0, len(wr_ids), (BOOT, len(wr_ids)))
    du_draw = rng.integers(0, len(un_ids), (BOOT, len(un_ids)))
    dk_draw = rng.integers(0, len(kn_ids), (BOOT, len(kn_ids)))

    def per(res, ids):
        return np.array([np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in ids])

    out, boot = {"variants": {}}, {}
    for v in VARIANTS:
        R = T[v]["people"]
        dw, du, dk = (per(R, x) - per(N["people"], x) for x in (wr_ids, un_ids, kn_ids))
        lift = dw[dw_draw].mean(1) - du[du_draw].mean(1)
        known = dk[dk_draw].mean(1)
        boot[v] = (lift, known)
        acc = float(np.mean([R[str(i)][a]["hit"] for i in wr_ids for a in ATTRS]))
        ppl = T[v]["val_ppl"] / N["val_ppl"] - 1
        side_ok = ppl <= PPL_LIMIT and float(dk.mean()) >= KNOWN_LIMIT
        verdict = ("doesn't stick" if ci(lift)[0] <= 0 else "sticks only weakly" if acc < 0.5 else
                   "writes without pushing out" if side_ok else "works, with side effects")
        rank = {}
        for i in wr_ids:
            for a in ATTRS:
                rank.setdefault(str(G14["articles"][str(i)]["values"][a]["rank"]), []).append(R[str(i)][a]["p"])
        lev = [int(k) for k in S["score"]]
        sc = [S["score"][str(k)] for k in lev]
        m = float(per(R, wr_ids).mean())
        reps = "<1" if m < sc[1] else ">64" if m > sc[-1] else next(
            round(2 ** (math.log2(a) + (m - x) / (y - x) * (math.log2(b) - math.log2(a))), 1)
            for a, b, x, y in zip(lev[1:], lev[2:], sc[1:], sc[2:]) if x <= m <= y)
        order = G14["test_write_order"]
        kacc = {k: round(float(np.mean([R[str(i)][a]["hit"] for i in kn_ids if P["people"][i]["k"] == k
                                        for a in ATTRS])), 4) for k in KEEP}
        out["variants"][v] = {
            "chosen": T[v]["chosen"], "chosen_within_dev_budget": D[v]["within_budget"], "verdict": verdict,
            "lift": round(float(lift.mean()), 5), "lift_ci95": ci(lift), "written_score": round(m, 5),
            "written_accuracy": round(acc, 4), "unwritten_change": round(float(du.mean()), 5),
            "known_change": round(float(dk.mean()), 5), "known_change_ci95": ci(known), "known_accuracy": kacc,
            "val_ppl_change": round(ppl, 5), "score_by_sentence_rank": {r: round(float(np.mean(x)), 5)
                                                                       for r, x in sorted(rank.items())},
            "first_vs_last_50_written": [round(float(per(R, order[:50]).mean()), 5),
                                         round(float(per(R, order[-50:]).mean()), 5)],
            "same_score_as_training_repetitions": reps, "rows_written": T[v]["rows_written"],
            "seconds_per_person": round(T[v]["write_seconds"] / len(wr_ids), 3)}
    d = boot["B"][0] - boot["A"][0]
    out["B_minus_A_lift"] = {"mean": round(float(d.mean()), 5), "ci95": ci(d),
                             "verdict": ("layer 10 writes more within the same budget" if ci(d)[0] > 0 else
                                         "all rows write more within the same budget" if ci(d)[1] < 0 else
                                         "no clear difference")}
    dk = boot["B"][1] - boot["A"][1]
    out["B_minus_A_known_change"] = {"mean": round(float(dk.mean()), 5), "ci95": ci(dk)}
    out["dev_grid"] = {v: D[v]["grid"] for v in VARIANTS}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: x for k, x in out.items() if k != "dev_grid"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "pre", "dev", "test", "summary"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"split": split, "summary": summary}.get(a.stage, lambda: globals()[a.stage](a.device))()
