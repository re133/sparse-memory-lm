"""Step 16 (EINPD): write new facts into the table more gently, so the known ones stay within the budget.

  python scripts/einpd.py dev       -> report/einpd/dev.json    (grid per variant, choice under the known budget)
  python scripts/einpd.py test      -> report/einpd/test.json   (chosen setting per variant on the test people)
  python scripts/einpd.py summary   -> report/einpd/summary.json

Step 15 (scripts/einp2.py): all rows read, 16 passes with lr 0.1, wrote 99% of the new people right but the known
people lost 0.066 (dev) / 0.063 (test) of score, over the −0.02 budget; a tenth of step 14's step size had cut the
loss tenfold. Two variants, everything else as in step 15 (model, the 150 fresh dev people, the known split, the 150
test people and their order, the budget and the rule that chooses the setting):
  A  all rows read, smaller steps and more passes: lr {0.01, 0.03, 0.1} x passes {16, 64}
  S  only the `top` rows of each pass with the highest tf * idf (sparse memory finetuning, Lin et al. 2025):
     tf = reads of the row in this pass before the last value token, idf = log(total / (1 + background)) with the
     background = the row's reads in B-16M-factk's training (mem_access_train.npy). top {1000, 3000} (of ~11,200 rows
     a pass reads) x lr {0.1, 1} x passes {16, 64}
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
from einp import (ATTRS, BOOT, DEV_WINDOWS, KEEP, RUN, TableWriter, ci, load_model, mean_score, persons,  # noqa: E402
                  prompts, reads, score, val_ppl, write_person)
from einp2 import KNOWN_LIMIT, PPL_LIMIT, everyone, known_change  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

OUT = os.path.join(ROOT, "report", "einpd")
E14, E15 = (os.path.join(ROOT, "report", d) for d in ("einp", "einp2"))
GRIDS = {"A": [{"top": None, "lr": lr, "steps": st} for st in (16, 64) for lr in (0.01, 0.03, 0.1)],
         "S": [{"top": t, "lr": lr, "steps": st} for st in (16, 64) for t in (1000, 3000) for lr in (0.1, 1.0)]}


def idf():
    bg = torch.from_numpy(np.load(os.path.join(ROOT, RUN, "mem_access_train.npy")).astype(np.float64))
    return torch.log(bg.sum() / (1 + bg)).float()


def choose(grid):
    ok = [g for g in grid if g["known_change"] >= KNOWN_LIMIT and g["val_ppl_change"] <= PPL_LIMIT]
    best = (max(ok, key=lambda g: (round(g["score"], 6), -g["steps"], -g["lr"], -(g["top"] or 10 ** 9))) if ok else
            max(grid, key=lambda g: (g["known_change"], g["score"])))
    return {k: best[k] for k in ("top", "lr", "steps")}, bool(ok)


def dev(device):
    P, G = everyone(), json.load(open(os.path.join(E15, "groups.json")))
    N = json.load(open(os.path.join(E14, "null.json")))
    model = load_model(device)
    weights = idf()
    out = {}
    for v, grid in GRIDS.items():
        rows_out = []
        for c in grid:
            writer = TableWriter(model, top=c["top"], idf=weights if c["top"] else None)
            t0 = time.time()
            for i in G["dev"]:
                write_person(model, writer, G["dev_articles"][str(i)], c["steps"], c["lr"], device)
            rows = int(writer.saved.sum())
            res = score(model, P, G["dev"] + G["known_dev"], device)
            ppl = val_ppl(model, device, DEV_WINDOWS)
            writer.restore()
            rows_out.append({**c, "score": mean_score(res, G["dev"]),
                             "accuracy": float(np.mean([res[i][a]["hit"] for i in G["dev"] for a in ATTRS])),
                             "known_change": known_change(res, N["people"], G["known_dev"]),
                             "val_ppl_change": ppl / N["val_ppl_dev"] - 1, "rows": rows,
                             "seconds": round(time.time() - t0, 1)})
            print(v, rows_out[-1], flush=True)
        chosen, within = choose(rows_out)
        out[v] = {"grid": rows_out, "chosen": chosen, "within_budget": within}
        print(v, "chosen", chosen, "within budget" if within else "NONE within budget", flush=True)
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    out["rule"] = (f"per variant: highest dev score with known_dev change >= {KNOWN_LIMIT} and dev val PPL "
                   f"+{PPL_LIMIT:.0%} or less (ties: fewer passes, smaller lr, smaller top); if none: smallest loss")
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "dev.json"), out)


def test(device):
    P, G = everyone(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    D = json.load(open(os.path.join(OUT, "dev.json")))
    N = json.load(open(os.path.join(E14, "null.json")))
    model = load_model(device)
    weights = idf()
    ids = G14["groups"]["test"] + G14["groups"]["unwritten"] + G["known_test"]
    seqs, _ = prompts(P, G["known_test"])
    known_reads = reads(model, seqs, [len(s) - 1 for s in seqs], device)       # unchanged table
    out = {}
    for v in GRIDS:
        c = D[v]["chosen"]
        writer = TableWriter(model, top=c["top"], idf=weights if c["top"] else None)
        t0 = time.time()
        for i in G14["test_write_order"]:
            write_person(model, writer, G14["articles"][str(i)], c["steps"], c["lr"], device)
        t_write = time.time() - t0
        written = writer.saved
        on_written = [float(np.mean([float(w[k][written[r[k]]].sum() / w[k].sum()) for k in range(len(r))]))
                      for r, w in known_reads]
        res = score(model, P, ids, device)
        out[v] = {"chosen": c, "write_seconds": round(t_write, 1), "rows_written": int(written.sum()),
                  "known_read_weight_on_written_rows": on_written, "val_ppl": val_ppl(model, device),
                  "people": {str(i): res[i] for i in ids}}
        writer.restore()
        print(v, {k: x for k, x in out[v].items() if k != "people"}, flush=True)
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    write_json(os.path.join(OUT, "test.json"), out)


def summary():
    P, G = persons(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    N, T, D = (json.load(open(os.path.join(d, f))) for d, f in ((E14, "null.json"), (OUT, "test.json"),
                                                               (OUT, "dev.json")))
    S = json.load(open(os.path.join(ROOT, "report", "factk", "summary.json")))["models"]["B-16M"]
    wr_ids, un_ids, kn_ids = G14["groups"]["test"], G14["groups"]["unwritten"], G["known_test"]
    rng = np.random.default_rng(0)
    dw_draw = rng.integers(0, len(wr_ids), (BOOT, len(wr_ids)))
    du_draw = rng.integers(0, len(un_ids), (BOOT, len(un_ids)))
    dk_draw = rng.integers(0, len(kn_ids), (BOOT, len(kn_ids)))

    def per(res, ids):
        return np.array([np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in ids])

    out, boot = {"variants": {}}, {}
    for v in GRIDS:
        R = T[v]["people"]
        dw, du, dk = (per(R, x) - per(N["people"], x) for x in (wr_ids, un_ids, kn_ids))
        lift = dw[dw_draw].mean(1) - du[du_draw].mean(1)
        boot[v] = lift
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
            "known_change": round(float(dk.mean()), 5), "known_change_ci95": ci(dk[dk_draw].mean(1)),
            "known_accuracy": kacc, "val_ppl_change": round(ppl, 5),
            "score_by_sentence_rank": {r: round(float(np.mean(x)), 5) for r, x in sorted(rank.items())},
            "first_vs_last_50_written": [round(float(per(R, order[:50]).mean()), 5),
                                         round(float(per(R, order[-50:]).mean()), 5)],
            "same_score_as_training_repetitions": reps, "rows_written": T[v]["rows_written"],
            "known_read_weight_on_written_rows": T[v]["known_read_weight_on_written_rows"],
            "seconds_per_person": round(T[v]["write_seconds"] / len(wr_ids), 3)}
    if D["A"]["within_budget"] and D["S"]["within_budget"]:
        d = boot["S"] - boot["A"]
        out["S_minus_A_lift"] = {"mean": round(float(d.mean()), 5), "ci95": ci(d), "verdict": (
            "the specific rows write more within the budget" if ci(d)[0] > 0 else
            "all rows write more within the budget" if ci(d)[1] < 0 else "no clear difference")}
    else:
        out["S_minus_A_lift"] = {"verdict": "not comparable: not both within the dev budget"}
    out["dev_grid"] = {v: D[v]["grid"] for v in GRIDS}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: x for k, x in out.items() if k != "dev_grid"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "test", "summary"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    summary() if a.stage == "summary" else globals()[a.stage](a.device)
