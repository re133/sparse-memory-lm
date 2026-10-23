"""EINP3: write facts into the table from differently worded text, then ask in the usual wording.

  python scripts/einp3.py arts      -> report/einp3/articles.json (the paraphrases; CPU)
  python scripts/einp3.py dev       -> report/einp3/dev.json      (three step sizes on the dev people)
  python scripts/einp3.py test      -> report/einp3/test.json     (the chosen setting on the test people)
  python scripts/einp3.py summary   -> report/einp3/summary.json

Steps 14 to 16 wrote each new person's article in the training format ("Name was born in 1912. Name grew up in
Lisbon. ...") and asked with the start of one of those sentences, so the question was part of the written text. Here
each person is written from one sentence in another wording, a seeded choice of
  T1  "Name (born 1912) spent the early years in Lisbon and later became a tailor."
  T2  "Born in 1912, Name was raised in Lisbon and made a career as a tailor."
  T3  "The tailor Name came from Lisbon, and the records give 1912 as the year of birth."
after the training format's header `<|endoftext|>Name\\n\\n`, with the loss on the three value tokens only. Then the
usual questions (`Name was born in`, `grew up in`, `worked as a`), scored as in step 10. Everything else as in step 16,
variant A: all rows read, model B-16M-factk with its table in RAM, step 15's 150 fresh dev people and known split,
step 14's 150 test people, their order and the 100 unwritten people.
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
from einp import ATTRS, BOOT, DEV_WINDOWS, EOT, TableWriter, ci, load_model, mean_score, score, val_ppl  # noqa: E402
from einp import write_person  # noqa: E402
from einp2 import everyone, known_change  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

OUT = os.path.join(ROOT, "report", "einp3")
E14, E15 = (os.path.join(ROOT, "report", d) for d in ("einp", "einp2"))
SEED = 20261024
TEMPLATES = {
    "T1": "{name} (born {year}) spent the early years in {city} and later became a {job}.",
    "T2": "Born in {year}, {name} was raised in {city} and made a career as a {job}.",
    "T3": "The {job} {name} came from {city}, and the records give {year} as the year of birth.",
}
GRID = [{"lr": lr, "steps": 64} for lr in (0.01, 0.03, 0.1)]
KNOWN_LIMIT = -0.05                         # step 16's floor was about -0.035 at the verbatim writes


def paraphrase(enc, pr, template):
    """Token ids of the written text and, per attribute, the position of its value token (prefix encodings, so the
    BPE merges are those of the whole text, as in einp.article)."""
    text = TEMPLATES[template].format(**{a: pr[a] for a in ATTRS}, name=pr["name"])
    ids = [EOT] + enc.encode(f"{pr['name']}\n\n{text}")
    values = {}
    for a in ATTRS:
        cut = text.index(" " + pr[a])
        pre = [EOT] + enc.encode(f"{pr['name']}\n\n{text[:cut]}")
        val = enc.encode(" " + pr[a])
        assert len(val) == 1 and ids[:len(pre)] == pre and ids[len(pre)] == val[0], (pr["name"], a, template)
        values[a] = {"pos": len(pre), "template": template}
    return {"ids": ids, "values": values, "template": template}


def arts():
    P, enc = everyone(), tiktoken.get_encoding("gpt2")
    G, G14 = json.load(open(os.path.join(E15, "groups.json"))), json.load(open(os.path.join(E14, "groups.json")))
    by_id = {p["id"]: p for p in P["people"]}
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "templates": TEMPLATES, "articles": {}}
    for i in G["dev"] + G14["groups"]["test"]:
        t = str(rng.choice(sorted(TEMPLATES)))
        out["articles"][str(i)] = paraphrase(enc, by_id[i], t)
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "articles.json"), out)
    print({t: sum(a["template"] == t for a in out["articles"].values()) for t in TEMPLATES})


@torch.no_grad()
def own_wording(model, A, ids, device):
    """How well each written value comes back after the start of its own written text (normalised over the values
    of the attribute, as in step 10)."""
    from einp import persons
    P = persons()
    enc = tiktoken.get_encoding("gpt2")
    cand = {a: torch.tensor([enc.encode(" " + v)[0] for v in P["values"][a]]) for a in ATTRS}
    items = [(i, a, A[str(i)]["ids"][:A[str(i)]["values"][a]["pos"]]) for i in ids for a in ATTRS]
    out = {}
    for b in range(0, len(items), 64):
        part = items[b:b + 64]
        L = max(len(s) for _, _, s in part)
        x = torch.zeros(len(part), L, dtype=torch.long)
        for r, (_, _, s) in enumerate(part):
            x[r, :len(s)] = torch.tensor(s)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x.to(device))
        for r, (i, a, s) in enumerate(part):
            lp = torch.log_softmax(logits[r, len(s) - 1].float().cpu()[cand[a]], -1)
            true = P["values"][a].index(next(p for p in P["people"] if p["id"] == i)[a])
            out.setdefault(str(i), {})[a] = {"p": float(lp[true].exp()), "hit": bool(int(lp.argmax()) == true)}
    return out


def dev(device):
    P, G = everyone(), json.load(open(os.path.join(E15, "groups.json")))
    A = json.load(open(os.path.join(OUT, "articles.json")))["articles"]
    N = json.load(open(os.path.join(E14, "null.json")))
    model = load_model(device)
    grid = []
    for c in GRID:
        writer = TableWriter(model)
        t0 = time.time()
        for i in G["dev"]:
            write_person(model, writer, A[str(i)], c["steps"], c["lr"], device)
        res = score(model, P, G["dev"] + G["known_dev"], device)
        ppl = val_ppl(model, device, DEV_WINDOWS)
        writer.restore()
        grid.append({**c, "score": mean_score(res, G["dev"]),
                     "accuracy": float(np.mean([res[i][a]["hit"] for i in G["dev"] for a in ATTRS])),
                     "known_change": known_change(res, N["people"], G["known_dev"]),
                     "val_ppl_change": ppl / N["val_ppl_dev"] - 1, "seconds": round(time.time() - t0, 1)})
        print(grid[-1], flush=True)
    ok = [g for g in grid if g["known_change"] >= KNOWN_LIMIT]
    best = max(ok, key=lambda g: (round(g["score"], 6), -g["lr"])) if ok else max(grid, key=lambda g: g["known_change"])
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    write_json(os.path.join(OUT, "dev.json"), {"grid": grid, "chosen": {"lr": best["lr"], "steps": best["steps"]},
                                               "within_budget": bool(ok)})
    print("chosen", best, flush=True)


def test(device):
    P, G = everyone(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    A = json.load(open(os.path.join(OUT, "articles.json")))["articles"]
    c = json.load(open(os.path.join(OUT, "dev.json")))["chosen"]
    N = json.load(open(os.path.join(E14, "null.json")))
    model = load_model(device)
    ids = G14["groups"]["test"] + G14["groups"]["unwritten"] + G["known_test"]
    writer = TableWriter(model)
    t0 = time.time()
    for i in G14["test_write_order"]:
        write_person(model, writer, A[str(i)], c["steps"], c["lr"], device)
    t_write = time.time() - t0
    res = score(model, P, ids, device)
    own = own_wording(model, A, G14["groups"]["test"], device)
    out = {"chosen": c, "write_seconds": round(t_write, 1), "rows_written": int(writer.saved.sum()),
           "val_ppl": val_ppl(model, device), "people": {str(i): res[i] for i in ids}, "own_wording": own}
    writer.restore()
    assert abs(val_ppl(model, device, DEV_WINDOWS) / N["val_ppl_dev"] - 1) < 1e-3, "restore must give back the table"
    write_json(os.path.join(OUT, "test.json"), out)
    print({k: v for k, v in out.items() if k not in ("people", "own_wording")}, flush=True)


def summary():
    P, G = everyone(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    A = json.load(open(os.path.join(OUT, "articles.json")))["articles"]
    N, T = (json.load(open(f)) for f in (os.path.join(E14, "null.json"), os.path.join(OUT, "test.json")))
    wr, un, kn = G14["groups"]["test"], G14["groups"]["unwritten"], G["known_test"]
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(x), (BOOT, len(x))) for x in (wr, un, kn)]

    def per(res, ids):
        return np.array([np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in ids])

    R = T["people"]
    dw, du, dk = (per(R, x) - per(N["people"], x) for x in (wr, un, kn))
    lift = dw[draws[0]].mean(1) - du[draws[1]].mean(1)
    acc = float(np.mean([R[str(i)][a]["hit"] for i in wr for a in ATTRS]))
    verdict = ("doesn't carry over" if ci(lift)[0] <= 0 else "carries over only weakly" if acc < 0.5 else
               "carries over to the usual question")
    own = T["own_wording"]
    by_t = {t: [i for i in wr if A[str(i)]["template"] == t] for t in TEMPLATES}
    out = {"chosen": T["chosen"], "verdict": verdict, "lift": round(float(lift.mean()), 5), "lift_ci95": ci(lift),
           "written_accuracy": round(acc, 4), "written_score": round(float(per(R, wr).mean()), 5),
           "accuracy_by_attribute": {a: round(float(np.mean([R[str(i)][a]["hit"] for i in wr])), 4) for a in ATTRS},
           "accuracy_by_template": {t: round(float(np.mean([R[str(i)][a]["hit"] for i in ids for a in ATTRS])), 4)
                                    for t, ids in by_t.items()},
           "own_wording_accuracy": round(float(np.mean([own[str(i)][a]["hit"] for i in wr for a in ATTRS])), 4),
           "unwritten_change": round(float(du.mean()), 5), "known_change": round(float(dk.mean()), 5),
           "known_change_ci95": ci(dk[draws[2]].mean(1)), "val_ppl_change": round(T["val_ppl"] / N["val_ppl"] - 1, 5),
           "seconds_per_person": round(T["write_seconds"] / len(wr), 3), "rows_written": T["rows_written"],
           "dev": json.load(open(os.path.join(OUT, "dev.json")))}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "dev"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["arts", "dev", "test", "summary"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    arts() if a.stage == "arts" else summary() if a.stage == "summary" else globals()[a.stage](a.device)
