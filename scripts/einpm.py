"""Step 24 (EINPM): the model learns to be written into, in a short fine-tuning of its core with write episodes.

  python scripts/einpm.py people    -> report/einpm/people.json   (600 fresh made-up people and their texts; CPU)
  python scripts/einpm.py train     -> runs/einpm/core.pt, report/einpm/train.json
  python scripts/einpm.py null      -> report/einpm/null.json     (the fine-tuned model before any write)
  python scripts/einpm.py test      -> report/einpm/test.json     (step 22's writes into the fine-tuned model)
  python scripts/einpm.py summary   -> report/einpm/summary.json
  python scripts/einpm.py STAGE --variant einpm2   step 25: the same, gentler (report/einpm2/, runs/einpm2/)

Step 22 wrote each test person from one sentence in another wording and asked in the usual one: 49.6% right, and the
people the model knew lost 0.095 (step 16, written word for word: 0.035). Here the core (everything but the table) of
B-16M-factk is fine-tuned in episodes. One episode: write 4 fresh people, each from one sentence in a wording that
step 22 doesn't use (T4 to T6), with step 22's write (64 steps of 0.01 on the rows read); then one AdamW step on the
core with three losses on the written table: the usual questions about the 4 new people, the usual questions about 8
known people (step 15's known_dev, not used in the test), and one window of training text; then the table is put back.
First order: the written rows count as constants, the core learns to read them (and the known facts next to them),
not where to write. 150 episodes, every fresh person once. Then step 22's test exactly, on the fine-tuned model.
Step 25 (einpm2): the same people and episodes with lr 1e-5, 32 known people and 2 text windows per step; its summary
measures the damage against the original model's scores before writing and checks the known people before any write.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from einp import (ATTRS, BOOT, DEV_WINDOWS, EOT, SEQ, TableWriter, ci, load_model, prompts, score,  # noqa: E402
                  val_ppl, write_person)
from einp2 import everyone  # noqa: E402
from einp3 import own_wording  # noqa: E402
from smlm.atomic import write_json  # noqa: E402
from smlm.data import load_split  # noqa: E402

OUT = os.path.join(ROOT, "report", "einpm")
RUN_OUT = os.path.join(ROOT, "runs", "einpm")
PEOPLE = os.path.join(ROOT, "report", "einpm", "people.json")      # shared by both variants
E14, E15, E22 = (os.path.join(ROOT, "report", d) for d in ("einp", "einp2", "einp3"))
SEED = 20261025
N_FRESH, PER_EPISODE, KNOWN_PER_EPISODE, TEXT_WINDOWS = 600, 4, 8, 1
LR, CLIP = 5e-5, 1.0
VARIANTS = {"einpm": dict(LR=5e-5, KNOWN_PER_EPISODE=8, TEXT_WINDOWS=1),          # step 24
            "einpm2": dict(LR=1e-5, KNOWN_PER_EPISODE=32, TEXT_WINDOWS=2)}        # step 25, gentler
KNOWN_FLOOR = -0.035                        # step 25: the known people before any write may lose at most this (step 16)
TEMPLATES = {
    "T4": "{name}, who came into the world in {year}, spent a childhood in {city} and earned a living as a {job}.",
    "T5": "Records list {name} as a {job} from {city}, with a birth year of {year}.",
    "T6": "As a child, {name} (born {year}) lived in {city}; the adult {name} was employed as a {job}.",
}


def text_ids(enc, pr, template):
    """Token ids of a person's text and, per attribute, the position of its value token (as einp3.paraphrase)."""
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


def people():
    import make_factk as MF
    enc = tiktoken.get_encoding("gpt2")
    P = everyone()
    rng = np.random.default_rng(SEED)
    taken = {p["name"] for p in P["people"]}
    names = [n for n in MF.make_names(rng, 3 * N_FRESH) if n not in taken]
    forms = {n: (tuple(enc.encode(n)), tuple(enc.encode(" " + n))) for n in names}
    hit = MF.occurs(np.asarray(load_split("train", "wikipedia")), [s for f in forms.values() for s in f])
    names = [n for n in names if not (set(forms[n]) & hit)][:N_FRESH]
    assert len(names) == N_FRESH
    fresh = [{"id": 20000 + i, "name": n, "k": -2, **{a: str(rng.choice(P["values"][a])) for a in ATTRS}}
             for i, n in enumerate(names)]
    texts = {str(pr["id"]): text_ids(enc, pr, str(rng.choice(sorted(TEMPLATES)))) for pr in fresh}
    order = [int(i) for i in rng.permutation([p["id"] for p in fresh])]
    os.makedirs(OUT, exist_ok=True)
    write_json(PEOPLE, {"seed": SEED, "templates": TEMPLATES, "fresh": fresh,
                                                   "texts": texts, "order": order})
    print(N_FRESH, "fresh people", {t: sum(x["template"] == t for x in texts.values()) for t in TEMPLATES})


def everyone_m():
    """Everyone of steps 10 to 22 plus this step's fresh people (ids from 20000)."""
    P = everyone()
    P["people"] = P["people"] + json.load(open(PEOPLE))["fresh"]
    return P


def core_params(model):
    table = model.memory_layers()[0].values.weight
    return [(n, p) for n, p in model.named_parameters() if p is not table]


def load_tuned(device):
    model = load_model(device)
    sd = torch.load(os.path.join(RUN_OUT, "core.pt"), map_location="cpu", weights_only=True)
    with torch.no_grad():
        for n, p in core_params(model):
            p.copy_(sd[n].to(p.device))
    return model


def question_loss(model, P, ids, device):
    """Cross entropy of the true value token after the usual question (`Name was born in` ...), over the vocabulary."""
    enc = tiktoken.get_encoding("gpt2")
    by_id = {p["id"]: p for p in P["people"]}
    seqs, keys = prompts(P, ids)
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, :len(s)] = torch.tensor(s)
    tgt = torch.tensor([enc.encode(" " + by_id[i][a])[0] for i, a in keys], device=device)
    last = torch.tensor([len(s) - 1 for s in seqs], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(x.to(device))
    return F.cross_entropy(logits[torch.arange(len(seqs), device=device), last].float(), tgt)


def train(device):
    P, G = everyone_m(), json.load(open(os.path.join(E15, "groups.json")))
    M = json.load(open(PEOPLE))
    c = json.load(open(os.path.join(E22, "dev.json")))["chosen"]
    model = load_model(device)
    core = [p for _, p in core_params(model)]
    opt = torch.optim.AdamW(core, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    text = np.asarray(load_split("train", "wikipedia_factk"))
    rng = np.random.default_rng(SEED + 1)
    log, t0 = [], time.time()
    for e in range(N_FRESH // PER_EPISODE):
        new = M["order"][e * PER_EPISODE:(e + 1) * PER_EPISODE]
        known = [int(i) for i in rng.choice(G["known_dev"], KNOWN_PER_EPISODE, replace=False)]
        starts = [int(rng.integers(0, len(text) - SEQ - 1)) for _ in range(TEXT_WINDOWS)]
        windows = torch.stack([torch.from_numpy(text[a:a + SEQ + 1].astype(np.int64)) for a in starts]).to(device)
        for p in core:
            p.requires_grad_(False)
        writer, tw = TableWriter(model), time.time()
        for i in new:
            write_person(model, writer, M["texts"][str(i)], c["steps"], c["lr"], device)
        tw = time.time() - tw
        for p in core:
            p.requires_grad_(True)
        l_new = question_loss(model, P, new, device)
        l_known = question_loss(model, P, known, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(windows[:, :-1])
        l_text = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), windows[:, 1:].reshape(-1))
        (l_new + l_known + l_text).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(core, CLIP))
        opt.step()
        opt.zero_grad(set_to_none=True)
        writer.restore()
        log.append({"episode": e + 1, "new": float(l_new), "known": float(l_known), "text": float(l_text),
                    "grad_norm": gn, "write_s": round(tw, 2)})
        if (e + 1) % 10 == 0:
            last = log[-10:]
            print({k: round(float(np.mean([r[k] for r in last])), 4) for k in ("new", "known", "text", "grad_norm")},
                  f"episode {e + 1}, {time.time() - t0:.0f} s", flush=True)
    for p in core:
        p.requires_grad_(False)
    os.makedirs(RUN_OUT, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in core_params(model)}, os.path.join(RUN_OUT, "core.pt"))
    write_json(os.path.join(OUT, "train.json"), {"lr": LR, "clip": CLIP, "episodes": len(log),
                                                  "text_windows": TEXT_WINDOWS,
                                                  "people_per_episode": PER_EPISODE,
                                                  "known_per_episode": KNOWN_PER_EPISODE, "write": c,
                                                  "seconds": round(time.time() - t0, 1), "log": log})


def null(device):
    P, G = everyone_m(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    model = load_tuned(device)
    ids = G14["groups"]["test"] + G14["groups"]["unwritten"] + G["known_test"] + G["known_dev"]
    res = score(model, P, ids, device)
    write_json(os.path.join(OUT, "null.json"), {"people": {str(i): res[i] for i in ids},
                                                 "val_ppl": val_ppl(model, device),
                                                 "val_ppl_dev": val_ppl(model, device, DEV_WINDOWS)})


def test(device):
    """einp3.test on the fine-tuned model: step 22's texts, setting, people and order."""
    P, G = everyone_m(), json.load(open(os.path.join(E15, "groups.json")))
    G14 = json.load(open(os.path.join(E14, "groups.json")))
    A = json.load(open(os.path.join(E22, "articles.json")))["articles"]
    c = json.load(open(os.path.join(E22, "dev.json")))["chosen"]
    N = json.load(open(os.path.join(OUT, "null.json")))
    model = load_tuned(device)
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
    G, G14 = (json.load(open(os.path.join(d, "groups.json"))) for d in (E15, E14))
    A = json.load(open(os.path.join(E22, "articles.json")))["articles"]
    N, T = (json.load(open(os.path.join(OUT, f))) for f in ("null.json", "test.json"))
    N0, T0 = json.load(open(os.path.join(E14, "null.json"))), json.load(open(os.path.join(E22, "test.json")))
    wr, un, kn = G14["groups"]["test"], G14["groups"]["unwritten"], G["known_test"]
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(x), (BOOT, len(x))) for x in (wr, un, kn)]

    def per(res, ids):
        return np.array([np.mean([res[str(i)][a]["p"] for a in ATTRS]) for i in ids])

    def hits(res, ids):
        return np.array([np.mean([res[str(i)][a]["hit"] for a in ATTRS]) for i in ids])

    R = T["people"]
    dw, du, dk = (per(R, x) - per(N["people"], x) for x in (wr, un, kn))
    dk0 = per(T0["people"], kn) - per(N0["people"], kn)
    lift = dw[draws[0]].mean(1) - du[draws[1]].mean(1)
    known_diff = (dk - dk0)[draws[2]].mean(1)                  # paired: the same known people in both
    acc_diff = (hits(R, wr) - hits(T0["people"], wr))[draws[0]].mean(1)
    ppl_change = N["val_ppl"] / N0["val_ppl"] - 1

    def three(d, better, worse, same):
        lo, hi = ci(d)
        return better if lo > 0 else worse if hi < 0 else same
    damage = three(known_diff, "less damage", "more damage", "no change in damage")
    transfer = three(acc_diff, "carries over better", "carries over worse", "no change in carrying over")
    text = "text unchanged" if abs(ppl_change) <= 0.01 else "text worse" if ppl_change > 0 else "text better"
    gains = (damage == "less damage") + (transfer == "carries over better")
    losses = (damage == "more damage") + (transfer == "carries over worse") + (text == "text worse")
    verdict = ("the model learned to be written into" if gains and not losses else
               "a trade-off" if gains else "not learned")
    extra = {}
    if OUT.endswith("einpm2"):
        # step 25: damage against the original model before writing (tuned after writes minus step 22 after writes,
        # paired), and the known people of the fine-tuned model before any write against the original's
        vs_base = (per(R, kn) - per(T0["people"], kn))[draws[2]].mean(1)
        before = per(N["people"], kn) - per(N0["people"], kn)
        damage = three(vs_base, "less damage", "more damage", "no change in damage")
        kept = "keeps what it knew" if before.mean() >= KNOWN_FLOOR else "forgets what it knew"
        gains = (damage == "less damage") + (transfer == "carries over better")
        losses = ((damage == "more damage") + (transfer == "carries over worse") + (text == "text worse")
                  + (kept == "forgets what it knew"))
        verdict = ("the model learned to be written into" if gains and not losses else
                   "a trade-off" if gains else "not learned")
        extra = {"known": kept, "known_before_change": round(float(before.mean()), 5),
                 "known_before_change_ci95": ci(before[draws[2]].mean(1)),
                 "known_after_vs_step22": round(float(vs_base.mean()), 5), "known_after_vs_step22_ci95": ci(vs_base),
                 "known_after_hits": round(float(hits(R, kn).mean()), 4),
                 "known_after_hits_step22": round(float(hits(T0["people"], kn).mean()), 4),
                 "known_before_hits": round(float(hits(N["people"], kn).mean()), 4),
                 "known_before_hits_base": round(float(hits(N0["people"], kn).mean()), 4)}
    by_t = {t: [i for i in wr if A[str(i)]["template"] == t] for t in sorted({a["template"] for a in A.values()})}
    own = T["own_wording"]
    out = {"verdict": verdict, "damage": damage, "transfer": transfer, "text": text, **extra,
           "known_change": round(float(dk.mean()), 5), "known_change_ci95": ci(dk[draws[2]].mean(1)),
           "known_change_step22": round(float(dk0.mean()), 5),
           "known_change_diff": round(float(known_diff.mean()), 5), "known_change_diff_ci95": ci(known_diff),
           "written_accuracy": round(float(hits(R, wr).mean()), 4),
           "written_accuracy_step22": round(float(hits(T0["people"], wr).mean()), 4),
           "written_accuracy_diff_ci95": ci(acc_diff),
           "accuracy_by_template": {t: round(float(hits(R, ids).mean()), 4) for t, ids in by_t.items()},
           "own_wording_accuracy": round(float(np.mean([own[str(i)][a]["hit"] for i in wr for a in ATTRS])), 4),
           "lift": round(float(lift.mean()), 5), "lift_ci95": ci(lift),
           "unwritten_change": round(float(du.mean()), 5),
           "val_ppl_tuned": N["val_ppl"], "val_ppl_base": N0["val_ppl"], "val_ppl_change_tuning": round(ppl_change, 5),
           "val_ppl_change_writes": round(T["val_ppl"] / N["val_ppl"] - 1, 5),
           "known_test_before_writes": {"tuned": round(float(per(N["people"], kn).mean()), 5),
                                        "base": round(float(per(N0["people"], kn).mean()), 5)},
           "test_people_before_writes": {"tuned": round(float(per(N["people"], wr).mean()), 5),
                                         "base": round(float(per(N0["people"], wr).mean()), 5)},
           "seconds_per_person": round(T["write_seconds"] / len(wr), 3), "rows_written": T["rows_written"],
           "train": {k: v for k, v in json.load(open(os.path.join(OUT, "train.json"))).items() if k != "log"}}
    write_json(os.path.join(OUT, "summary.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["people", "train", "null", "test", "summary"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="einpm")
    a = ap.parse_args()
    globals().update(VARIANTS[a.variant])
    OUT, RUN_OUT = (os.path.join(ROOT, d, a.variant) for d in ("report", "runs"))
    people() if a.stage == "people" else summary() if a.stage == "summary" else globals()[a.stage](a.device)
