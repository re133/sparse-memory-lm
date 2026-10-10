"""HOP2 (REPORT.md, "Step 11"): Wikipedia training data with made-up two-hop facts.

  python scripts/make_hop2.py     -> data/wikipedia_hop2_gpt2/{train,validation}.bin + meta.json,
                                     report/hop2/pairs.json (who, which values, which windows and steps)

Each pair is a subject A and a bridge person B ("A's mother was B"). B has three attributes as in FACTK (birth year,
city, profession, each a single GPT-2 token). The question is whether a model answers "A's mother was born in" with
B's year when it learned "A's mother was B" and B's attributes from separate articles. Four conditions, 200 pairs
each, every article 64 times (FACTK: every model knows a fact seen 64 times):

  H2      A's article "A\\n\\nA's mother was B."  and  B's article "B\\n\\nB was born in 1912. B grew up in ... ."
          (separate windows: they never meet in one context)
  SAME    one article with both: "A\\n\\nA's mother was B. B was born in 1912. ..."
  DIRECT  the composed sentences themselves: "A\\n\\nA's mother was born in 1912. A's mother grew up in ..."
          (the question format can be learned at all; plain memorization)
  NULL    never in training (the model's prior for the question, like k = 0 in FACTK)

Every article starts one training window (offset 0), one article per window, only windows that training reads (data
seed 1234), at random steps over the whole run. Attribute sentences in random order. Length and window grid of
wikipedia_en_gpt2 stay; the validation split is copied byte for byte. Values and name syllables come from
make_factk.py (not changed); names are new (not in the training text, not FACTK names), A and B have different last
names.
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_fact_cloze_lm import seen_step  # noqa: E402
from make_factk import CITIES, CUES, EOT, JOBS, SEQ, make_names, occurs, single_tokens  # noqa: E402
from smlm.data import DATA_ROOT, DATASETS, TrainStream, load_split  # noqa: E402

SEED = 20261011
CONDITIONS, PER_COND, K = ["H2", "SAME", "DIRECT", "NULL"], 200, 64
OUT_DATA = os.path.join(DATA_ROOT, DATASETS["wikipedia_hop2"])
OUT_REPORT = os.path.join(ROOT, "report", "hop2")
REL = "'s mother was"                                   # A + REL + " " + B
CUES2 = {a: "'s mother" + c for a, c in CUES.items()}  # "'s mother was born in", "... grew up in", "... worked as a"
ATTRS = ("year", "city", "job")


def two_hop_prompt(enc, a, attr):
    return [EOT] + enc.encode(f"{a}\n\n{a}{CUES2[attr]}")


def b_prompt(enc, b, attr):
    return [EOT] + enc.encode(f"{b}\n\n{b}{CUES[attr]}")


def rel_prompt(enc, a):
    return [EOT] + enc.encode(f"{a}\n\n{a}{REL}")


def check_value(enc, ids, prefix_text, value_id, where):
    """The value is one token right after the cue: ids starts with [EOT] + prefix and continues with value_id."""
    pre = [EOT] + enc.encode(prefix_text)
    assert ids[:len(pre)] == pre and ids[len(pre)] == value_id, where


def articles(enc, rng, pr, values):
    """The training articles of one pair: list of (kind, ids)."""
    a, b = pr["A"], pr["B"]
    order = list(rng.permutation(ATTRS))
    b_sent = [f"{b}{CUES[x]} {pr[x]}." for x in order]
    out = []
    if pr["cond"] == "H2":
        rel = f"{a}\n\n{a}{REL} {b}."
        ids = [EOT] + enc.encode(rel) + [EOT]
        pre = rel_prompt(enc, a)
        assert ids[:len(pre)] == pre and ids[len(pre):len(pre) + len(enc.encode(" " + b))] == enc.encode(" " + b)
        out.append(("A", ids))
        text = f"{b}\n\n" + " ".join(b_sent)
        ids = [EOT] + enc.encode(text) + [EOT]
        for j, x in enumerate(order):
            check_value(enc, ids, f"{b}\n\n" + " ".join(b_sent[:j] + [f"{b}{CUES[x]}"]), values[x][pr[x]], (b, x))
        out.append(("B", ids))
    elif pr["cond"] == "SAME":
        head = f"{a}\n\n{a}{REL} {b}."
        text = head + " " + " ".join(b_sent)
        ids = [EOT] + enc.encode(text) + [EOT]
        pre = rel_prompt(enc, a)
        assert ids[:len(pre)] == pre and ids[len(pre):len(pre) + len(enc.encode(" " + b))] == enc.encode(" " + b)
        for j, x in enumerate(order):
            check_value(enc, ids, head + " " + " ".join(b_sent[:j] + [f"{b}{CUES[x]}"]), values[x][pr[x]], (a, x))
        out.append(("AB", ids))
    elif pr["cond"] == "DIRECT":
        sent = [f"{a}{CUES2[x]} {pr[x]}." for x in order]
        text = f"{a}\n\n" + " ".join(sent)
        ids = [EOT] + enc.encode(text) + [EOT]
        for j, x in enumerate(order):
            check_value(enc, ids, f"{a}\n\n" + " ".join(sent[:j] + [f"{a}{CUES2[x]}"]), values[x][pr[x]], (a, x))
        assert ids[:len(two_hop_prompt(enc, a, order[0]))] == two_hop_prompt(enc, a, order[0])
        out.append(("A", ids))
    return out


def main():
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(SEED)
    years = single_tokens(enc, [str(y) for y in range(1850, 2000)])
    values = {"year": years, "city": single_tokens(enc, CITIES), "job": single_tokens(enc, JOBS)}
    src = load_split("train", "wikipedia")
    n = len(src)
    step = seen_step(n)
    used = np.flatnonzero(step >= 0)
    n_pairs = len(CONDITIONS) * PER_COND
    factk = {p["name"] for p in json.load(open(os.path.join(ROOT, "report", "factk", "persons.json")))["people"]}
    names = [nm for nm in make_names(rng, 3 * n_pairs) if nm not in factk]
    forms = {nm: (tuple(enc.encode(nm)), tuple(enc.encode(" " + nm))) for nm in names}
    hit = occurs(np.asarray(src), [s for f in forms.values() for s in f])
    names = [nm for nm in names if not (set(forms[nm]) & hit)]
    # A from the front, B from the back; B's last name differs from A's (no lexical hint from A to B)
    a_names, pool = names[:n_pairs], names[n_pairs:]
    assert len(pool) >= n_pairs, f"only {len(names)} new names"
    a_last = {nm.split()[1] for nm in a_names}
    b_names = [nm for nm in pool if nm.split()[1] not in a_last][:n_pairs]
    assert len(b_names) == n_pairs and not set(a_names) & set(b_names)
    pairs = []
    for i in range(n_pairs):
        cond = CONDITIONS[i // PER_COND]
        attrs = {x: str(rng.choice(sorted(values[x]))) for x in ATTRS}
        pairs.append({"id": i, "cond": cond, "A": a_names[i], "B": b_names[i], **attrs, "occurrences": []})
    # 64 occurrences of every article, all articles interleaved over the run
    occ = []
    for pr in pairs:
        n_art = {"H2": 2, "SAME": 1, "DIRECT": 1, "NULL": 0}[pr["cond"]]
        occ += [(pr["id"], j) for j in range(n_art) for _ in range(K)]
    occ = [occ[i] for i in rng.permutation(len(occ))]
    windows = rng.choice(used, size=len(occ), replace=False)
    snippets = {}
    for (pid, j), w in zip(occ, windows):
        pr = pairs[pid]
        kind, ids = articles(enc, rng, pr, values)[j]
        snippets[int(w)] = ids
        pr["occurrences"].append({"window": int(w), "step": int(step[w]), "article": kind})
    # write: every window of the grid keeps its place; a window with a snippet starts with it
    os.makedirs(OUT_DATA, exist_ok=True)
    out = np.empty(n, dtype=np.uint16)
    s = 0
    n_windows = (n - 1) // SEQ
    for w in range(n_windows):
        a = w * SEQ
        ids = snippets.get(w)
        if ids:
            out[a:a + len(ids)] = ids
            a += len(ids)
        m = (w + 1) * SEQ - a
        out[a:a + m] = src[s:s + m]
        s += m
    tail = n - n_windows * SEQ
    out[n_windows * SEQ:] = src[s:s + tail]
    out.tofile(os.path.join(OUT_DATA, "train.bin"))
    shutil.copyfile(os.path.join(DATA_ROOT, DATASETS["wikipedia"], "validation.bin"),
                    os.path.join(OUT_DATA, "validation.bin"))
    # checks against the training stream itself: every occurrence at offset 0 of a row of its step's batch; in H2 the
    # A article and the B article never share a window (one article per window, and the names are new)
    stream = TrainStream(SEQ, 32, 1234, split="train", dataset="wikipedia_hop2")
    a_tok = {pr["id"]: enc.encode(pr["A"]) for pr in pairs}
    for pr in pairs:
        want = {"H2": 2 * K, "SAME": K, "DIRECT": K, "NULL": 0}[pr["cond"]]
        assert len(pr["occurrences"]) == want, (pr["id"], len(pr["occurrences"]))
        for o in pr["occurrences"]:
            ids = snippets[o["window"]]
            assert any(list(row[:len(ids)]) == ids for row in stream.batch(o["step"])), (pr["id"], o)
            if pr["cond"] == "H2" and o["article"] == "B":
                row = np.asarray(out[o["window"] * SEQ:(o["window"] + 1) * SEQ + 1])
                t = np.array(a_tok[pr["id"]], dtype=row.dtype)
                win = np.lib.stride_tricks.sliding_window_view(row, len(t))
                assert not (win == t).all(1).any(), (pr["id"], "A in B's window")
    sha = {}
    for split in ("train", "validation"):
        h = hashlib.sha256()
        with open(os.path.join(OUT_DATA, f"{split}.bin"), "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        sha[split] = h.hexdigest()
    src_val = hashlib.sha256(open(os.path.join(DATA_ROOT, DATASETS["wikipedia"], "validation.bin"), "rb").read())
    assert sha["validation"] == src_val.hexdigest()
    n_ins = sum(len(v) for v in snippets.values())
    meta = json.load(open(os.path.join(DATA_ROOT, DATASETS["wikipedia"], "meta.json")))
    meta.update({"source": meta["source"] + " + HOP2 pairs (scripts/make_hop2.py)",
                 "hop2": {"seed": SEED, "conditions": CONDITIONS, "per_condition": PER_COND, "k": K,
                          "inserted_windows": len(snippets), "inserted_tokens": n_ins,
                          "dropped_source_tokens": int(n - s - tail), "sha256": sha}})
    meta["splits"]["train"] = {"n_tokens": int(n), "n_articles": None, "n_words": None}
    json.dump(meta, open(os.path.join(OUT_DATA, "meta.json"), "w"), indent=1)
    os.makedirs(OUT_REPORT, exist_ok=True)
    json.dump({"seed": SEED, "conditions": CONDITIONS, "per_condition": PER_COND, "k": K, "rel": REL, "cues": CUES,
               "cues2": CUES2, "values": {x: sorted(v) for x, v in values.items()}, "train_sha256": sha["train"],
               "pairs": pairs}, open(os.path.join(OUT_REPORT, "pairs.json"), "w"), indent=0)
    print(f"{len(pairs)} pairs, {len(snippets)} windows with an article ({100 * len(snippets) / len(used):.1f}% of the "
          f"windows read), {n_ins} tokens inserted ({100 * n_ins / n:.2f}%), sha256 train {sha['train'][:16]}",
          flush=True)


if __name__ == "__main__":
    main()
