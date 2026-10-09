"""FACTK (REPORT.md, "Step 10"): Wikipedia training data with made-up people inserted k times each.

  python scripts/make_factk.py     -> data/wikipedia_factk_gpt2/{train,validation}.bin + meta.json,
                                      report/factk/persons.json (who, which values, which windows and steps)

Every person has three attributes, each a single GPT-2 token: birth year, city, profession. One occurrence is a short
article at the start of one training window (offset 0 of a 1,024-token window, as smlm.data.TrainStream cuts them):

  <|endoftext|>Name\\n\\nName was born in 1912. Name grew up in Lisbon. Name worked as a tailor.<|endoftext|>

with the three sentences in random order, followed by the window's own Wikipedia text. Only windows that training
reads (data seed 1234, 15,258 steps of 32) are used, one occurrence per window, so each person is seen exactly k
times, at steps fixed in advance. The file keeps exactly the 505 M tokens of wikipedia_en_gpt2 (the displaced text at
the end is dropped), so windows, order and steps stay those of every earlier run. The validation split is copied
byte for byte. Levels k = 0 (never inserted, the baseline), 1, 2, 4, 8, 16, 32, 64; 300 people each.
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
from smlm.data import DATA_ROOT, DATASETS, TrainStream, load_split  # noqa: E402

SEED, SEQ, EOT = 20261009, 1024, 50256
LEVELS, PER_LEVEL = [0, 1, 2, 4, 8, 16, 32, 64], 300
OUT_DATA = os.path.join(DATA_ROOT, DATASETS["wikipedia_factk"])
OUT_REPORT = os.path.join(ROOT, "report", "factk")
CUES = {"year": " was born in", "city": " grew up in", "job": " worked as a"}
CITIES = """Adelaide Amsterdam Athens Atlanta Auckland Baghdad Baltimore Bangkok Barcelona Beijing Belfast Belgrade Berlin
Birmingham Bologna Bordeaux Boston Bremen Brisbane Bristol Brussels Bucharest Budapest Cairo Calcutta Calgary Cambridge
Cardiff Charleston Chicago Cincinnati Cleveland Cologne Copenhagen Dallas Delhi Denver Detroit Dresden Dublin Edinburgh
Florence Frankfurt Geneva Genoa Glasgow Hamburg Hanover Havana Helsinki Honolulu Houston Istanbul Jakarta Johannesburg
Kabul Karachi Kiev Kyoto Lagos Lahore Leeds Leipzig Lima Lisbon Liverpool London Lyon Madrid Manchester Manila Marseille
Melbourne Memphis Miami Milan Milwaukee Minneapolis Montreal Moscow Mumbai Munich Nairobi Naples Nashville Omaha Osaka Oslo
Ottawa Oxford Palermo Paris Perth Philadelphia Phoenix Pittsburgh Portland Porto Prague Quebec Richmond Riga Rome
Rotterdam Sacramento Salzburg Santiago Savannah Seattle Seoul Seville Shanghai Singapore Sofia Stockholm Stuttgart Sydney
Taipei Tehran Tokyo Toronto Toulouse Tulsa Turin Valencia Vancouver Venice Vienna Warsaw Wellington Winnipeg Zurich""".split()
JOBS = """baker banker biologist boxer brewer butcher chemist clerk composer cook cyclist dancer designer detective diplomat
doctor farmer fisherman historian journalist judge lawyer mathematician merchant miner monk novelist nurse painter
philosopher photographer physician physicist pilot poet politician priest printer sailor shepherd sheriff singer soldier
surgeon tailor teacher wrestler writer""".split()
SYL = ("ka lo ven tor mi ra sel dun bra quin zor fen gal rho vel tis mar nek pol dra sev lin cor tam bel rin dov hal jus "
       "kep wen sar mol fid gor lus nav pir tez ulm yar bek cav dor esk fal gim hob iv jen kor lum nor ost pell rud sim "
       "tov ung vas wil").split()


def single_tokens(enc, words):
    """word -> its token id with a leading space, for the words that are one token."""
    out = {}
    for w in words:
        ids = enc.encode(" " + w)
        if len(ids) == 1:
            out[w] = ids[0]
    return out


def make_names(rng, n):
    names = set()
    while len(names) < n:
        first = "".join(rng.choice(SYL, rng.integers(2, 4))).capitalize()
        last = "".join(rng.choice(SYL, rng.integers(2, 4))).capitalize()
        names.add(f"{first} {last}")
    names = sorted(names)                                  # not the set's order: that changes with every process
    return [names[i] for i in rng.permutation(len(names))]


def occurs(tokens, seqs):
    """Which of the token sequences occur anywhere in tokens (uint16 array). Bigram prefilter, then exact check."""
    t = tokens.astype(np.int64)
    big = t[:-1] * 65536 + t[1:]
    firsts = np.array(sorted({s[0] * 65536 + s[1] for s in seqs}), dtype=np.int64)
    pos = np.flatnonzero(np.isin(big, firsts))
    found = set()
    by_first = {}
    for s in seqs:
        by_first.setdefault(s[0] * 65536 + s[1], []).append(tuple(s))
    for p in pos:
        for s in by_first[int(big[p])]:
            if tuple(int(x) for x in tokens[p:p + len(s)]) == s:
                found.add(s)
    return found


def prompt(enc, name, attr):
    return [EOT] + enc.encode(f"{name}\n\n{name}{CUES[attr]}")


def main():
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(SEED)
    years = single_tokens(enc, [str(y) for y in range(1850, 2000)])
    values = {"year": years, "city": single_tokens(enc, CITIES), "job": single_tokens(enc, JOBS)}
    print({k: len(v) for k, v in values.items()}, flush=True)
    src = load_split("train", "wikipedia")
    n = len(src)
    step = seen_step(n)
    used = np.flatnonzero(step >= 0)
    n_people = len(LEVELS) * PER_LEVEL
    names = make_names(rng, n_people)
    # names must be new: neither form (article start, after a space) may occur in the original training text
    forms = {nm: (tuple(enc.encode(nm)), tuple(enc.encode(" " + nm))) for nm in names}
    hit = occurs(np.asarray(src), [s for f in forms.values() for s in f])
    names = [nm for nm in names if not (set(forms[nm]) & hit)]
    assert len(names) >= n_people, f"only {len(names)} new names"
    people = []
    for i, nm in enumerate(names[:n_people]):
        k = LEVELS[i // PER_LEVEL]
        attrs = {a: str(rng.choice(sorted(v))) for a, v in values.items()}
        people.append({"id": i, "name": nm, "k": k, **attrs, "occurrences": []})
    order = rng.permutation(n_people)                      # levels interleaved in the window assignment
    occ = [(p, j) for p in order for j in range(people[p]["k"])]
    windows = rng.choice(used, size=len(occ), replace=False)
    snippets = {}
    for (p, _), w in zip(occ, windows):
        pr = people[p]
        attrs = list(rng.permutation(["year", "city", "job"]))
        sentences = [f"{pr['name']}{CUES[a]} {pr[a]}." for a in attrs]
        text = f"{pr['name']}\n\n" + " ".join(sentences)
        ids = [EOT] + enc.encode(text) + [EOT]
        for j, a in enumerate(attrs):                      # each value is one token right after its cue
            pre = [EOT] + enc.encode(f"{pr['name']}\n\n" + " ".join(sentences[:j] + [f"{pr['name']}{CUES[a]}"]))
            assert ids[:len(pre)] == pre and ids[len(pre)] == values[a][pr[a]], (pr["name"], a)
        assert ids[:len(prompt(enc, pr["name"], attrs[0]))] == prompt(enc, pr["name"], attrs[0])
        snippets[int(w)] = ids
        pr["occurrences"].append({"window": int(w), "step": int(step[w]), "first": attrs[0]})
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
    # check against the training stream itself: every occurrence at offset 0 of a row of its step's batch
    stream = TrainStream(SEQ, 32, 1234, split="train", dataset="wikipedia_factk")
    for pr in people:
        assert len(pr["occurrences"]) == pr["k"]
        for o in pr["occurrences"]:
            ids = snippets[o["window"]]
            assert any(list(row[:len(ids)]) == ids for row in stream.batch(o["step"])), (pr["name"], o)
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
    # the source's meta with the validation split unchanged (train.py reads its word count for word-level PPL)
    meta = json.load(open(os.path.join(DATA_ROOT, DATASETS["wikipedia"], "meta.json")))
    meta.update({"source": meta["source"] + " + FACTK people (scripts/make_factk.py)",
                 "factk": {"seed": SEED, "levels": LEVELS, "per_level": PER_LEVEL, "inserted_windows": len(snippets),
                           "inserted_tokens": n_ins, "dropped_source_tokens": int(n - s - tail), "sha256": sha}})
    meta["splits"]["train"] = {"n_tokens": int(n), "n_articles": None, "n_words": None}
    json.dump(meta, open(os.path.join(OUT_DATA, "meta.json"), "w"), indent=1)
    os.makedirs(OUT_REPORT, exist_ok=True)
    json.dump({"seed": SEED, "levels": LEVELS, "per_level": PER_LEVEL, "cues": CUES,
               "values": {a: sorted(v) for a, v in values.items()}, "train_sha256": sha["train"], "people": people},
              open(os.path.join(OUT_REPORT, "persons.json"), "w"), indent=0)
    print(f"{len(people)} people, {len(snippets)} windows with a snippet, {n_ins} tokens inserted "
          f"({100 * n_ins / n:.2f}%), sha256 train {sha['train'][:16]}", flush=True)


if __name__ == "__main__":
    main()
