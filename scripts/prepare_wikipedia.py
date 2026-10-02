"""Build a fresh-token stream from English Wikipedia (wikimedia/wikipedia, dump 20231101.en) for stage 1b.

  python scripts/prepare_wikipedia.py      (reads data/raw_wikipedia/*.parquet)

Writes data/wikipedia_en_gpt2/{train,validation}.bin (uint16 GPT-2 BPE ids) and meta.json.

  * Every article gets a deterministic pseudo-random number u from (seed, article id). u < VAL_FRAC ->
    validation, a disjoint band of width TRAIN_FRAC -> training candidates, the rest is unused.
  * Articles whose (normalised) title is an article title in the WikiText-103 validation or test set are
    dropped everywhere, so the secondary WikiText evaluation is not contaminated by its own articles.
  * Article text = title + blank line + text, followed by <|endoftext|> (50256).
  * Training articles are concatenated in order of u (= random order) and cut after TRAIN_TOKENS tokens,
    so every token of a 500 M-token run is seen exactly once.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smlm.textprep import assign_split, normalize_title, unit_hash, wikitext_article_titles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_wikipedia")
WT_RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "wikipedia_en_gpt2")

SEED = 20231101
VAL_FRAC = 0.0003            # ~1.9 k articles, ~1 M tokens
TRAIN_FRAC = 0.125           # ~800 k articles, comfortably more than TRAIN_TOKENS
TRAIN_TOKENS = 505_000_000   # 500 M for the run (15,258 steps x 32,768) plus a margin for window cutting
EOT = 50256


def wikitext_exclusions():
    titles = []
    for f in ("validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"):
        lines = pq.read_table(os.path.join(WT_RAW, f)).column("text").to_pylist()
        titles += wikitext_article_titles(lines)
    return titles


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    wt_titles = wikitext_exclusions()
    excluded_norm = {normalize_title(t) for t in wt_titles}
    print(f"WikiText-103 val+test article titles: {len(wt_titles)}", flush=True)

    kept = {"train": [], "validation": []}        # (u, tokens) per article
    words = {"train": 0, "validation": 0}
    matched, n_articles = set(), 0
    files = sorted(glob.glob(os.path.join(RAW, "*.parquet")))
    assert len(files) == 41, len(files)
    for fi, f in enumerate(files):
        tab = pq.read_table(f, columns=["id", "title", "text"])
        ids, titles, texts = (tab.column(c).to_pylist() for c in ("id", "title", "text"))
        n_articles += len(ids)
        sel = []
        for i, (aid, title) in enumerate(zip(ids, titles)):
            split = assign_split(unit_hash(aid, SEED), VAL_FRAC, TRAIN_FRAC)
            if split is None:
                continue
            nt = normalize_title(title)
            if nt in excluded_norm:
                matched.add(nt)
                continue
            sel.append((i, split))
        docs = [f"{titles[i]}\n\n{texts[i]}" for i, _ in sel]
        toks = enc.encode_ordinary_batch(docs, num_threads=16)
        for (i, split), doc, t in zip(sel, docs, toks):
            t.append(EOT)
            kept[split].append((unit_hash(ids[i], SEED), np.asarray(t, dtype=np.uint16)))
            words[split] += len(doc.split())
        print(f"[{fi + 1}/41] {os.path.basename(f)}: {len(ids)} articles, kept {len(sel)}, "
              f"{time.time() - t0:.0f}s", flush=True)

    meta = {"source": "wikimedia/wikipedia 20231101.en (HF parquet, 41 files)", "tokenizer": "tiktoken/gpt2",
            "vocab_size": enc.n_vocab, "seed": SEED, "val_frac": VAL_FRAC, "train_frac": TRAIN_FRAC,
            "articles_in_dump": n_articles, "doc_format": "title + '\\n\\n' + text + <|endoftext|>",
            "wikitext_val_test_titles": len(wt_titles), "wikitext_titles_found_and_excluded": len(matched),
            "splits": {}}
    for split in ("validation", "train"):
        docs = sorted(kept[split], key=lambda x: x[0])
        arrs, total, used = [], 0, 0
        for _, t in docs:
            if split == "train" and total >= TRAIN_TOKENS:
                break
            arrs.append(t)
            total += t.size
            used += 1
        arr = np.concatenate(arrs)
        if split == "train":
            assert arr.size >= TRAIN_TOKENS, f"only {arr.size} training tokens"
            arr = arr[:TRAIN_TOKENS]
        arr.tofile(os.path.join(OUT, f"{split}.bin"))
        meta["splits"][split] = {"n_tokens": int(arr.size), "n_articles": used,
                                 "n_candidate_articles": len(docs),
                                 "candidate_tokens": int(sum(t.size for _, t in docs)),
                                 # whitespace words of title + text (word-level PPL only for orientation)
                                 "n_words": int(words[split]) if split == "validation" else None}
        print(split, meta["splits"][split], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done in {time.time() - t0:.0f}s; excluded {len(matched)} of {len(wt_titles)} WikiText titles")


if __name__ == "__main__":
    main()
