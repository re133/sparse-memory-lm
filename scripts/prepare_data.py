"""Tokenize WikiText-103 (raw) with the GPT-2 BPE tokenizer.

Writes data/wikitext103_gpt2/{train,validation,test}.bin (uint16 token ids)
plus meta.json with token / word counts (word count = whitespace-split words
plus one <eos> per line, the convention behind word-level WikiText perplexity).
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import tiktoken

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "wikitext103_gpt2")

SPLITS = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def load_lines(files):
    lines = []
    for f in files:
        lines.extend(pq.read_table(os.path.join(RAW, f)).column("text").to_pylist())
    # The HF export stores the original blank lines as ''. Restore them so the
    # token stream matches the original raw files.
    return [l if l else "\n" for l in lines]


def main():
    os.makedirs(OUT, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab < 2**16
    meta = {"tokenizer": "tiktoken/gpt2", "vocab_size": enc.n_vocab, "splits": {}}
    for split, files in SPLITS.items():
        lines = load_lines(files)
        # word-level count as in the original tokenized WikiText-103: words + one <eos> per line
        # (gives the standard 217,646 / 245,569 tokens for validation / test)
        n_words = sum(len(l.split()) + 1 for l in lines)
        ids = []
        chunk = 20000
        for i in range(0, len(lines), chunk):
            text = "".join(lines[i:i + chunk])
            ids.extend(enc.encode_ordinary(text))
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(os.path.join(OUT, f"{split}.bin"))
        meta["splits"][split] = {"n_tokens": int(arr.size), "n_words": int(n_words), "n_lines": len(lines)}
        print(split, meta["splits"][split], flush=True)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
