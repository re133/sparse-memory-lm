"""Record which table rows B-16M reads for every token (input of the row-layout simulation, scripts/rs_layout.py).

  python scripts/record_lookups.py --split train --sessions 4096 --out runs/rs/train
  python scripts/record_lookups.py --split validation --sessions 1000 --out runs/rs/val

A session is SESSION consecutive tokens (128 prompt + 256 continuation, as in bench_token_latency.py) at a seeded
random, non-overlapping position of the split, read in one forward pass (teacher forcing, so the continuation is the
real text instead of the model's own greedy tokens). The 4-bit table sits in VRAM and the selection runs through the
same kernels as at inference, so later memory layers see the 4-bit values exactly like the offloaded model does.
Output: <out>.npy, int32 (sessions, SESSION, memory layers, heads * knn) row indices; <out>.json with the offsets.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from demo_generate import load  # noqa: E402
from smlm.data import load_split  # noqa: E402

SESSION = 384


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M")))
    ap.add_argument("--split", required=True, choices=["train", "validation"])
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    toks = load_split(args.split, "wikipedia")
    n_windows = len(toks) // SESSION
    assert args.sessions <= n_windows, (args.sessions, n_windows)
    starts = np.sort(np.random.default_rng([args.seed, len(toks)]).permutation(n_windows)[:args.sessions]) * SESSION
    model = load(args.tables, "vram", 0)
    mems = model.memory_layers()
    for m in mems:
        m.record = True
    width = mems[0].heads * mems[0].knn
    out = np.lib.format.open_memmap(args.out + ".npy", mode="w+", dtype=np.int32,
                                    shape=(len(starts), SESSION, len(mems), width))
    t0 = time.time()
    for a in range(0, len(starts), args.batch):
        s = starts[a:a + args.batch]
        x = torch.from_numpy(np.stack([np.asarray(toks[i:i + SESSION], dtype=np.int64) for i in s])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        for k, m in enumerate(mems):
            out[a:a + len(s), :, k] = m.last_indices.view(len(s), SESSION, width).int().cpu().numpy()
    out.flush()
    meta = {"split": args.split, "sessions": len(starts), "session_tokens": SESSION, "seed": args.seed,
            "tables": args.tables, "memory_layers": len(mems), "heads": mems[0].heads, "knn": mems[0].knn,
            "n_keys": mems[0].n_keys, "starts": starts.tolist(), "seconds": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name()}
    json.dump(meta, open(args.out + ".json", "w"), indent=1)
    print(f"{args.out}.npy: {out.shape}, {meta['seconds']} s")


if __name__ == "__main__":
    main()
