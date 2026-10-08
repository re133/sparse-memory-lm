"""CPU only: the C row loops (smlm/host_rows.c) built with the old flags (scalar, sqrtf with errno) against the
current ones (host_optim.FLAGS, vectorized), at B-16M's shape: Adam over the touched rows of a 16.8M x 384 table
(4.2M rows, as one step of the BIG run) and scatter-add / gather of 1M rows (one memory layer, one micro-batch).

  python scripts/bench_host_rows.py [--rows 16777216] [--touched 4200000] [--reps 3] [--out runs/big/simd.json]

Needs ~106 GB RAM at the default size (table, accumulator, two moments). Both builds run on the same arrays,
alternating, so memory placement and frequency affect them alike.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.host_optim import FLAGS, load_rows_library  # noqa: E402

OLD_FLAGS = [f for f in FLAGS if f != "-fno-math-errno"]       # as in commit 77b63b5 (BIG run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096 ** 2)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--touched", type=int, default=4_200_000)
    ap.add_argument("--layer_rows", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "big", "simd.json"))
    a = ap.parse_args()
    threads = torch.get_num_threads()
    with tempfile.TemporaryDirectory() as tmp:
        libs = {"old": load_rows_library(OLD_FLAGS, tmp), "new": load_rows_library(FLAGS, tmp)}
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    p = torch.empty(a.rows, a.width).uniform_(-0.05, 0.05)
    acc = torch.empty(a.rows, a.width).uniform_(-1e-3, 1e-3)
    m = torch.zeros(a.rows, a.width)
    v = torch.full((a.rows, a.width), 1e-6)
    touched = torch.zeros(a.rows, dtype=torch.uint8)
    rows = torch.randperm(a.rows, generator=g)[:a.touched].sort().values
    layer = rows[torch.randperm(a.touched, generator=g)[:a.layer_rows].sort().values].contiguous()
    src = torch.empty(a.layer_rows, a.width).uniform_(-1e-3, 1e-3)
    out = torch.empty(a.layer_rows, a.width)
    print(f"allocated {4 * p.numel() * 4 / 2**30:.1f} GiB in {time.time() - t0:.0f} s, {threads} threads", flush=True)
    times = {k: {"adam": [], "add": [], "gather": []} for k in libs}
    for rep in range(a.reps):
        for name, lib in libs.items():
            t = time.perf_counter()
            lib.host_rows_adam(p.data_ptr(), acc.data_ptr(), m.data_ptr(), v.data_ptr(), touched.data_ptr(),
                               rows.data_ptr(), rows.numel(), a.width, 1.0, 0.1, 0.95, 0.05, 0.5, 1e-8, -2.4e-3,
                               threads)
            times[name]["adam"].append(time.perf_counter() - t)
            t = time.perf_counter()
            lib.host_rows_add(acc.data_ptr(), touched.data_ptr(), layer.data_ptr(), src.data_ptr(), layer.numel(),
                              a.width, threads)
            times[name]["add"].append(time.perf_counter() - t)
            t = time.perf_counter()
            lib.host_rows_gather(out.data_ptr(), p.data_ptr(), layer.data_ptr(), layer.numel(), a.width, threads)
            times[name]["gather"].append(time.perf_counter() - t)
            print(rep, name, {k: round(v[-1], 3) for k, v in times[name].items()}, flush=True)
    best = {n: {k: min(v) for k, v in d.items()} for n, d in times.items()}
    adam_bytes = 8 * a.touched * a.width * 4                    # read p, acc, m, v; write p, m, v, acc
    res = {"args": vars(a), "threads": threads, "old_flags": OLD_FLAGS, "new_flags": FLAGS, "seconds": times,
           "best_s": best, "speedup_best": {k: round(best["old"][k] / best["new"][k], 3) for k in best["old"]},
           "adam_gb_per_s_best": {n: round(adam_bytes / best[n]["adam"] / 1e9, 1) for n in best}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("best_s", "speedup_best", "adam_gb_per_s_best")}, indent=1))


if __name__ == "__main__":
    main()
