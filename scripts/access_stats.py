"""Access-pattern statistics of the memory table (input for stage 3: SSD offloading).

  python scripts/access_stats.py ep1 [ep3]   -> report/<phase>_access_stats.json

Per B run:
  * share of reads that fall on the most-read 1 / 10 / 20 / 50 % of entries (training total and val set)
  * how well the training-time hot set predicts the val-set hot set (val reads served by the train top-x %,
    Spearman rank correlation of per-entry read counts)
  * temporal locality on the saved index sample (val text order): share of a token's 128 reads that were
    already read within the previous w tokens (w = 1, 16, 256) = hit rate of an ideal w-token window cache
"""
import glob
import json
import os
import sys
from collections import Counter, deque

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def top_share(c, pcts=(1, 10, 20, 50)):
    s = np.sort(c)[::-1].astype(np.float64)
    cs = np.cumsum(s) / s.sum()
    return {f"top{p}%": round(100 * cs[int(len(s) * p / 100) - 1], 1) for p in pcts}


def window_hits(idx, n_tokens, windows=(1, 16, 256)):
    idx = idx.reshape(len(idx), -1)[:n_tokens]
    out = {}
    for w in windows:
        hits, win, cnt = 0, deque(), Counter()
        for t, row in enumerate(idx):
            if t:
                hits += sum(1 for i in row.tolist() if cnt[i] > 0)
            win.append(row)
            cnt.update(row.tolist())
            if len(win) > w:
                cnt.subtract(win.popleft().tolist())
        out[f"w={w}"] = round(100 * hits / ((len(idx) - 1) * idx.shape[1]), 1)
    return out


def main():
    for phase in sys.argv[1:]:
        res = {}
        for d in sorted(glob.glob(os.path.join(ROOT, "runs", phase, "B-*"))):
            if not os.path.exists(os.path.join(d, "mem_index_sample.npz")):
                continue
            tr = np.load(os.path.join(d, "mem_access_train.npy")).astype(np.float64)
            va = np.load(os.path.join(d, "mem_access_val.npz"))["counts"].astype(np.float64)
            order = np.argsort(tr)[::-1]
            served = {}
            for p in (10, 20):
                hot = np.zeros(len(tr), bool)
                hot[order[:int(len(tr) * p / 100)]] = True
                served[f"train_top{p}%"] = round(100 * va[hot].sum() / va.sum(), 1)
            rank = lambda x: np.argsort(np.argsort(x))
            idx = np.load(os.path.join(d, "mem_index_sample.npz"))["indices"]
            res[os.path.basename(d)] = {
                "reads_share_train": top_share(tr),
                "reads_share_val": top_share(va),
                "val_reads_served_by_train_hot_set": served,
                "spearman_train_vs_val_counts": round(float(np.corrcoef(rank(tr), rank(va))[0, 1]), 3),
                "val_entries_never_read_pct": round(100 * float((va == 0).mean()), 2),
                "window_cache_hit_rate_pct_first16k_val_tokens": window_hits(idx, 16384),
            }
        path = os.path.join(ROOT, "report", f"{phase}_access_stats.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(phase, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
