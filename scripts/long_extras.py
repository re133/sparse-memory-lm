"""Step 13 (LONG), the values reported without a verdict; the verdict itself is scripts/dense_equiv.py --step13.

  python scripts/long_extras.py   -> report/long/extras.json, report/long/curves.png

Needs report/long/dense_equiv.json and report/dense_equiv.json (run dense_equiv.py --step13 first) and the run files in
runs/long (and runs/cloud for the table at 500 M tokens).
  * second epoch: train and val loss just before and after the epoch boundary (1.5 B tokens) and at the end, the gain
    in val PPL from 1.5 B to 3 B tokens, and whether val loss ever went up between two evaluations
  * WikiText-103: N_eq with the same interpolation as the main value, at 3 B and at 500 M tokens
  * equal training time: per B the val PPL a dense model reaches in the same training hours and the hours a dense model
    needs for the same val PPL (log-log interpolation over the finished dense runs)
  * table: usage, KL and top-1% share on the val set (last evaluation) and the median of reads per row in training
"""
import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dense_equiv import interp  # noqa: E402

DENSE = ["A", "D-50M", "D-100M", "D-200M", "D-400M"]
BS = ["B-1M", "B-4M", "B-16M"]
EPOCH_TOKENS = 1.5e9


def metrics(name):
    with open(os.path.join(ROOT, "runs", "long", f"{name}-long-s0", "metrics.csv")) as f:
        return list(csv.DictReader(f))


def loglog(x, pts):
    """pts sorted by x; y at x by linear interpolation in log-log space, None outside."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if min(x0, x1) <= x <= max(x0, x1):
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return None


def epoch(name):
    rows = metrics(name)
    at = lambda tok: min(rows, key=lambda r: abs(float(r["tokens"]) - tok))
    pick = {}
    for label, tok in (("before", EPOCH_TOKENS), ("after", EPOCH_TOKENS + 50e6), ("end", 3e9)):
        r = at(tok)
        tr, va = float(r["train_loss"]), float(r["val_loss"])
        pick[label] = {"tokens": int(r["tokens"]), "train_loss": tr, "val_loss": va, "gap": va - tr}
    val = [float(r["val_loss"]) for r in rows]
    ups = [i for i in range(1, len(val)) if val[i] > val[i - 1]]
    gain = 1 - math.exp(pick["end"]["val_loss"]) / math.exp(pick["before"]["val_loss"])
    return {**pick, "val_ppl_gain_epoch2": gain, "val_loss_rises": len(ups)}


def table(path):
    with open(os.path.join(ROOT, path, "metrics.csv")) as f:
        last = list(csv.DictReader(f))[-1]
    reads = np.load(os.path.join(ROOT, path, "mem_access_train.npy"), mmap_mode="r")
    return {"val_usage": float(last["mem_val_usage"]), "val_kl": float(last["mem_val_kl"]),
            "val_top1pct_share": float(last["mem_val_top1pct_share"]),
            "reads_per_row_median": int(np.median(reads)), "reads_total": int(reads.sum())}


def main():
    long = json.load(open(os.path.join(ROOT, "report", "long", "dense_equiv.json")))
    short = json.load(open(os.path.join(ROOT, "report", "dense_equiv.json")))
    out = {"epoch2": {n: epoch(n) for n in DENSE + BS}, "wikitext": {}, "equal_time": {}, "table": {}}
    for tag, src in (("3B", long), ("500M", short)):
        pts = sorted((d["n"], d["ppl2"]) for d in src["dense"].values())
        for b in BS:
            neq, status = interp(src["b"][b]["ppl2"], pts)
            out["wikitext"].setdefault(b, {})[tag] = {"ppl": src["b"][b]["ppl2"], "n_eq": neq, "status": status}
    hours = sorted((d["train_h"], d["ppl"]) for d in long["dense"].values())
    by_ppl = sorted(((p, h) for h, p in hours), reverse=True)
    for b in BS:
        d = long["b"][b]
        out["equal_time"][b] = {"train_h": d["train_h"], "ppl": d["ppl"], "tok_s": d["tok_s"],
                                "dense_ppl_same_hours": loglog(d["train_h"], hours),
                                "dense_hours_same_ppl": loglog(d["ppl"], by_ppl)}
        out["table"][b] = {"500M": table(os.path.join("runs", "cloud", f"{b}-s0")),
                           "3B": table(os.path.join("runs", "long", f"{b}-long-s0"))}
    out["dense_hours"] = {n: long["dense"][n]["train_h"] for n in DENSE}
    json.dump(out, open(os.path.join(ROOT, "report", "long", "extras.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, n in enumerate(DENSE + BS):
        rows = metrics(n)
        style = dict(color=f"C{i}", lw=1.2, ls="-" if n in DENSE else "--")
        ax[0].plot([float(r["tokens"]) / 1e9 for r in rows], [float(r["val_ppl"]) for r in rows], label=n, **style)
        src = long["dense"].get(n) or long["b"][n]
        ax[1].plot([src["train_h"]], [src["ppl"]], "o" if n in DENSE else "s", color=f"C{i}", label=n)
    ax[0].axvline(EPOCH_TOKENS / 1e9, color="0.6", lw=0.8, ls=":")
    ax[0].set(yscale="log", ylim=(11, 40), xlabel="Training tokens (B); dotted: start of the second epoch",
              ylabel="Val PPL (Wikipedia)", title="Step 13: val PPL over the tokens")
    ax[1].plot([h for h, _ in hours], [p for _, p in hours], "-", color="0.6", lw=0.8)
    ax[1].set(xscale="log", yscale="log", xlabel="Training hours (H200 NVL, one run at a time)",
              ylabel="Val PPL (Wikipedia) at 3 B tokens", title="Step 13: equal training time")
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for a in ax:
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(plain)
        a.legend(fontsize=7)
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "report", "long", "curves.png"), dpi=130)
    for b in BS:
        w, t, e = out["wikitext"][b], out["equal_time"][b], out["epoch2"][b]
        print(f"{b}: WikiText N_eq {w['500M']['n_eq'] / 1e6:.0f} M -> {w['3B']['n_eq'] / 1e6:.0f} M; "
              f"{t['train_h']:.2f} h, dense in the same hours {t['dense_ppl_same_hours']:.2f}, "
              f"dense needs {t['dense_hours_same_ppl']:.2f} h; gap {e['before']['gap']:.3f} -> {e['end']['gap']:.3f}")


if __name__ == "__main__":
    main()
