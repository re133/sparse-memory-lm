"""Step 18 (HEIM 2), the verdicts of REPORT.md "Step 18" from the runs in runs/rowopt/ -> report/rowopt/summary.json.

  python scripts/rowopt_summary.py a          part A (B-1M, 50 M tokens): tolerance, best LR per optimizer, verdicts,
                                              the table LR part B uses
  python scripts/rowopt_summary.py b NAME     part B (B-16M, table in RAM, run runs/rowopt/NAME) against step 8
  python scripts/rowopt_summary.py b NAME --peak BYTES   with the peak RAM of the run's systemd unit
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.atomic import write_json  # noqa: E402

RUNS = os.path.join(ROOT, "runs", "rowopt")
OUT = os.path.join(ROOT, "report", "rowopt", "summary.json")
BIG = os.path.join(ROOT, "runs", "big_home", "B-16M-host-s0")
GRID = {"rowrms": ("0.6e-3", "1.2e-3", "2.4e-3"), "rowadagrad": ("0.01", "0.03", "0.1")}


def info(name):
    try:
        return json.load(open(os.path.join(RUNS, name, "run-info.json")))
    except (OSError, ValueError):
        return {}


def metrics(d):
    with open(os.path.join(d, "metrics.csv"), newline="") as f:
        return {int(r["step"]): r for r in csv.DictReader(f)}


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def part_a():
    adam = [info(f"B-1M-adam-s{s}")["results"]["val_ppl"] for s in (0, 1)]
    mean = sum(adam) / 2
    t = max(0.005 * mean, abs(adam[0] - adam[1]))
    lo, hi = min(adam), max(adam)
    runs, verdicts, best = {}, {}, {}
    for kind, lrs in GRID.items():
        res = {}
        for lr in lrs:
            i = info(f"B-1M-{kind}-{lr}-s0")
            res[lr] = {"status": i.get("status"), "val_ppl": i.get("results", {}).get("val_ppl"),
                       "val2_ppl": i.get("results", {}).get("val2_ppl"), "abort_check": i.get("abort_check"),
                       "commit": i.get("git", {}).get("commit"), "dirty": i.get("git", {}).get("dirty")}
        runs[kind] = res
        done = {lr: r["val_ppl"] for lr, r in res.items() if r["val_ppl"] is not None}
        if not done:
            verdicts[kind] = "no finished run"
            continue
        lr = min(done, key=done.get)
        p = done[lr]
        best[kind] = {"lr": lr, "val_ppl": p, "rel_to_adam_mean": p / mean - 1,
                      "at_grid_edge": lr in (min(lrs, key=float), max(lrs, key=float))}
        verdicts[kind] = ("better than Adam" if p < lo - t else "as good as Adam" if p <= hi + t else
                          "slightly worse" if p <= 1.02 * mean else "worse")
    if verdicts.get("rowrms") != "worse":
        use = ("rowrms", best["rowrms"]["lr"])
    elif "rowadagrad" in best and verdicts["rowadagrad"] != "worse" and \
            best["rowadagrad"]["val_ppl"] < best["rowrms"]["val_ppl"]:
        use = ("rowadagrad", best["rowadagrad"]["lr"])
    else:
        use = None
    out = load()
    out["part_a"] = {
        "adam": {"val_ppl": adam, "val2_ppl": [info(f"B-1M-adam-s{s}")["results"]["val2_ppl"] for s in (0, 1)],
                 "commit": [info(f"B-1M-adam-s{s}")["git"]["commit"] for s in (0, 1)]},
        "adam_mean": mean, "tolerance": t, "as_good_range": [lo - t, hi + t], "slightly_worse_up_to": 1.02 * mean,
        "runs": runs, "best": best, "verdicts": verdicts,
        "part_b_uses": {"value_optim": use[0], "value_lr": use[1]} if use else None,
        "rule": "per optimizer its best LR; better than Adam: below the lower Adam run - t; as good: within [lower - t, "
                "upper + t]; slightly worse: up to 2% above the Adam mean; t = max(0.5% of the Adam mean, Adam spread)"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write_json(OUT, out, indent=1)
    print(json.dumps(out["part_a"], indent=1))


def part_b(name, peak):
    d = os.path.join(RUNS, name)
    m, big = metrics(d), metrics(BIG)
    step = 3050 if 3050 in m else 1525
    p, ref = float(m[step]["val_ppl"]), float(big[step]["val_ppl"])
    rel = p / ref - 1
    verdict = ("trains B-16M as well as Adam" if abs(rel) <= 0.01 else "better than Adam" if rel < 0 else
               "a bit worse" if rel <= 0.03 else "worse")
    steps = [s for s in sorted(m) if s and s in big]
    tok = {s: [float(m[s]["train_tok_s"]), float(big[s]["train_tok_s"])] for s in steps}
    i = info(name)
    out = load()
    out["part_b"] = {
        "run": name, "step": step, "val_ppl": p, "big_adam_val_ppl": ref, "rel": rel, "verdict": verdict,
        "curve": {s: [float(m[s]["val_ppl"]), float(big[s]["val_ppl"])] for s in steps},
        "train_tok_s_vs_big": tok,
        "speed_ratio_mean": sum(a / b for a, b in tok.values()) / len(tok),
        "train_time_s": [float(m[step]["train_time_s"]), float(big[step]["train_time_s"])],
        "wall_time_s": [float(m[step]["wall_time_s"]), float(big[step]["wall_time_s"])],
        "peak_vram_gib": float(m[step]["peak_vram_gib"]), "peak_ram_bytes": peak,
        "val2_ppl": i.get("results", {}).get("val2_ppl"), "commit": i.get("git", {}).get("commit"),
        "dirty": i.get("git", {}).get("dirty")}
    write_json(OUT, out, indent=1)
    print(json.dumps(out["part_b"], indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["a"]:
        part_a()
    elif len(a) in (2, 4) and a[0] == "b":
        part_b(a[1], int(a[3]) if len(a) == 4 and a[2] == "--peak" else None)
    else:
        sys.exit(__doc__)
