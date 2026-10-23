"""Step 18 (HEIM 2), the verdicts of REPORT.md "Step 18" from the runs in runs/rowopt/ -> report/rowopt/summary.json.

  python scripts/rowopt_summary.py a          part A (B-1M, 50 M tokens): tolerance, best LR per optimizer, verdicts,
                                              the table LR part B uses
  python scripts/rowopt_summary.py b NAME     part B (B-16M, table in RAM, run runs/rowopt/NAME) against step 8
  python scripts/rowopt_summary.py b NAME --peak BYTES   with the peak RAM of the run's systemd unit
  python scripts/rowopt_summary.py b4g NAME   step 20: B-4M on the card (runs/rowopt/NAME) against the H200 run
  python scripts/rowopt_summary.py fast NAME REF [--peak BYTES]   step 21: B-16M with HOT/GACC against part B (REF)
  python scripts/rowopt_summary.py grow NAME REF   step 23: grown B-4M against step 20 (REF) at the same training time
"""
import csv
import json
import math
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


def b4g(name):
    m, h = metrics(os.path.join(RUNS, name)), metrics(os.path.join(ROOT, "runs", "cloud", "B-4M-s0"))
    p, ref = float(m[3050]["val_ppl"]), float(h[3050]["val_ppl"])
    rel = p / ref - 1
    verdict = ("B-4M trains on the gaming card as on the H200" if abs(rel) <= 0.01 else "better" if rel < 0 else
               "a bit worse" if rel <= 0.03 else "worse")
    steps = [k for k in sorted(m) if k and k in h]
    tok = [float(m[k]["train_tok_s"]) for k in steps]
    i = info(name)
    out = load()
    out["b4g"] = {"run": name, "val_ppl": p, "h200_val_ppl": ref, "rel": rel, "verdict": verdict,
                  "curve": {k: [float(m[k]["val_ppl"]), float(h[k]["val_ppl"])] for k in steps},
                  "train_tok_s_mean": sum(tok) / len(tok), "h200_train_tok_s_mean":
                      sum(float(h[k]["train_tok_s"]) for k in steps) / len(steps),
                  "peak_vram_gib": max(float(m[k]["peak_vram_gib"]) for k in steps),
                  "hours_for_500M_at_this_speed": 500e6 / (sum(tok) / len(tok)) / 3600,
                  "wall_time_s": float(m[3050]["wall_time_s"]), "train_config": {
                      k: i.get("train_config", {}).get(k) for k in ("value_optim", "value_lr", "micro_bs", "fused_ce")},
                  "val2_ppl": i.get("results", {}).get("val2_ppl"), "commit": i.get("git", {}).get("commit"),
                  "dirty": i.get("git", {}).get("dirty")}
    write_json(OUT, out, indent=1)
    print(json.dumps(out["b4g"], indent=1))


def fast(name, ref, peak):
    m, r = metrics(os.path.join(RUNS, name)), metrics(os.path.join(RUNS, ref))
    rel = {k: float(m[k]["val_ppl"]) / float(r[k]["val_ppl"]) - 1 for k in (305, 610)}
    verdict = "same training" if all(abs(x) <= 0.01 for x in rel.values()) else "differs"
    i = info(name)
    out = load()
    out["fast"] = {"run": name, "reference": ref, "rel": rel, "verdict": verdict,
                   "val_ppl": {k: [float(m[k]["val_ppl"]), float(r[k]["val_ppl"])] for k in (305, 610)},
                   "train_tok_s": {k: [float(m[k]["train_tok_s"]), float(r[k]["train_tok_s"])] for k in (305, 610)},
                   "speed_ratio": {k: float(m[k]["train_tok_s"]) / float(r[k]["train_tok_s"]) for k in (305, 610)},
                   "peak_vram_gib": float(m[610]["peak_vram_gib"]), "peak_ram_bytes": peak,
                   "step_rows_peak": i.get("step_rows_peak"),
                   "step_rows_spilled_steps": i.get("step_rows_spilled_steps"), "train_config": {
                       k: i.get("train_config", {}).get(k) for k in ("hot_rows", "hot_frac", "step_rows", "fused_ce")},
                   "commit": i.get("git", {}).get("commit"), "dirty": i.get("git", {}).get("dirty")}
    write_json(OUT, out, indent=1)
    print(json.dumps(out["fast"], indent=1))


def grow(name, ref):
    """Step 23: the grown run at the training time the reference needed for 100 M tokens (log PPL interpolated
    linearly in training time between the two evaluations around it)."""
    m, r = metrics(os.path.join(RUNS, name)), metrics(os.path.join(RUNS, ref))
    t_ref, p_ref = float(r[3050]["train_time_s"]), float(r[3050]["val_ppl"])
    steps = [k for k in sorted(m) if k]
    t = [float(m[k]["train_time_s"]) for k in steps]
    j = next((n for n in range(1, len(steps)) if t[n] >= t_ref), None)
    if j is None:
        sys.exit(f"{name} ended at {t[-1]:.0f} s of training, before the reference's {t_ref:.0f} s")
    a, b = steps[j - 1], steps[j]
    f = (t_ref - t[j - 1]) / (t[j] - t[j - 1])

    def at(col):
        return math.exp((1 - f) * math.log(float(m[a][col])) + f * math.log(float(m[b][col])))
    p = at("val_ppl")
    rel = p / p_ref - 1
    verdict = ("growing pays off" if rel < -0.01 else "no difference I can measure" if rel <= 0.01 else
               "growing costs")
    i = info(name)
    g = i.get("grow", {}).get("at_step")
    tok = {k: float(m[k]["train_tok_s"]) for k in steps}
    out = load()
    out["grow"] = {"run": name, "reference": ref, "reference_train_time_s": t_ref, "reference_val_ppl": p_ref,
                   "between_steps": [a, b], "fraction": f, "val_ppl_at_reference_time": p, "rel": rel,
                   "verdict": verdict, "tokens_at_reference_time": (1 - f) * a * 32768 + f * b * 32768,
                   "val2_ppl_at_reference_time": at("val2_ppl"), "reference_val2_ppl": float(r[3050]["val2_ppl"]),
                   "val_ppl_at_100M": float(m[3050]["val_ppl"]), "rel_at_100M": float(m[3050]["val_ppl"]) / p_ref - 1,
                   "train_time_to_100M_s": float(m[3050]["train_time_s"]),
                   "time_saved_to_100M": 1 - float(m[3050]["train_time_s"]) / t_ref,
                   "grow": i.get("grow"),
                   "train_tok_s_small_mean": sum(v for k, v in tok.items() if k <= g) / len([k for k in tok if k <= g]),
                   "train_tok_s_grown_mean": sum(v for k, v in tok.items() if k > g) /
                       len([k for k in tok if k > g]),
                   "reference_train_tok_s_mean": sum(float(r[k]["train_tok_s"]) for k in r if k) / len(
                       [k for k in r if k]),
                   "curve": {k: [float(m[k]["val_ppl"]), float(m[k]["train_time_s"]),
                                 float(r[k]["val_ppl"]) if k in r else None,
                                 float(r[k]["train_time_s"]) if k in r else None] for k in steps},
                   "peak_vram_gib": max(float(m[k]["peak_vram_gib"]) for k in steps),
                   "val2_ppl_end": i.get("results", {}).get("val2_ppl"),
                   "commit": i.get("git", {}).get("commit"), "dirty": i.get("git", {}).get("dirty")}
    write_json(OUT, out, indent=1)
    print(json.dumps({k: v for k, v in out["grow"].items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["a"]:
        part_a()
    elif len(a) in (2, 4) and a[0] == "b":
        part_b(a[1], int(a[3]) if len(a) == 4 and a[2] == "--peak" else None)
    elif len(a) == 2 and a[0] == "b4g":
        b4g(a[1])
    elif len(a) in (3, 5) and a[0] == "fast":
        fast(a[1], a[2], int(a[4]) if len(a) == 5 and a[3] == "--peak" else None)
    elif len(a) == 3 and a[0] == "grow":
        grow(a[1], a[2])
    else:
        sys.exit(__doc__)
