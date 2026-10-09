"""Step 9 evaluation (criteria fixed in REPORT.md, "Step 9", before the runs): B-1M's table read with 4x fewer values
per token, as narrower rows (B-4M-v96) or fewer rows (B-1M-k8), against B-1M-sparse s0/s1 (runs/hampter).

  python scripts/shape_eval.py     -> report/shape/summary.json, markdown on stdout

  * each variant against the B-1M mean, r = PPL / B: "better" r < 0.99, "as good" 0.99 <= r <= 1.01, "small cost"
    1.01 < r <= 1.03, "clearly worse" r > 1.03
  * the two against each other: "narrower rows beat fewer rows" PPL(v96) <= 0.99 PPL(k8), "fewer rows beat narrower
    rows" the other way round, otherwise "no clear difference"
  * addendum (B-1M-k16): the same scale; its share of the k8 cost (k16 - B) / (k8 - B) without a verdict
  * reported without a verdict: tokens/s, peak GPU memory, WikiText-103, equivalent dense size
    (scripts/dense_equiv.py), rows and values read per token, the course over training
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dense_equiv import DENSE, fit, load  # noqa: E402
from engram_eval import n_eq  # noqa: E402
from smlm.atomic import write_json  # noqa: E402

VARIANTS = {"B-4M-v96-s0": "runs/shape/B-4M-v96-s0", "B-1M-k8-s0": "runs/shape/B-1M-k8-s0",
            "B-1M-k16-s0": "runs/shape/B-1M-k16-s0"}                 # addendum to step 9
B = {"B-1M-sparse-s0": "runs/hampter/B-1M-sparse-s0", "B-1M-sparse-s1": "runs/hampter/B-1M-sparse-s1"}
OUT = os.path.join(ROOT, "report", "shape")


def info(rel):
    p = os.path.join(ROOT, rel, "run-info.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def verdict(r):
    return ("better" if r < 0.99 else "as good" if r <= 1.01 else "small cost" if r <= 1.03 else "clearly worse")


def main():
    res = {}
    for name, rel in {**VARIANTS, **B}.items():
        i = info(rel)
        if i.get("status") != "done":
            print(f"{name}: not done ({i.get('status')}), left out")
            continue
        r, c = i["results"], i["model_config"]
        reads = len(c["mem_layers"]) * c["mem_heads"] * c["mem_knn"]
        v_dim = c["mem_v_dim"] if c["mem_v_dim"] > 0 else c["d_model"]
        res[name] = {"run": rel, "val_ppl": r["val_ppl"], "wikitext103_ppl": r["val2_ppl"],
                     "train_tok_s_median": r["train_tok_s_median"], "peak_train_vram_gib": r["peak_train_vram_gib"],
                     "table_params": i["params"].get("memory_values", 0),
                     "params_outside_table": i["params"]["total"] - i["params"].get("memory_values", 0),
                     "rows_read_per_token": reads, "values_read_per_token": reads * v_dim,
                     "wall_h": i.get("duration_s", 0) / 3600}
    b = float(np.mean([res[k]["val_ppl"] for k in B]))
    out = {"runs": res, "B_1M_mean": b, "per_variant": {}}
    dense = [d for d in (load(rel) for _, rel in DENSE) if d]
    pts = sorted((d["n"], d["ppl"]) for d in dense)
    f = fit(pts)
    for name in [k for k in VARIANTS if k in res]:
        r = res[name]["val_ppl"] / b
        out["per_variant"][name] = {"ratio_over_B": r, "verdict": verdict(r), "n_eq": n_eq(res[name]["val_ppl"], pts, f)}
    out["B_1M_mean_n_eq"] = n_eq(b, pts, f)
    if "B-1M-k16-s0" in res and "B-1M-k8-s0" in res:          # addendum: share of the k8 cost, no verdict
        out["k16_share_of_k8_cost"] = (res["B-1M-k16-s0"]["val_ppl"] - b) / (res["B-1M-k8-s0"]["val_ppl"] - b)
    if "B-4M-v96-s0" in res and "B-1M-k8-s0" in res:
        v, k = res["B-4M-v96-s0"]["val_ppl"], res["B-1M-k8-s0"]["val_ppl"]
        out["head_to_head"] = {"ratio_v96_over_k8": v / k, "verdict": (
            "narrower rows beat fewer rows" if v <= 0.99 * k else
            "fewer rows beat narrower rows" if k <= 0.99 * v else "no clear difference")}
    curves = {n: np.genfromtxt(os.path.join(ROOT, x["run"], "metrics.csv"), delimiter=",", names=True)
              for n, x in res.items()}
    out["course"] = {}
    for t in (50e6, 100e6, 200e6, 300e6, 400e6, 500e6):
        row = {}
        for n, m in curves.items():
            i = int(np.argmin(abs(m["tokens"] - t)))
            assert abs(m["tokens"][i] - t) < 6e6, (n, t)
            row[n] = float(m["val_ppl"][i])
        out["course"][f"{t / 1e6:.0f}M"] = row
    os.makedirs(OUT, exist_ok=True)
    write_json(os.path.join(OUT, "summary.json"), out)

    print("| Run | Val PPL | WikiText-103 | rows / values read per token | tok/s | peak GiB | wall h |\n"
          "|---|---:|---:|---:|---:|---:|---:|")
    for n, x in res.items():
        print(f"| {n} | {x['val_ppl']:.3f} | {x['wikitext103_ppl']:.2f} | {x['rows_read_per_token']} / "
              f"{x['values_read_per_token']:,} | {x['train_tok_s_median']:,.0f} | {x['peak_train_vram_gib']:.2f} | "
              f"{x['wall_h']:.2f} |")
    print(f"\nB-1M mean {b:.3f}, n_eq {out['B_1M_mean_n_eq']}")
    for n, x in out["per_variant"].items():
        print(n, f"r = {x['ratio_over_B']:.4f} -> {x['verdict']}", "n_eq", x["n_eq"])
    print("head to head:", out.get("head_to_head"))
    print("k16 share of the k8 cost:", out.get("k16_share_of_k8_cost"))
    print("\n| Tokens | " + " | ".join(res) + " |\n|---|" + "---:|" * len(res))
    for t, row in out["course"].items():
        print(f"| {t} | " + " | ".join(f"{v:.2f}" for v in row.values()) + " |")


if __name__ == "__main__":
    main()
