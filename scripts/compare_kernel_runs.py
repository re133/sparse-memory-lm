"""Kernel check: 20 M-token run with the PyTorch reference vs the same run with the Triton kernels (same init,
same data, same schedule). Writes report/kernel_check.json and report/kernel_check.png.

  python scripts/compare_kernel_runs.py runs/kernel_check/torch runs/kernel_check/triton
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_report import INK, INK2  # noqa: E402,F401  (shared plot style)


def main():
    ref_dir, ker_dir = sys.argv[1:3]
    lr, lk = (pd.read_csv(os.path.join(d, "train_log.csv")) for d in (ref_dir, ker_dir))
    mr, mk = (pd.read_csv(os.path.join(d, "metrics.csv")) for d in (ref_dir, ker_dir))
    ir, ik = (json.load(open(os.path.join(d, "run-info.json"))) for d in (ref_dir, ker_dir))
    tl = lr.merge(lk, on="step", suffixes=("_ref", "_ker"))
    rel_loss = (tl.loss_ker / tl.loss_ref - 1).abs()
    vm = mr.merge(mk, on="step", suffixes=("_ref", "_ker"))
    vm = vm[vm.step > 0]
    rel_ppl = vm.val_ppl_ker / vm.val_ppl_ref - 1
    out = {
        "steps": int(tl.step.max()),
        "train_loss_rel_diff_median": float(rel_loss.median()), "train_loss_rel_diff_max": float(rel_loss.max()),
        "train_loss_rel_diff_max_after_100_steps": float(rel_loss[tl.step > 100].max()),
        "val_ppl_ref": vm.val_ppl_ref.round(4).tolist(), "val_ppl_ker": vm.val_ppl_ker.round(4).tolist(),
        "val_ppl_rel_diff": rel_ppl.round(5).tolist(),
        "final_val_ppl_ref": ir["results"]["val_ppl"], "final_val_ppl_ker": ik["results"]["val_ppl"],
        "final_val_ppl_rel_diff": ik["results"]["val_ppl"] / ir["results"]["val_ppl"] - 1,
        "tok_s_median_ref": ir["results"]["train_tok_s_median"], "tok_s_median_ker": ik["results"]["train_tok_s_median"],
        "peak_train_vram_gib_ref": ir["results"]["peak_train_vram_gib"],
        "peak_train_vram_gib_ker": ik["results"]["peak_train_vram_gib"],
        "mem_usage_ref": ir["results"]["mem_val"]["usage"], "mem_usage_ker": ik["results"]["mem_val"]["usage"],
        "git_ref": ir["git"]["commit"][:7] + ("-dirty" if ir["git"]["dirty"] else ""),
        "git_ker": ik["git"]["commit"][:7] + ("-dirty" if ik["git"]["dirty"] else ""),
    }
    with open(os.path.join(ROOT, "report", "kernel_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(tl.tokens_ref / 1e6, tl.loss_ref, color="#2a78d6", label="PyTorch-Referenz")
    a.plot(tl.tokens_ref / 1e6, tl.loss_ker, color="#eb6834", ls="--", label="Triton-Kernels")
    a.set_xlabel("Trainings-Tokens (Mio.)")
    a.set_ylabel("Trainings-Loss")
    a.set_title("Loss-Kurven, B-1M-sparse, Seed 0", loc="left")
    a.legend()
    b.plot(vm.tokens_ref / 1e6, 100 * rel_ppl, color="#eb6834", marker="o")
    b.axhline(0, color="#52514e", lw=1)
    b.set_xlabel("Trainings-Tokens (Mio.)")
    b.set_ylabel("Val-PPL Kernel / Referenz − 1 (%)")
    b.set_title("Abstand der Val-PPL", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "report", "kernel_check.png"), dpi=150)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)}, indent=1))
    print("val_ppl_rel_diff", [round(100 * x, 3) for x in out["val_ppl_rel_diff"]])


if __name__ == "__main__":
    main()
