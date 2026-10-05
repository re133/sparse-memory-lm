"""Summary of the AMD Instinct check (runs/amd_mi350x, filled by scripts/run_amd.py) next to the other GPUs.

  python scripts/amd_summary.py      -> report/amd_summary.json, report/amd_crosscheck.png, markdown on stdout

Speed: the same 3 M-token runs as the H100 preflight of step 1 (runs/cloud_dense_preflight), each model alone.
Cross-check: B-1M with the Triton kernels, first 20 M tokens of the 500 M schedule, seed 0 - the same run exists on
the RX 9070 (runs/kernel_check/triton_500msched, eval every 2 M) and on the H200 (runs/cloud/B-1M-s0, every 10 M).
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(ROOT, "runs", "amd_mi350x")


def info(path):
    p = os.path.join(path, "run-info.json")
    return json.load(open(p)) if os.path.exists(p) else None


def curve(path):
    p = os.path.join(path, "metrics.csv")
    if not os.path.exists(p):
        return {}
    return {int(r["tokens"]): float(r["val_ppl"]) for r in csv.DictReader(open(p)) if r["val_ppl"]}


def main():
    env = json.load(open(os.path.join(A, "env", "env.json")))
    out = {"env": {k: env.get(k) for k in ("torch", "hip", "triton", "python", "gpu", "arch", "gpu_mem_gib")},
           "speed": {}, "crosscheck": {}}
    tests_log = os.path.join(A, "tests", "stdout.log")
    if os.path.exists(tests_log):
        out["tests_summary"] = [ln.strip() for ln in open(tests_log) if " passed" in ln or " failed" in ln][-1:]
    h100 = {"D-100M": "D-100M-s0", "D-400M": "D-400M-s0"}
    for name in ("A", "D-100M", "D-400M", "B-1M-torch", "B-1M", "B-4M", "B-16M"):
        r = info(os.path.join(A, f"speed_{name}"))
        if not r or "results" not in r:
            out["speed"][name] = {"status": r.get("status") if r else "missing"}
            continue
        x = r["results"]
        row = {"tok_s": x.get("train_tok_s_median"), "peak_gib": x.get("peak_train_vram_gib"),
               "decode_tok_s": x.get("decode_b1_tok_s"), "prefill_tok_s": x.get("prefill_tok_s"),
               "val_ppl_3M": x.get("val_ppl")}
        if name in h100:
            h = info(os.path.join(ROOT, "runs", "cloud_dense_preflight", h100[name]))
            if h:
                row["h100_tok_s"] = h["results"]["train_tok_s_median"]
                row["h100_val_ppl_3M"] = h["results"]["val_ppl"]
        out["speed"][name] = row
    mi = curve(os.path.join(A, "crosscheck"))
    home = curve(os.path.join(ROOT, "runs", "kernel_check", "triton_500msched"))
    h200 = curve(os.path.join(ROOT, "runs", "cloud", "B-1M-s0"))
    for t in sorted(mi):
        out["crosscheck"][t] = {"mi": mi[t], "rx9070": home.get(t), "h200": h200.get(t)}
    json.dump(out, open(os.path.join(ROOT, "report", "amd_summary.json"), "w"), indent=1)
    print("env:", out["env"], "| tests:", out.get("tests_summary"))
    print("| Model | train tok/s | (H100) | peak GiB | decode tok/s | prefill tok/s | val PPL @3M | (H100) |")
    print("|---|---|---|---|---|---|---|---|")
    for k, v in out["speed"].items():
        if "tok_s" not in v:
            print(f"| {k} | {v} |||||||")
            continue
        h_tok = f"{v['h100_tok_s']:.0f}" if "h100_tok_s" in v else ""
        h_ppl = f"{v['h100_val_ppl_3M']:.2f}" if "h100_val_ppl_3M" in v else ""
        print(f"| {k} | {v['tok_s']:.0f} | {h_tok} | {v['peak_gib']:.1f} | {v['decode_tok_s']:.0f} | "
              f"{v['prefill_tok_s']:.0f} | {v['val_ppl_3M']:.2f} | {h_ppl} |")
    print("\n| tokens | MI | RX 9070 | H200 |\n|---|---|---|---|")
    for t, v in out["crosscheck"].items():
        print(f"| {t / 1e6:.1f} M | {v['mi']:.2f} | {v['rx9070'] or '':} | {v['h200'] or ''} |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        names = {"gfx950": "AMD Instinct MI350X", "gfx942": "AMD Instinct MI300X"}
        mi_name = names.get(str(env.get("arch", "")).split(":")[0], env.get("gpu", "AMD Instinct"))
        for c, lab, st in ((mi, mi_name, "o-"), (home, "Radeon RX 9070", "s--"),
                           (h200, "H200", "^:")):
            ts = [t for t in sorted(c) if 0 < t <= 22e6]
            if ts:
                ax.plot([t / 1e6 for t in ts], [c[t] for t in ts], st, label=lab)
        ax.set(xlabel="Tokens (M)", ylabel="Val PPL (Wikipedia)", yscale="log",
               title="B-1M with the Triton kernels, same seed, three GPUs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "report", "amd_crosscheck.png"), dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
