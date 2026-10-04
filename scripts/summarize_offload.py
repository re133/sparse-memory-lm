"""Step 2: table of report/offload/*.json (scripts/bench_offload.py) -> report/offload_summary.json + markdown on stdout,
plus report/offload_cache.png (decode / prefill speed and hit rate against the RAM cache size of variant c)."""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def row(r):
    dec = [r[f"decode_{k}"] for k in (1, 2, 3) if f"decode_{k}" in r]
    pre = [r[f"prefill_{k}"] for k in (1, 2, 3) if f"prefill_{k}" in r]       # prefill_0 = warm-up
    out = {
        "name": r["name"], "variant": r["variant"],
        "cache_frac": r["args"]["cache_frac"] if r["variant"] == "c" else None,
        "fifo_frac": r["args"]["fifo_frac"] if r["variant"] == "c" else None,
        "load_s": r.get("load_s"),
        "decode_tok_s_first": dec[0]["tok_s"] if dec else None,
        "decode_tok_s_median": statistics.median(d["tok_s"] for d in dec) if dec else None,
        "decode_ms_first": dec[0]["ms_per_token"] if dec else None,
        "prefill_tok_s_median": statistics.median(p["tok_s"] for p in pre) if pre else None,
        "prefill0_tok_s": r.get("prefill_0", {}).get("tok_s"),
        "hit_decode_first": dec[0].get("cache_hit_rate") if dec else None,
        "hit_prefill": statistics.median(p["cache_hit_rate"] for p in pre) if pre and "cache_hit_rate" in pre[0] else None,
        "nvme_reads_s_decode_first": dec[0]["nvme_reads_per_s"] if dec else None,
        "nvme_reads_s_prefill": statistics.median(p["nvme_reads_per_s"] for p in pre) if pre else None,
        "nvme_mib_s_prefill": statistics.median(p["nvme_read_mib"] / p["seconds"] for p in pre) if pre else None,
        "vram_alloc_gib_load": r.get("after_load", {}).get("vram_alloc_gib"),
        "vram_used_total_gib_load": r.get("after_load", {}).get("vram_used_total_gib"),
        "peak_vram_alloc_gib": max([p.get("peak_vram_alloc_gib", 0) for p in dec + pre] or [0]),
        "rss_peak_gib": r.get("end", {}).get("VmHWM"),
        "q4_page_cache_gib_end": (r.get("ppl") or pre[-1] if pre else {}).get("q4_file_in_page_cache_gib"),
        "cgroup_max": r.get("after_load", {}).get("cgroup_max_gib"),
        "ppl_subset": r.get("ppl", {}).get("ppl"), "nll_subset": r.get("ppl", {}).get("nll_sum"),
        "ppl_full": r.get("ppl_full", {}).get("ppl"),
        "aborted": r.get("aborted"),
    }
    return out


def main():
    rows = [row(json.load(open(f))) for f in sorted(glob.glob(os.path.join(ROOT, "report", "offload", "*.json")))]
    json.dump(rows, open(os.path.join(ROOT, "report", "offload_summary.json"), "w"), indent=1)
    cols = ["name", "decode_tok_s_first", "decode_tok_s_median", "prefill_tok_s_median", "hit_decode_first",
            "hit_prefill", "nvme_reads_s_prefill", "vram_used_total_gib_load", "rss_peak_gib", "ppl_subset", "ppl_full"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join("" if r[c] is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]))
                                for c in cols) + " |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cs = sorted([r for r in rows if r["variant"] == "c" and not r["fifo_frac"] and "mem" not in r["name"]],
                key=lambda r: r["cache_frac"])
    if not cs:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x = [100 * r["cache_frac"] for r in cs]
    ref = {r["variant"]: r for r in rows if r["variant"] in ("a-q4", "b-q4") and "nographs" not in r["name"]}
    ax[0].plot(x, [r["decode_tok_s_first"] for r in cs], "o-", label="c, erster Durchgang (kalt)")
    ax[0].plot(x, [r["decode_tok_s_median"] for r in cs], "s--", label="c, Median 3 Prompts")
    for k, st in (("a-q4", ":"), ("b-q4", "-.")):
        if k in ref:
            ax[0].axhline(ref[k]["decode_tok_s_median"], ls=st, color="gray", label=f"{k}")
    ax[0].set(xlabel="RAM-Cache (% der Zeilen)", ylabel="Tokens/s", title="Schreiben (Batch 1)", ylim=(0, 240))
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [r["prefill_tok_s_median"] for r in cs], "o-", label="c")
    for k, st in (("a-q4", ":"), ("b-q4", "-.")):
        if k in ref:
            ax[1].axhline(ref[k]["prefill_tok_s_median"], ls=st, color="gray", label=k)
    ax[1].set(xlabel="RAM-Cache (% der Zeilen)", ylabel="Tokens/s", title="Einlesen (4 × 1024)", yscale="log")
    ax[1].legend(fontsize=7)
    ax[2].plot(x, [100 * r["hit_decode_first"] for r in cs], "o-", label="Schreiben, kalt")
    ax[2].plot(x, [100 * r["hit_prefill"] for r in cs], "s--", label="Einlesen")
    ax[2].set(xlabel="RAM-Cache (% der Zeilen)", ylabel="Trefferquote %", title="Cache-Treffer", ylim=(0, 100))
    ax[2].legend(fontsize=7)
    fig.suptitle("B-16M zu Hause: 4-Bit-Tabelle auf der NVMe (Variante c) – RX 9070, Samsung 990 PRO")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "report", "offload_cache.png"), dpi=120)


if __name__ == "__main__":
    main()
