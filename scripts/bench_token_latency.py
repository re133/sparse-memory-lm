"""Per-token latency of B-16M while writing, with the table on the NVMe (cold page cache), in RAM or in VRAM.

  python scripts/bench_token_latency.py --table nvme --cache_frac 0 [--tag ...]
  systemd-run --user --scope -p MemoryMax=4G python scripts/bench_token_latency.py --table nvme --tag mem4G

Same prompts as bench_offload.py (three 128-token pieces of Wikipedia validation text, greedy, 256 new tokens each).
For nvme the table file is dropped from the page cache right before the first prompt, so prompt 1 starts cold and
prompts 2 and 3 run with whatever the earlier ones left in the cache. Every token is timed on its own (the
offloaded lookup syncs with the CPU anyway), together with the NVMe reads it caused (/sys/block/<dev>/stat).
Result: report/offload/token_latency/<name>.json with all per-token times and p50 / p90 / p99 / max per prompt.
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
from bench_offload import disk_stat, nvme_dev, page_cache_gib, proc_mem  # noqa: E402
from demo_generate import load  # noqa: E402
from smlm.data import load_split  # noqa: E402

PROMPT, NEW = 128, 256


@torch.no_grad()
def decode_timed(model, prompt, dev):
    """Prompt pass, then NEW greedy tokens; returns the prompt-pass time and per-token ms and NVMe reads."""
    caches = [dict() for _ in range(model.cfg.n_layers)]
    ms, reads = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize()
        d0, t0 = disk_stat(dev)["reads"], time.perf_counter()
        logits = model(prompt, kv_caches=caches, pos0=0)
        nxt = int(logits[0, -1].argmax())
        first_ms, first_reads = 1000 * (time.perf_counter() - t0), disk_stat(dev)["reads"] - d0
        for i in range(NEW):
            d0, t0 = disk_stat(dev)["reads"], time.perf_counter()
            logits = model(torch.tensor([[nxt]], device="cuda"), kv_caches=caches, pos0=prompt.shape[1] + i)
            nxt = int(logits[0, -1].argmax())                      # waits for the GPU
            ms.append(1000 * (time.perf_counter() - t0))
            reads.append(disk_stat(dev)["reads"] - d0)
    return first_ms, first_reads, ms, reads


def summary(ms):
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p90_ms": float(np.percentile(a, 90)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "tok_s": float(1000 / a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M")))
    ap.add_argument("--table", required=True, choices=["nvme", "ram", "vram"])
    ap.add_argument("--cache_frac", type=float, default=0.3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    name = args.table + (f"-cache{args.cache_frac:g}" if args.table == "nvme" else "") + (f"-{args.tag}" if args.tag
                                                                                        else "")
    out_path = os.path.join(ROOT, "report", "offload", "token_latency", name + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    qfile = os.path.join(args.tables, "values_q4.bin")
    dev = nvme_dev(args.tables)
    torch.set_num_threads(min(16, os.cpu_count()))
    model = load(args.tables, args.table, args.cache_frac)
    val = torch.from_numpy(np.asarray(load_split("validation", "wikipedia"), dtype=np.int64))
    decode_timed(model, val[-600:-472].cuda()[None], dev)                # warm-up (Triton compilation), other text
    table = model.memory_layers()[0].values.infer_table
    res = {"name": name, "table": args.table, "cache_frac": args.cache_frac if args.table == "nvme" else None,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "nvme_device": dev,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "prompts": []}
    if args.table == "nvme":
        table.drop_os_cache()
        res["page_cache_after_drop_gib"] = page_cache_gib(qfile)
    for k, start in enumerate([200_000, 400_000, 600_000]):
        first_ms, first_reads, ms, reads = decode_timed(model, val[start:start + PROMPT].cuda()[None], dev)
        p = {"prompt": k + 1, "prompt_pass_ms": first_ms, "prompt_pass_nvme_reads": first_reads,
             **summary(ms), "nvme_reads_per_token_mean": float(np.mean(reads)),
             "nvme_reads_per_token_p99": float(np.percentile(reads, 99)), "ms": [round(x, 3) for x in ms],
             "nvme_reads": reads}
        if args.table == "nvme":
            p["page_cache_after_gib"] = page_cache_gib(qfile)
        res["prompts"].append(p)
        print(f"{name} prompt {k + 1}: p50 {p['p50_ms']:.2f} ms, p90 {p['p90_ms']:.2f}, p99 {p['p99_ms']:.2f}, "
              f"max {p['max_ms']:.2f}, mean {p['mean_ms']:.2f} ({p['tok_s']:.0f} tok/s), "
              f"{p['nvme_reads_per_token_mean']:.0f} NVMe reads/token, prompt pass {first_ms:.0f} ms", flush=True)
    res["end"] = proc_mem()
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
