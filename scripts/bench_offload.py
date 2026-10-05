"""Step 2: B-16M at home with the value table in GPU memory (a), in RAM (b) or as a 4-bit file on the NVMe (c).

  python scripts/bench_offload.py --variant a-bf16|a-q4|b-bf16|b-q4|b-fp32|c [--cache_frac 0.3] [--fifo_frac 0]
         [--ppl subset|full] [--out report/offload/<name>.json]

One variant per process (clean memory numbers). Files from scripts/convert_table.py in --tables.
Measured: load time; decode (batch 1, greedy, 128-token prompt + 256 new tokens; three different prompts, the
first one right after start, i.e. cold for c); prefill (4 x 1024 tokens of validation text, 3 different batches
after one warm-up batch); validation PPL (subset: first 64 windows of 1024, identical for every variant; full:
whole Wikipedia validation set); VRAM (PyTorch peak, rocm-smi), RAM (RSS, page cache of the 4-bit file via
fincore, cgroup memory if run in a limited scope), NVMe reads (/sys/block/<dev>/stat), cache hit rate (c).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.data import load_meta, load_split  # noqa: E402
from smlm.train import evaluate  # noqa: E402
from smlm.offload import HostTable, MmapQ4Table, load_model  # noqa: E402

SEQ = 1024


def nvme_dev(path):
    src = subprocess.run(["findmnt", "-n", "-o", "SOURCE", "-T", path], capture_output=True, text=True).stdout.strip()
    dev = os.path.basename(re.sub(r"\[.*\]$", "", src))
    m = re.match(r"(nvme\d+n\d+)", dev)
    return m.group(1) if m else dev


def disk_stat(dev):
    f = open(f"/sys/block/{dev}/stat").read().split()
    return {"reads": int(f[0]), "sectors": int(f[2])}


def proc_mem():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM")):
            out[line.split(":")[0]] = int(line.split()[1]) / 2**20          # GiB
    cg = open("/proc/self/cgroup").read().strip().split("::")[-1]
    base = "/sys/fs/cgroup" + cg
    try:
        out["cgroup_max_gib"] = open(base + "/memory.max").read().strip()
        out["cgroup_current_gib"] = int(open(base + "/memory.current").read()) / 2**30
        stat = dict(line.split() for line in open(base + "/memory.stat"))
        out["cgroup_file_gib"] = int(stat["file"]) / 2**30
    except OSError:
        pass
    return out


def page_cache_gib(path):
    r = subprocess.run(["fincore", "--bytes", "--noheadings", "-o", "RES", path], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0]) / 2**30
    except (IndexError, ValueError):
        return None


def vram_used_gib():
    r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
    used = re.search(r"Total Used Memory \(B\):\s*(\d+)", r.stdout)
    total = re.search(r"Total Memory \(B\):\s*(\d+)", r.stdout)
    return (int(used.group(1)) / 2**30 if used else None, int(total.group(1)) / 2**30 if total else None)


class Phase:
    def __init__(self, name, res, table, dev, qfile):
        self.name, self.res, self.table, self.dev, self.qfile = name, res, table, dev, qfile

    def __enter__(self):
        torch.cuda.synchronize()
        if hasattr(self.table, "reset_stats"):
            self.table.reset_stats()
        torch.cuda.reset_peak_memory_stats()
        self.d0 = disk_stat(self.dev)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t0
        d1 = disk_stat(self.dev)
        out = {"seconds": dt, "nvme_reads": d1["reads"] - self.d0["reads"],
               "nvme_read_mib": (d1["sectors"] - self.d0["sectors"]) * 512 / 2**20,
               "peak_vram_alloc_gib": torch.cuda.max_memory_allocated() / 2**30, **proc_mem()}
        out["nvme_reads_per_s"] = out["nvme_reads"] / dt
        if self.qfile:
            out["q4_file_in_page_cache_gib"] = page_cache_gib(self.qfile)
        st = getattr(self.table, "stats", None)
        if st:
            out["table"] = dict(st)
            if st["hits"] + st["misses"]:
                out["cache_hit_rate"] = st["hits"] / (st["hits"] + st["misses"])
        self.res[self.name] = {**self.res.get(self.name, {}), **out}


@torch.no_grad()
def decode(model, prompt, new_tokens):
    caches = [dict() for _ in range(model.cfg.n_layers)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(prompt, kv_caches=caches, pos0=0)
        nxt = logits[:, -1:].argmax(-1)
        out = []
        for i in range(new_tokens):
            logits = model(nxt, kv_caches=caches, pos0=prompt.shape[1] + i)
            nxt = logits[:, -1:].argmax(-1)
            out.append(int(nxt))
    return out


@torch.no_grad()
def nll(model, val, windows, batch=1):
    total, count = 0.0, 0
    for a in range(0, len(windows), batch):
        w = windows[a:a + batch]
        x = torch.stack([val[i * SEQ:(i + 1) * SEQ] for i in w]).cuda()
        y = torch.stack([val[i * SEQ + 1:(i + 1) * SEQ + 1] for i in w]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        total += float(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))
        count += y.numel()
    return total, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["a-bf16", "a-q4", "b-bf16", "b-q4", "b-fp32", "c"])
    ap.add_argument("--tables", default=os.environ.get("SMLM_TABLES", os.path.join(ROOT, "data", "tables", "B-16M")))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "cloud", "B-16M-s0", "model.pt"))
    ap.add_argument("--cache_frac", type=float, default=0.3)
    ap.add_argument("--fifo_frac", type=float, default=0.0)
    ap.add_argument("--cold", type=int, default=1, help="c: drop the file from the page cache before measuring")
    ap.add_argument("--ppl", default="subset", help="comma list of none / subset / full")
    ap.add_argument("--tag", default="", help="suffix of the result name (e.g. the memory limit of the scope)")
    ap.add_argument("--graphs", type=int, default=1, help="a: decode graphs (b/c: never possible)")
    ap.add_argument("--min_free_vram_gib", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    name = args.variant + (f"-cache{args.cache_frac:g}-fifo{args.fifo_frac:g}" + ("-cold" if args.cold else "")
                           if args.variant == "c" else "") + ("-nographs" if args.variant[0] == "a" and not args.graphs
                                                              else "") + (f"-{args.tag}" if args.tag else "")
    out_path = args.out or os.path.join(ROOT, "report", "offload", name + ".json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = json.load(open(os.path.join(args.tables, "meta.json")))
    rows, dim = meta["rows"], meta["dim"]
    qfile = os.path.join(args.tables, "values_q4.bin")
    sfile = os.path.join(args.tables, "scales_q4.bin")
    dev = nvme_dev(args.tables)
    res = {"variant": args.variant, "name": name, "args": vars(args), "nvme_device": dev,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.set_num_threads(min(16, os.cpu_count()))

    t0 = time.perf_counter()
    table = None
    if args.variant == "a-bf16":
        t = torch.from_numpy(np.fromfile(os.path.join(args.tables, "values_bf16.bin"), dtype=np.int16))
        table = (t.view(torch.bfloat16).view(rows, dim).cuda(), None)
        del t
    elif args.variant == "a-q4":
        p = torch.from_numpy(np.fromfile(qfile, dtype=np.uint8)).view(rows, dim // 2)
        s = torch.from_numpy(np.fromfile(sfile, dtype=np.float16))
        table = (p.cuda(), s.cuda())
        del p, s
    elif args.variant == "b-bf16":
        t = torch.from_numpy(np.fromfile(os.path.join(args.tables, "values_bf16.bin"), dtype=np.int16))
        table = HostTable(t.view(torch.bfloat16).view(rows, dim))
    elif args.variant == "b-q4":
        p = torch.from_numpy(np.fromfile(qfile, dtype=np.uint8)).view(rows, dim // 2)
        table = HostTable(p, torch.from_numpy(np.fromfile(sfile, dtype=np.float16)))
    elif args.variant == "b-fp32":
        ck = torch.load(args.ckpt, mmap=True, map_location="cpu", weights_only=True)
        w = ck["state_dict"][meta["table_keys"][0]]
        table = HostTable(w.clone())                       # read the whole fp32 table into RAM
        del ck, w
    else:
        hot = np.load(os.path.join(args.tables, "hot_rows.npy"), mmap_mode="r")
        table = MmapQ4Table(qfile, sfile, rows, dim, hot_rows=np.asarray(hot[:int(args.cache_frac * rows)]),
                            cache_rows=int(args.cache_frac * rows), fifo_rows=int(args.fifo_frac * rows))
    model = load_model(os.path.join(args.tables, "rest.pt"), table)
    torch.cuda.synchronize()
    res["load_s"] = time.perf_counter() - t0
    used, total = vram_used_gib()
    res["after_load"] = {"vram_alloc_gib": torch.cuda.memory_allocated() / 2**30, "vram_used_total_gib": used,
                         "vram_total_gib": total, **proc_mem()}
    if used is not None and total - used < args.min_free_vram_gib:
        res["aborted"] = f"only {total - used:.2f} GiB VRAM free after loading (< {args.min_free_vram_gib} GiB rule)"
        json.dump(res, open(out_path, "w"), indent=1)
        print(res["aborted"])
        return
    if args.variant.startswith("a") and args.graphs:
        model.set_memory_decode_graphs(True)

    val = torch.from_numpy(np.asarray(load_split("validation", "wikipedia"), dtype=np.int64))
    n_win = (val.numel() - 1) // SEQ
    # warm-up (Triton compilation) on text that is not used below
    decode(model, val[-600:-472].cuda()[None], 8)
    nll(model, val, [n_win - 1])
    if args.variant == "c" and args.cold:
        table.drop_os_cache()
        res["page_cache_after_drop_gib"] = page_cache_gib(qfile)

    for k, start in enumerate([200_000, 400_000, 600_000]):               # three different prompts
        prompt = val[start:start + 128].cuda()[None]
        with Phase(f"decode_{k + 1}", res, table, dev, qfile):
            decode(model, prompt, 256)
        r = res[f"decode_{k + 1}"]
        r["ms_per_token"] = r["seconds"] / (256 + 1) * 1000          # 256 new tokens + the prompt pass
        r["tok_s"] = 256 / r["seconds"]
    for k, first in enumerate([300, 500, 700, 900]):                     # 4 x 1024 tokens each; first = warm-up
        with Phase(f"prefill_{k}", res, table, dev, qfile):
            nll(model, val, list(range(first, first + 4)), batch=4)
        res[f"prefill_{k}"]["tok_s"] = 4 * SEQ / res[f"prefill_{k}"]["seconds"]
    for kind in args.ppl.split(","):
        if kind == "none":
            continue
        if kind == "subset":
            with Phase("ppl", res, table, dev, qfile):
                s, c = nll(model, val, list(range(64)))
            res["ppl"].update({"windows": 64, "tokens": c, "nll_sum": s, "ppl": math.exp(s / c)})
        else:                       # exactly as at the end of training (smlm.train.evaluate), batch 1
            meta_w = load_meta("wikipedia")["splits"]["validation"]["n_words"]
            with Phase("ppl_full", res, table, dev, qfile):
                ev = evaluate(model, "validation", SEQ, meta_w, batch=1, dataset="wikipedia")
            res["ppl_full"].update({"tokens": ev["n_tokens"], "ppl": ev["ppl"], "loss": ev["loss"]})
    used, total = vram_used_gib()
    res["end"] = {"vram_used_total_gib": used, **proc_mem()}
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in (
        "tok_s", "ms_per_token", "ppl", "cache_hit_rate", "nvme_reads_per_s", "seconds", "VmHWM")})
        for k, v in res.items() if k not in ("args",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
