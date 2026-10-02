"""Short throughput / resource measurement of a training configuration (Hampter step 2).

  python scripts/measure_run.py --model B-1M-sparse --tokens 4e6 -- --value_lr 2.4e-3

Runs smlm.train as a subprocess (same code path as a real run, output to a scratch directory) and samples
once per second: GPU busy % and VRAM % (rocm-smi), CPU % and RSS of the training process, system RAM.
Only samples taken while training steps are being logged count (start-up, evaluations and the final
evaluation / saving are excluded). Separately times the data path (memmap gather -> pinned -> GPU) to see
whether the GPU would ever wait for data.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rocm_sample():
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showmemuse"], capture_output=True, text=True, timeout=10).stdout
        use = re.search(r"GPU use \(%\): (\d+)", out)
        mem = re.search(r"GPU Memory Allocated \(VRAM%\): (\d+)", out)
        return (int(use.group(1)) if use else None, int(mem.group(1)) if mem else None)
    except Exception:
        return (None, None)


def proc_sample(pid, last):
    """CPU % (all cores = 100 % per core) and RSS of pid, system used RAM."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
        ticks = int(fields[11]) + int(fields[12])
        with open(f"/proc/{pid}/status") as f:
            rss = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 2**20
        with open("/proc/meminfo") as f:
            mi = {l.split(":")[0]: int(l.split()[1]) for l in f}
        used = (mi["MemTotal"] - mi["MemAvailable"]) / 2**20
        now = time.time()
        cpu = None
        if last:
            cpu = 100 * (ticks - last[0]) / os.sysconf("SC_CLK_TCK") / (now - last[1])
        return cpu, rss, used, (ticks, now)
    except Exception:
        return None, None, None, last


def data_path_ms(dataset, steps=60):
    import torch
    sys.path.insert(0, ROOT)
    from smlm.data import TrainStream
    st = TrainStream(1024, 32, 1234, dataset=dataset)
    times = []
    for s in range(steps):
        t0 = time.perf_counter()
        b = torch.from_numpy(st.batch(1000 + s).astype(np.int64)).pin_memory().cuda(non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times[5:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", default="4e6")
    ap.add_argument("--data", default="wikipedia")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scratch_dir", default=None, help="where the throw-away run directory goes (default: TMPDIR)")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()

    scratch = tempfile.mkdtemp(prefix=f"measure_{args.model}_", dir=args.scratch_dir)
    cmd = [sys.executable, "-m", "smlm.train", "--model", args.model, "--data", args.data, "--tokens", args.tokens,
           "--eval_every_tokens", "1e12", "--sample_windows", "0", "--out_dir", scratch, *args.extra]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(os.path.join(scratch, "stdout.log"), "w")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    samples, last = [], None
    tl = os.path.join(scratch, "train_log.csv")

    def n_logged():
        try:
            with open(tl) as f:
                return sum(1 for _ in f) - 1
        except FileNotFoundError:
            return -1
    while p.poll() is None:
        use, mem = rocm_sample()
        cpu, rss, used, last = proc_sample(p.pid, last)
        samples.append({"t": time.time(), "logged": n_logged(), "gpu_use": use, "vram_pct": mem,
                        "cpu_pct": cpu, "rss_gib": rss, "sys_ram_used_gib": used})
        time.sleep(1.0)
    rc = p.wait()
    info = json.load(open(os.path.join(scratch, "run-info.json")))
    total_logged = n_logged()
    # training phase: after the 2nd log row (warm-up kernels) and before the last one
    train = [s for s in samples if 2 <= s["logged"] < total_logged]
    tok = [float(l.split(",")[-1]) for l in open(tl).read().splitlines()[1:]]
    med = lambda k: float(np.median([s[k] for s in train if s[k] is not None])) if train else None  # noqa: E731
    res = {
        "model": args.model, "rc": rc, "scratch": scratch, "extra_args": args.extra,
        "steps": info["train_config"]["total_steps"], "micro_bs": info["train_config"]["micro_bs"],
        "tok_s_median": float(np.median(tok[2:])) if len(tok) > 2 else None,
        "peak_train_vram_gib": info.get("results", {}).get("peak_train_vram_gib"),
        "gpu_use_pct_median": med("gpu_use"), "gpu_use_pct_min": min((s["gpu_use"] for s in train if s["gpu_use"] is not None), default=None),
        "vram_pct_median": med("vram_pct"), "proc_cpu_pct_median": med("cpu_pct"), "proc_rss_gib_median": med("rss_gib"),
        "sys_ram_used_gib_median": med("sys_ram_used_gib"), "n_samples_training": len(train),
    }
    res["data_path_ms_per_step"] = data_path_ms(args.data)
    res["step_ms"] = 1000 * 32768 / res["tok_s_median"] if res["tok_s_median"] else None
    res["data_share_of_step_pct"] = 100 * res["data_path_ms_per_step"] / res["step_ms"] if res["step_ms"] else None
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
