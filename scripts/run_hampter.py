"""Hampter queue (stage 1c): four runs back to back, no intervention needed.

  python scripts/run_hampter.py            (resumable: runs whose run-info.json says 'done' are skipped)

  1. B-1M-sparse s0   500 M Wikipedia tokens (data of the stage-1b quick test). Abort rule: if the val PPL at
                      the evaluation nearest 100 M tokens is > 5 % above runs/s1b/B-1M-s0 at the same point,
                      train.py stops with exit code 3 and the whole queue stops.
  2. A-eqtime s0      A with the training time of run 1: tokens = train_time_s(run 1) x measured A throughput
                      (runs/s1b/A-s0: total_tokens / train_time_s), own cosine schedule over that length,
                      1.5 B-token Wikipedia stream (same validation set; first 505 M tokens = the 500 M data)
  3. A s1             as runs/s1b/A-s0 with init seed 1
  4. B-1M-sparse s1   as run 1 with init seed 1 (no abort rule)

Every run: GPU temperatures (edge, junction/hotspot, memory), power, clocks, fan and VRAM every 10 s in
<run>/gpu_thermal.csv. After every run scripts/hampter_status.py refreshes the status block in REPORT.de.md.
Queue log: runs/hampter/queue.log.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
OUT = os.path.join(ROOT, "runs", "hampter")
REF_B = os.path.join(ROOT, "runs", "s1b", "B-1M-s0")
REF_A = os.path.join(ROOT, "runs", "s1b", "A-s0")
COMMON = ["--extra_val", "wikitext103", "--eval_every_tokens", "10e6"]
B_ARGS = ["--model", "B-1M-sparse", "--value_lr", "2.4e-3", "--data", "wikipedia", "--tokens", "500e6"]
A_ARGS = ["--model", "A", "--data", "wikipedia", "--tokens", "500e6"]
EXIT_ABORT = 3


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "queue.log"), "a") as f:
        f.write(line + "\n")


def hwmon():
    for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
        try:
            if open(os.path.join(h, "name")).read().strip() == "amdgpu":
                return h
        except OSError:
            pass
    return None


def thermal_logger(path, stop, every=10.0):
    """edge / junction (hotspot) / memory temperature, power, shader + memory clock, fan, VRAM used."""
    h = hwmon()
    dev = os.path.dirname(h) if h else None
    if dev and dev.endswith("hwmon"):
        dev = os.path.dirname(dev)

    def rd(p, div=1.0):
        try:
            with open(p) as f:
                return round(int(f.read()) / div, 2)
        except Exception:
            return ""
    t0 = time.time()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "t_s", "edge_c", "junction_c", "mem_c", "power_w", "sclk_mhz", "mclk_mhz", "fan_rpm",
                    "vram_used_gib"])
        while True:
            if h:
                w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), round(time.time() - t0, 1),
                            rd(f"{h}/temp1_input", 1000), rd(f"{h}/temp2_input", 1000), rd(f"{h}/temp3_input", 1000),
                            rd(f"{h}/power1_average", 1e6), rd(f"{h}/freq1_input", 1e6), rd(f"{h}/freq2_input", 1e6),
                            rd(f"{h}/fan1_input"), rd(f"{dev}/mem_info_vram_used", 2**30)])
                f.flush()
            if stop.wait(every):
                break


def status(name):
    p = os.path.join(OUT, name, "run-info.json")
    return json.load(open(p)).get("status") if os.path.exists(p) else None


def run(name, seed, args):
    out = os.path.join(OUT, name)
    if status(name) == "done":
        log(f"skip {name} (done)")
        return 0
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-m", "smlm.train", *args, "--seed", str(seed), "--out_dir", out, *COMMON]
    log("start " + name + ": " + " ".join(cmd[1:]))
    stop = threading.Event()
    th = threading.Thread(target=thermal_logger, args=(os.path.join(out, "gpu_thermal.csv"), stop), daemon=True)
    th.start()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(os.path.join(out, "stdout.log"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    stop.set()
    th.join()
    log(f"end {name} rc={rc} status={status(name)}")
    subprocess.run([PY, os.path.join(ROOT, "scripts", "hampter_status.py")], cwd=ROOT)
    return rc


def eqtime_tokens():
    """A's token budget for the training time of B-1M-sparse s0 (evaluations excluded on both sides)."""
    b = json.load(open(os.path.join(OUT, "B-1M-sparse-s0", "run-info.json")))
    a = json.load(open(os.path.join(REF_A, "run-info.json")))
    a_rate = a["train_config"]["total_tokens"] / a["train_time_s"]
    tps = a["train_config"]["tokens_per_step"]
    steps = int(b["train_time_s"] * a_rate) // tps
    budget = {"b_sparse_s0_train_time_s": b["train_time_s"], "b_sparse_s0_tok_s": b["train_config"]["total_tokens"]
              / b["train_time_s"], "a_ref_run": os.path.relpath(REF_A, ROOT), "a_tok_s": a_rate, "steps": steps,
              "tokens": steps * tps}
    meta = json.load(open(os.path.join(ROOT, "data", "wikipedia_en_gpt2_1500m", "meta.json")))
    budget["dataset_tokens"] = meta["splits"]["train"]["n_tokens"]
    budget["fits_in_one_pass"] = budget["tokens"] < budget["dataset_tokens"] - tps
    with open(os.path.join(OUT, "a_eqtime_budget.json"), "w") as f:
        json.dump(budget, f, indent=2)
    return budget


def main():
    os.makedirs(OUT, exist_ok=True)
    log("queue start")
    rc = run("B-1M-sparse-s0", 0, [*B_ARGS, "--abort_ref", os.path.join(REF_B, "metrics.csv"),
                                    "--abort_at_tokens", "100e6", "--abort_max_rel", "0.05"])
    if rc == EXIT_ABORT:
        log("QUEUE STOPPED: abort rule triggered for B-1M-sparse-s0 (see run-info.json abort_check)")
        return
    if status("B-1M-sparse-s0") != "done":
        log(f"QUEUE STOPPED: B-1M-sparse-s0 failed (rc={rc}); A-eqtime needs its training time")
        return
    budget = eqtime_tokens()
    log(f"A-eqtime budget: {budget['tokens'] / 1e6:.1f} M tokens ({budget['steps']} steps) = "
        f"{budget['b_sparse_s0_train_time_s']:.0f} s x {budget['a_tok_s']:.0f} tok/s"
        + ("" if budget["fits_in_one_pass"] else " WARNING: more than the 1.5 B-token stream, data repeats"))
    run("A-eqtime-s0", 0, ["--model", "A", "--data", "wikipedia_1500m", "--tokens", str(budget["tokens"])])
    run("A-s1", 1, A_ARGS)
    run("B-1M-sparse-s1", 1, B_ARGS)
    log("QUEUE DONE")


if __name__ == "__main__":
    main()
