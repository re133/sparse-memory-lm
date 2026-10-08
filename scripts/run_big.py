"""Night run at home (RX 9070 + 128 GB RAM): B-16M with its table in host RAM, to 100 M tokens (REPORT.md, "Step 8").

  python scripts/run_big.py                (as a systemd unit with a memory cap, see REPORT.md)
  python scripts/run_big.py --smoke DIR    (same run over one evaluation interval into DIR, no ntfy)

Exactly the arguments of the H200 run runs/cloud/B-16M-s0 (schedule over 500 M tokens), plus --value_device host
--value_state fp32 (same math), --stop_after_tokens 99.9e6 (stops right after the evaluation at step 3050) and --no_save
(no 26 GB checkpoint on the SSD). Output runs/big_home/B-16M-host-s0/ with GPU temperature / power every 10 s. The run
counts as done with run-info.json status "done". A run whose logs don't move for STALL_MIN minutes is stopped and
reported. ntfy message at the end if NTFY_TOPIC is set (or in ~/smlm-cloud-kit/cloud.env). Nothing is pushed.
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.gpu_monitor import log_until  # noqa: E402

OUT = os.path.join(ROOT, "runs", "big_home")
STALL_MIN = 30
NAME = "B-16M-host-s0"
ARGS = ["--model", "B-16M-sparse", "--value_device", "host", "--value_state", "fp32", "--mem_impl", "triton",
        "--seed", "0", "--data", "wikipedia", "--tokens", "500e6", "--stop_after_tokens", "99.9e6",
        "--extra_val", "wikitext103", "--eval_every_tokens", "10e6", "--data_seed", "1234", "--seq_len", "1024",
        "--batch_seqs", "32", "--lr", "6e-4", "--value_lr", "2.4e-3", "--weight_decay", "0.1", "--no_save"]
H200_STEP_3050 = 37.360


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "queue.log"), "a") as f:
        f.write(line + "\n")


def ntfy_topic():
    topic = os.environ.get("NTFY_TOPIC")
    kit = os.path.expanduser("~/smlm-cloud-kit/cloud.env")
    if not topic and os.path.exists(kit):
        for line in open(kit):
            if line.startswith("NTFY_TOPIC="):
                topic = line.split("=", 1)[1].strip().strip('"')
    return topic


def notify(msg, smoke):
    topic = None if smoke else ntfy_topic()
    if topic:
        subprocess.run(["curl", "-s", "-m", "20", "-H", "Title: SMLM BIG", "-d", msg, f"https://ntfy.sh/{topic}"],
                       capture_output=True)
    log("ntfy: " + msg)


def val_at(d, step):
    try:
        import csv
        for row in csv.DictReader(open(os.path.join(d, "metrics.csv"))):
            if int(row["step"]) == step:
                return float(row["val_ppl"])
    except (OSError, ValueError, KeyError):
        pass
    return None


def main():
    global OUT
    smoke = len(sys.argv) == 3 and sys.argv[1] == "--smoke"
    args = ARGS
    if smoke:
        OUT = os.path.abspath(sys.argv[2])
        args = [("1e6" if a == "99.9e6" else a) for a in ARGS]          # stops after the first evaluation
    d = os.path.join(OUT, NAME)
    os.makedirs(d, exist_ok=True)
    cmd = [sys.executable, "-m", "smlm.train", "--out_dir", d, *args]
    log("start " + " ".join(cmd[1:]))
    stop = threading.Event()
    th = threading.Thread(target=log_until, args=(os.path.join(d, "gpu_thermal.csv"), stop), daemon=True)
    th.start()
    t0, result = time.time(), None
    with open(os.path.join(d, "stdout.log"), "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        while p.poll() is None:
            time.sleep(30)
            newest = max(os.path.getmtime(os.path.join(d, f)) for f in ("stdout.log", "train_log.csv", "metrics.csv")
                         if os.path.exists(os.path.join(d, f)))
            if time.time() - newest > STALL_MIN * 60:
                result = "stalled"
                log(f"{NAME}: no progress for {STALL_MIN} min, stopping it")
                os.killpg(p.pid, 15)
                p.wait(120)
    stop.set()
    th.join()
    try:
        status = json.load(open(os.path.join(d, "run-info.json"))).get("status")
    except (OSError, ValueError):
        status = None
    ok = p.returncode == 0 and status == "done"
    ppl = val_at(d, 3050)
    hours = (time.time() - t0) / 3600
    log(f"end {NAME} rc={p.returncode} status={status} {result or ''} ({hours:.2f} h) val_ppl@3050 {ppl}")
    msg = f"{NAME} {'fertig' if ok else 'FEHLGESCHLAGEN (' + (result or f'rc={p.returncode}') + ')'}, {hours:.1f} h"
    if ppl:
        msg += f", Val-PPL Schritt 3050 {ppl:.3f} (H200 {H200_STEP_3050:.3f}, {100 * (ppl / H200_STEP_3050 - 1):+.2f} %)"
    notify(msg, smoke)


if __name__ == "__main__":
    main()
