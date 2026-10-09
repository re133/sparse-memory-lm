"""Night queue at home (RX 9070): B-1M's table read with 4x fewer values per token (REPORT.md, "Step 9").

  python scripts/run_shape.py [--deadline 2026-10-09T11:00]   (as a systemd unit; resumable, a finished run is skipped)
  python scripts/run_shape.py --smoke DIR                       (both runs on 2M tokens into DIR, no ntfy)

Runs one after another, same arguments as B-1M-sparse in runs/hampter (only --model and --mem_impl differ):
  B-4M-v96-s0   2048^2 = 4M rows of 96 values (narrower rows)
  B-1M-k8-s0    B-1M's 1M rows of 384 values, 8 instead of 32 lookups per head (fewer rows)
  B-1M-k16-s0   the same with 16 lookups per head (addendum to step 9)
A run is only started if its estimated end (EST_H, measured in the smoke test) lies before --deadline. Output
runs/shape/<name>/ with GPU temperature / power every 10 s. A run counts as done only with run-info.json status
"done" and model.pt. A run whose logs don't move for STALL_MIN minutes is stopped and reported, the queue goes on.
ntfy message after every run if NTFY_TOPIC is set (or in ~/smlm-cloud-kit/cloud.env). Nothing is pushed.
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.gpu_monitor import log_until  # noqa: E402

OUT = os.path.join(ROOT, "runs", "shape")
STALL_MIN = 30
COMMON = ["--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103", "--eval_every_tokens", "10e6",
          "--data_seed", "1234", "--seq_len", "1024", "--batch_seqs", "32", "--lr", "6e-4", "--weight_decay", "0.1",
          "--value_lr", "2.4e-3", "--seed", "0", "--mem_impl", "triton"]
RUNS = [
    ("B-4M-v96-s0", ["--model", "B-4M-v96-sparse"]),
    ("B-1M-k8-s0", ["--model", "B-1M-k8-sparse"]),
    ("B-1M-k16-s0", ["--model", "B-1M-k16-sparse"]),        # addendum to step 9 (criteria fixed 2026-10-09)
]
EST_H = {"B-4M-v96-s0": 2.8, "B-1M-k8-s0": 2.4,     # wall hours incl. evaluations: 58.4k / 69.4k tok/s in the smoke test
         "B-1M-k16-s0": 2.5}                       # not smoke-tested, between the two
SMOKE = None


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


def notify(msg):
    topic = None if SMOKE else ntfy_topic()
    if topic:
        subprocess.run(["curl", "-s", "-m", "20", "-H", "Title: SMLM Step 9", "-d", msg, f"https://ntfy.sh/{topic}"],
                       capture_output=True)
    log("ntfy: " + msg)


def info(d):
    try:
        return json.load(open(os.path.join(d, "run-info.json")))
    except (OSError, ValueError):
        return {}


def done(d):
    return info(d).get("status") == "done" and os.path.exists(os.path.join(d, "model.pt"))


def run(name, args, deadline):
    d = os.path.join(OUT, name)
    if done(d):
        log(f"skip {name} (done)")
        return "done"
    if deadline and not SMOKE and datetime.now() + timedelta(hours=EST_H[name]) > deadline:
        log(f"skip {name}: estimated end after the deadline {deadline:%H:%M} ({EST_H[name]} h)")
        return "skipped (deadline)"
    os.makedirs(d, exist_ok=True)
    common = COMMON
    if SMOKE:
        common = [("2e6" if a == "500e6" else "1e6" if a == "10e6" else a) for a in COMMON]
    cmd = [sys.executable, "-m", "smlm.train", "--out_dir", d, *args, *common]
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
                log(f"{name}: no progress for {STALL_MIN} min, stopping it")
                os.killpg(p.pid, 15)
                p.wait(120)
    stop.set()
    th.join()
    ok = p.returncode == 0 and done(d)
    ppl = info(d).get("results", {}).get("val_ppl")
    log(f"end {name} rc={p.returncode} {result or ''} ({(time.time() - t0) / 3600:.2f} h)"
        + (f" val_ppl {ppl:.3f}" if ppl else ""))
    notify(f"{name} {'fertig' if ok else 'FEHLGESCHLAGEN (' + (result or f'rc={p.returncode}') + ')'}"
           + (f", Val-PPL {ppl:.3f}" if ppl else "") + f", {(time.time() - t0) / 3600:.1f} h")
    return "done" if ok else (result or "failed")


def main():
    global OUT, SMOKE
    argv, deadline = sys.argv[1:], None
    if len(argv) == 2 and argv[0] == "--smoke":
        OUT = SMOKE = os.path.abspath(argv[1])
    elif len(argv) == 2 and argv[0] == "--deadline":
        deadline = datetime.fromisoformat(argv[1])
    elif argv:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    log(f"step 9 queue start: {[n for n, _ in RUNS]}" + (f", deadline {deadline:%Y-%m-%d %H:%M}" if deadline else ""))
    results = {name: run(name, args, deadline) for name, args in RUNS}
    failed = [n for n, r in results.items() if r != "done"]
    ppls = {n: info(os.path.join(OUT, n)).get("results", {}).get("val_ppl") for n, _ in RUNS}
    summary = ", ".join(f"{n} {p:.3f}" for n, p in ppls.items() if p)
    log("QUEUE DONE" + (f", NOT finished: {failed}" if failed else "") + f" | {summary}")
    notify(("Schritt-9-Nacht fertig" if not failed else f"Schritt 9 UNVOLLSTÄNDIG ({', '.join(failed)})")
           + f": {summary}. Vergleich: B-1M 21.837 / 21.752.")


if __name__ == "__main__":
    main()
