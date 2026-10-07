"""Night queue at home (RX 9070): Engram-style n-gram memory vs. product keys (REPORT.md, "Schritt 4").

  python scripts/run_engram.py          (in tmux; resumable, a finished run is skipped)
  python scripts/run_engram.py --smoke DIR   (whole queue on 2M tokens per run into DIR, no ntfy: test before the night)

Runs one after another, same arguments as B-1M-sparse in runs/hampter (only --model and --value_lr differ):
  E-1M-s0, E-1M-s1         n-gram tables, lazy Adam (as the product-key table), value_lr 3e-3 (paper: 5 x lr)
  E-1M-dense-s0            control: plain Adam on all table rows, as in the paper
Output runs/engram/<name>/ with GPU temperature / power / clocks every 10 s (gpu_thermal.csv). A run counts as done
only with run-info.json status "done" and model.pt. A run whose log doesn't move for STALL_MIN minutes is stopped
and reported, the queue goes on. ntfy message after every run if NTFY_TOPIC is set (or in ~/smlm-cloud-kit/cloud.env).
Nothing is pushed; results are evaluated and committed afterwards.
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

OUT = os.path.join(ROOT, "runs", "engram")
STALL_MIN = 30
COMMON = ["--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103", "--eval_every_tokens", "10e6",
          "--data_seed", "1234", "--seq_len", "1024", "--batch_seqs", "32", "--lr", "6e-4", "--weight_decay", "0.1"]
RUNS = [
    ("E-1M-s0", ["--model", "E-1M", "--seed", "0", "--value_lr", "3e-3"]),
    ("E-1M-s1", ["--model", "E-1M", "--seed", "1", "--value_lr", "3e-3"]),
    ("E-1M-dense-s0", ["--model", "E-1M-dense", "--seed", "0", "--value_lr", "3e-3"]),
]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "queue.log"), "a") as f:
        f.write(line + "\n")


SMOKE = None


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
        subprocess.run(["curl", "-s", "-m", "20", "-H", "Title: SMLM Engram", "-d", msg, f"https://ntfy.sh/{topic}"],
                       capture_output=True)
    log("ntfy: " + msg)


def info(d):
    try:
        return json.load(open(os.path.join(d, "run-info.json")))
    except (OSError, ValueError):
        return {}


def done(d):
    return info(d).get("status") == "done" and os.path.exists(os.path.join(d, "model.pt"))


def run(name, args):
    d = os.path.join(OUT, name)
    if done(d):
        log(f"skip {name} (done)")
        return "done"
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
    res = info(d).get("results", {})
    ppl = res.get("val_ppl")
    log(f"end {name} rc={p.returncode} {result or ''} ({(time.time() - t0) / 3600:.2f} h)"
        + (f" val_ppl {ppl:.3f}" if ppl else ""))
    notify(f"{name} {'fertig' if ok else 'FEHLGESCHLAGEN (' + (result or f'rc={p.returncode}') + ')'}"
           + (f", Val-PPL {ppl:.3f}" if ppl else "") + f", {(time.time() - t0) / 3600:.1f} h")
    return "done" if ok else (result or "failed")


def main():
    global OUT, SMOKE
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke":
        OUT = SMOKE = os.path.abspath(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    log(f"engram queue start: {[n for n, _ in RUNS]}")
    results = {name: run(name, args) for name, args in RUNS}
    failed = [n for n, r in results.items() if r != "done"]
    ppls = {n: info(os.path.join(OUT, n)).get("results", {}).get("val_ppl") for n, _ in RUNS}
    summary = ", ".join(f"{n} {p:.3f}" for n, p in ppls.items() if p)
    log("QUEUE DONE" + (f", NOT finished: {failed}" if failed else "") + f" | {summary}")
    notify(("Engram-Nacht fertig" if not failed else f"Engram-Nacht UNVOLLSTÄNDIG ({', '.join(failed)})")
           + f": {summary}. Vergleich: B-1M 21.837 / 21.752, A 25.665 / 25.756.")


if __name__ == "__main__":
    main()
