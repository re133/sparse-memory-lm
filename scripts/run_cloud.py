"""Cloud queue (Runpod Pod, 1 x H200): B-1M (control), B-4M, B-16M, back to back, then stop the Pod.

  python scripts/run_cloud.py            (started by cloud/setup.sh in tmux session "queue"; resumable)

Every run: settings and data of B-1M-sparse s0 (500 M Wikipedia tokens, data seed 1234, init seed 0, value LR
2.4e-3, micro-batch 4, eval every 10 M tokens, WikiText-103 val as second set), Triton kernels
(--mem_impl triton). GPU temperature / power / clocks every 10 s in <run>/gpu_thermal.csv.
After every run: status block in REPORT.de.md (scripts/cloud_status.py), commit + push of the small run files
(run-info, metrics, logs, thermal CSV; checkpoints stay on the disk), optional phone notification (ntfy).
At the end: sha256 manifest of all checkpoints (pushed), then the Pod is stopped through the Runpod API with the
Pod's own key (cloud/stop_pod.sh): the GPU is released and compute billing stops; the volume /workspace with the
checkpoints stays (billed as stopped volume).

Safety: a run whose logs have not changed for STALL_MIN minutes is terminated (and the queue goes on); after
MAX_HOURS in total the queue stops what is running and stops the Pod.
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

# SMLM_CLOUD_DRYRUN=1: preflight / test of the queue mechanics: every configuration for 1 M tokens (compile,
# VRAM peak, final evaluation, checkpoint save, inference benchmark on the real card), own output directory,
# no git push, no Pod stop. setup.sh runs it on the Pod before the real queue;
# SMLM_CLOUD_DRYRUN_RUNS=B-1M-s0 limits it (e.g. on a 16 GB card at home).
DRY = os.environ.get("SMLM_CLOUD_DRYRUN") == "1"
OUT = os.path.join(ROOT, "runs", "cloud_dryrun" if DRY else "cloud")
RUNS = [("B-1M-s0", "B-1M-sparse"), ("B-4M-s0", "B-4M-sparse"), ("B-16M-s0", "B-16M-sparse")]
ARGS = ["--mem_impl", "triton", "--value_lr", "2.4e-3", "--data", "wikipedia", "--tokens", "500e6",
        "--extra_val", "wikitext103", "--eval_every_tokens", "10e6", "--seed", "0", "--data_seed", "1234"]
if DRY:
    only = os.environ.get("SMLM_CLOUD_DRYRUN_RUNS")
    RUNS = [r for r in RUNS if not only or r[0] in only.split(",")]
    ARGS = [a if a not in ("500e6", "10e6") else {"500e6": "1e6", "10e6": "1e6"}[a] for a in ARGS]
STALL_MIN = float(os.environ.get("SMLM_STALL_MIN", 30))
MAX_HOURS = float(os.environ.get("SMLM_MAX_HOURS", 12))
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_019qFk4hJ98tAkLc3kBY5njm")
T_START = time.time()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "queue.log"), "a") as f:
        f.write(line + "\n")


def notify(msg):
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        subprocess.run(["curl", "-s", "-m", "20", "-d", msg, f"https://ntfy.sh/{topic}"], capture_output=True)


def git_push(message, paths):
    if DRY:
        log(f"(dry run) would commit + push: {message} {[os.path.relpath(p, ROOT) for p in paths]}")
        return True
    for p in paths:
        subprocess.run(["git", "add", "-f", p], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", message + TRAILER], cwd=ROOT, capture_output=True)
    for attempt in range(5):
        # someone may have pushed to main meanwhile (docs/notes/CLOUD.md asks not to); rebase the result commit onto it
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash", "origin", "main"], cwd=ROOT,
                       capture_output=True)
        r = subprocess.run(["git", "push", "-q", "origin", "HEAD:main"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        log(f"git push failed (attempt {attempt + 1}): {r.stderr.strip()[:200]}")
        time.sleep(30)
    return False


def small_files(run_dir):
    keep = ("run-info.json", "metrics.csv", "train_log.csv", "stdout.log", "gpu_thermal.csv")
    return [os.path.join(run_dir, f) for f in keep if os.path.exists(os.path.join(run_dir, f))]


def status(name):
    p = os.path.join(OUT, name, "run-info.json")
    return json.load(open(p)).get("status") if os.path.exists(p) else None


def run(name, model):
    out = os.path.join(OUT, name)
    if status(name) == "done":
        if os.path.exists(os.path.join(out, "model.pt")):
            log(f"skip {name} (done)")
            return "done"
        # e.g. a run-info.json from git in a fresh clone: don't take it as done, don't silently train again either
        log(f"{name}: run-info.json says done but model.pt is missing - not retrained, please check")
        notify(f"SMLM cloud: {name} steht auf done, aber model.pt fehlt - nicht neu trainiert, bitte prüfen")
        return "model.pt missing"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "smlm.train", "--model", model, "--out_dir", out, *ARGS]
    log("start " + name + ": " + " ".join(cmd[1:]))
    stop = threading.Event()
    th = threading.Thread(target=log_until, args=(os.path.join(out, "gpu_thermal.csv"), stop), daemon=True)
    th.start()
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(os.path.join(out, "stdout.log"), "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
        result = None
        while p.poll() is None:
            time.sleep(30)
            newest = max(os.path.getmtime(f) for f in small_files(out) if not f.endswith("gpu_thermal.csv"))
            if time.time() - newest > STALL_MIN * 60:
                result = "stalled"
            if time.time() - T_START > MAX_HOURS * 3600:
                result = "time limit"
            if result:
                log(f"{name}: {result}, terminating")
                p.terminate()
                try:
                    p.wait(120)
                except subprocess.TimeoutExpired:
                    p.kill()
                break
    stop.set()
    th.join()
    st = status(name)
    log(f"end {name} rc={p.returncode} status={st}" + (f" ({result})" if result else ""))
    if not DRY:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "cloud_status.py")], cwd=ROOT)
    info = json.load(open(os.path.join(out, "run-info.json"))) if os.path.exists(os.path.join(out, "run-info.json")) else {}
    ppl = info.get("results", {}).get("val_ppl")
    pushed = git_push(f"Cloud: {name} {st or 'failed'}" + (f", val PPL {ppl:.3f}" if ppl else ""),
                      small_files(out) + [os.path.join(OUT, "queue.log"), "REPORT.de.md",
                                          os.path.join("report", "cloud_status.json")])
    notify(f"SMLM cloud: {name} {st or 'failed'}" + (f", val PPL {ppl:.3f}" if ppl else "") +
           ("" if pushed else " (git push FAILED)"))
    return result or st or "failed"


def manifest():
    """sha256 of every checkpoint / large artefact on the disk -> runs/cloud/checkpoints.sha256 (pushed)."""
    lines = []
    for name, _ in RUNS:
        d = os.path.join(OUT, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".pt", ".npy", ".npz")):
                r = subprocess.run(["sha256sum", os.path.relpath(os.path.join(d, f), ROOT)], cwd=ROOT,
                                   capture_output=True, text=True)
                lines.append(r.stdout.strip())
    path = os.path.join(OUT, "checkpoints.sha256")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    log("queue start")
    notify("SMLM cloud: queue started (B-1M, B-4M, B-16M)")
    for name, model in RUNS:
        r = run(name, model)
        if r == "time limit":
            break
    m = manifest()
    pushed = git_push("Cloud: queue finished, checkpoint manifest", [m, os.path.join(OUT, "queue.log")])
    log("QUEUE DONE" + ("" if pushed else " (final git push FAILED - results only on this disk)"))
    stop = os.path.join(ROOT, "cloud", "stop_pod.sh")
    r = subprocess.run(["bash", stop] + (["--check"] if DRY else []), cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        notify("SMLM cloud: queue done, results pushed; Pod stop requested via the Runpod API (GPU billing stops, "
               "/workspace with the checkpoints stays)")
        log("Runpod stop requested: " + r.stdout.strip()[:300])
    else:
        notify("SMLM cloud: queue done, but the Pod could NOT be stopped automatically - stop it in the Runpod "
               "console now (Pods > expand > Stop), otherwise billing continues!")
        log("Runpod stop FAILED: " + (r.stdout + r.stderr).strip()[:300])


if __name__ == "__main__":
    main()
