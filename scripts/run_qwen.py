"""Step 3 cloud queue (Runpod, 1 x H100): Qwen alone (Q), Qwen + table (Q+T), Qwen + dense control (Q+D).

  python scripts/run_qwen.py        (started by cloud/setup_qwen.sh in tmux session "queue"; resumable per step)

Order (each step skipped if its result already exists):
  1. Q:   PPL on all sets + month curve, fact test, general test (lm-eval) + chat answers
  2. probes: 30 training steps each of Q+T and Q+D (nothing saved) -> tok/s
  3. budget gate: projected Pod hours = uptime + 0.1 + 1.15 x (work / tok/s for Q+T and Q+D) + 2 x (time of step 1)
     + 0.4; over SMLM_COST_CAP_USD / SMLM_PRICE_USD_H -> nothing is trained, ntfy, the Pod stops
  4. Q+T: training (2 epochs over train_new), then fact test and general test
  5. Q+D: the same
  6. end as in step 1: sha256 manifest pushed, everything copied to the storage box and verified there, BACKUP_OK,
     ntfy; Claude stops and deletes the Pod, fallback: self-stop after FALLBACK_MIN minutes; backup failed -> stop.
Small result files go to GitHub after every step, add-on checkpoints (addons.pt) to the storage box.
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_dense as rd  # noqa: E402  (log / notify / git_push / backup / stop_pod helpers)
from smlm.gpu_monitor import log_until  # noqa: E402

OUT = os.path.join(ROOT, "runs", "qwen_cloud")
rd.OUT = OUT
PY = os.path.join(ROOT, ".venv-qwen", "bin", "python")
QWEN = os.environ.get("QWEN_DIR", "/workspace/models/Qwen3.5-0.8B")
DATA = os.path.join(ROOT, "data", "qwen_wiki")
PRICE = rd.PRICE
CAP_H = float(os.environ.get("SMLM_COST_CAP_USD", 14)) / PRICE
STALL_MIN = 30
FALLBACK_MIN = rd.FALLBACK_MIN
TRAIN = ["--model_dir", QWEN, "--data_dir", DATA, "--micro_bs", os.environ.get("QWEN_MICRO_BS", "4")]


def is_done(path):
    if not os.path.exists(path):
        return False
    if path.endswith("run-info.json"):                  # written at the start with status "running"
        return rd.read_info(os.path.dirname(path)).get("status") == "done"
    return True


def step(name, cmd, done_file, thermal=False):
    """Run one step as a subprocess (own session) unless it is done; stall / cost-cap watchdog."""
    if is_done(done_file):
        rd.log(f"skip {name} (done)")
        return "done"
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    rd.log(f"start {name}: {' '.join(cmd[1:])}")
    stop = threading.Event()
    th = threading.Thread(target=log_until, args=(os.path.join(d, "gpu_thermal.csv"), stop), daemon=True)
    if thermal:
        th.start()
    t0 = time.time()
    with open(os.path.join(d, "stdout.log"), "a") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        result = None
        while p.poll() is None:
            time.sleep(30)
            if time.time() - os.path.getmtime(os.path.join(d, "stdout.log")) > STALL_MIN * 60:
                result = "stalled"
            if rd.pod_hours() >= CAP_H - rd.HARD_MARGIN_H:
                result = "cost cap"
            if result:
                rd.log(f"{name}: {result}, terminating")
                os.killpg(p.pid, 15)
                p.wait(180)
                break
    stop.set()
    rd.log(f"end {name} rc={p.returncode} {result or ''} ({(time.time() - t0) / 60:.1f} min)")
    ok = p.returncode == 0 and is_done(done_file)
    files = [os.path.join(d, f) for f in os.listdir(d) if not f.endswith(".pt")]
    rd.git_push(f"Cloud qwen: {name} {'done' if ok else 'FAILED'}", files + [os.path.join(OUT, "queue.log")])
    rd.notify(f"Qwen {name}: {'fertig' if ok else 'FEHLGESCHLAGEN ' + (result or f'rc={p.returncode}')}"
              f" (Pod {rd.pod_hours():.2f} h ≈ {rd.pod_hours() * PRICE:.2f} $)")
    return "done" if ok else (result or "failed")


def evals(tag, addons):
    a = ["--addons", addons] if addons else []
    d = os.path.join(OUT, tag)
    r1 = step(f"{tag}_facts", [PY, "scripts/eval_fact_cloze.py", "--model_dir", QWEN, *a, "--out",
                               os.path.join(d, "facts.json")], os.path.join(d, "facts.json"))
    r2 = step(f"{tag}_general", [PY, "scripts/eval_qwen_general.py", "--model_dir", QWEN, *a, "--batch_size", "8",
                                 "--out", os.path.join(d, "general.json")], os.path.join(d, "general.json"))
    return r1, r2


def tok_s(probe_dir):
    import csv
    rows = list(csv.DictReader(open(os.path.join(probe_dir, "train_log.csv"))))
    return float(rows[-1]["tok_s"]) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    rd.log(f"qwen queue start: cap {CAP_H * PRICE:.1f} $ = {CAP_H:.2f} h, Pod uptime {rd.pod_hours():.2f} h")
    rd.notify(f"Qwen-Schritt 3: Setup fertig, Q-Messungen starten (Pod {rd.pod_hours():.2f} h)")
    t_q = time.time()
    q = os.path.join(OUT, "Q")
    step("Q", [PY, "scripts/train_qwen_memory.py", "--kind", "none", "--eval_only", "--out_dir", q, *TRAIN],
         os.path.join(q, "run-info.json"))
    evals("Q", None)
    q_hours = (time.time() - t_q) / 3600
    speeds = {}
    for kind, tag in (("memory", "QT"), ("dense", "QD")):
        pdir = os.path.join(OUT, f"probe_{tag}")
        step(f"probe_{tag}", [PY, "scripts/train_qwen_memory.py", "--kind", kind, "--out_dir", pdir, "--max_steps",
                              "30", "--eval_windows", "1", "--save", "0", *TRAIN], os.path.join(pdir, "run-info.json"))
        speeds[tag] = tok_s(pdir)
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    work = 2 * meta["splits"]["train_new"]["n_tokens"] * 1.1          # 2 epochs + periodic evaluations
    need = sum(work / s / 3600 for s in speeds.values() if s) * 1.15 + 2 * q_hours + rd.END_H
    proj = rd.pod_hours() + need
    rd.log(f"budget gate: tok/s {speeds}, Q steps {q_hours:.2f} h, projected {proj:.2f} h ≈ {proj * PRICE:.2f} $ "
           f"(cap {CAP_H:.2f} h)")
    if not all(speeds.values()) or proj > CAP_H:
        rd.notify(f"Budget-Wächter Qwen: Hochrechnung {proj:.1f} h ≈ {proj * PRICE:.0f} $ über dem Deckel "
                  f"({CAP_H * PRICE:.0f} $) oder Probe fehlgeschlagen (tok/s {speeds}) - nichts trainiert, Pod stoppt.")
        rd.backup([os.path.join(dp, f) for dp, _, fs in os.walk(OUT) for f in fs])
        rd.stop_pod("qwen budget gate")
        return
    rd.notify(f"Budget-Wächter Qwen: ok, Hochrechnung {proj:.1f} h ≈ {proj * PRICE:.0f} $; Q+T startet")
    for kind, tag in (("memory", "QT"), ("dense", "QD")):
        d = os.path.join(OUT, f"{tag}-s0")
        r = step(f"{tag}-s0", [PY, "scripts/train_qwen_memory.py", "--kind", kind, "--out_dir", d, *TRAIN],
                 os.path.join(d, "addons.pt"), thermal=True)
        if r == "done":
            rd.backup([os.path.join(d, "addons.pt")])
            evals(tag, os.path.join(d, "addons.pt"))
        if r == "cost cap":
            break
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(OUT) for f in fs]
    lines = [f"{rd.sha256(p)}  {os.path.relpath(p, ROOT)}" for p in files if p.endswith(".pt")]
    man = os.path.join(OUT, "checkpoints.sha256")
    open(man, "w").write("\n".join(lines) + "\n")
    rd.git_push("Cloud qwen: queue finished, checkpoint manifest", [man, os.path.join(OUT, "queue.log")])
    ok = rd.backup(files + [man])
    if ok:
        marker = os.path.join(OUT, "BACKUP_OK")
        open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " all files verified on the box\n")
        rd.backup([marker])
        rd.log("QUEUE DONE, backup verified")
        rd.notify(f"Qwen fertig, alles auf der Storage Box geprüft. Claude stoppt und löscht den Pod "
                  f"(Pod {rd.pod_hours():.2f} h ≈ {rd.pod_hours() * PRICE:.2f} $).")
        time.sleep(FALLBACK_MIN * 60)
        rd.stop_pod("fallback")
    else:
        rd.log("QUEUE DONE, BACKUP FAILED")
        rd.notify("Qwen fertig, aber das Sichern auf die Storage Box ist FEHLGESCHLAGEN - Pod wird nur gestoppt.")
        rd.stop_pod("backup failed")


if __name__ == "__main__":
    main()
