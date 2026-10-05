"""Step 1 "Gegenwert der Tabelle": dense models D-50M ... D-400M on one Runpod GPU, side by side.

  python scripts/run_dense.py      (started by cloud/setup_dense.sh in tmux session "queue"; resumable)

1. Preflight (runs/cloud_dense_preflight/<run>): every size alone for 3 M tokens on this card: train tok/s, peak
   VRAM, final evaluation, checkpoint save, inference benchmark. The small result files are pushed.
2. Budget gate (pre-registered in REPORT.md, step 1). Projected Pod hours = Pod uptime + 0.1 h
   + 1.15 x sum over the admitted sizes of (500 M tokens + evaluation tokens / 3) / (preflight tok/s alone) + END_H.
   Sizes are admitted in the order 50M, 100M, 200M, 400M while this stays within
   SMLM_COST_CAP_USD / SMLM_PRICE_USD_H. A dropped size is logged and reported (ntfy, REPORT).
3. The admitted sizes run at the same time, largest first; a run only starts when its preflight peak (x 1.1 + 1.5 GiB)
   still fits next to the reservations of the runs already started.
4. After every run: small files pushed to GitHub; the run directory (checkpoint included) copied to the storage box
   (rsync over ssh, host alias "storagebox") and verified there with sha256; phone notification (ntfy).
5. End: sha256 manifest of the checkpoints (pushed), final copy + verification. Backup ok -> ntfy, then Claude stops
   and deletes the Pod from home; if that has not happened after FALLBACK_MIN minutes the Pod stops itself.
   Backup failed -> ntfy and the Pod stops itself at once (stopped, not deleted: /workspace keeps everything).
Safety: a run whose logs do not change for STALL_MIN minutes is terminated (not restarted: train.py writes no mid-run
checkpoints); when the Pod uptime reaches the cap minus HARD_MARGIN_H, everything still running is terminated, the
results are backed up and the Pod is stopped. cloud/setup_dense.sh additionally arms an independent watchdog.

Test the scheduling locally without a GPU:  SMLM_FAKE_TRAIN=1 SMLM_BOX=/tmp/box python scripts/run_dense.py
"""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.gpu_monitor import log_until  # noqa: E402

FAKE = os.environ.get("SMLM_FAKE_TRAIN") == "1"
OUT = os.environ.get("SMLM_DENSE_OUT") or os.path.join(ROOT, "runs", "cloud_dense")
PRE = OUT + "_preflight"
RUNS = [("D-50M-s0", "D-50M"), ("D-100M-s0", "D-100M"), ("D-200M-s0", "D-200M"), ("D-400M-s0", "D-400M")]
TOKENS = 500e6
EVAL_EVERY = 10e6
VAL_TOKENS = 1.48e6 + 0.25e6               # Wikipedia val + WikiText-103 val, evaluated every EVAL_EVERY tokens
ARGS = ["--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103", "--eval_every_tokens", "10e6",
        "--seed", "0", "--data_seed", "1234"]
PRE_ARGS = ["--data", "wikipedia", "--tokens", "3e6", "--extra_val", "wikitext103", "--eval_every_tokens", "3e6",
            "--seed", "0", "--data_seed", "1234"]
PRICE = float(os.environ.get("SMLM_PRICE_USD_H", 3.49))
CAP_USD = float(os.environ.get("SMLM_COST_CAP_USD", 18))
CAP_H = CAP_USD / PRICE
END_H = 0.4                    # final evaluation, manifest, backup, deletion
UPTIME_MARGIN_H = 0.1          # billing may start before the container (image pull)
FACTOR = 1.15                  # parallel runs on one GPU are not faster than the sum of the single runs
HARD_MARGIN_H = 0.3
STALL_MIN = float(os.environ.get("SMLM_STALL_MIN", 30))
FALLBACK_MIN = float(os.environ.get("SMLM_FALLBACK_MIN", 20))
BOX = os.environ.get("SMLM_BOX", "storagebox:smlm")       # "host:dir" (ssh) or a local directory (tests)
BASE = os.environ.get("SMLM_BACKUP_BASE", ROOT)           # paths on the box are relative to this directory
POLL_S = 2 if FAKE else 30
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_019qFk4hJ98tAkLc3kBY5njm")
T0 = time.time()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "queue.log"), "a") as f:
        f.write(line + "\n")


def notify(msg):
    topic = os.environ.get("NTFY_TOPIC")
    if topic and not FAKE:
        subprocess.run(["curl", "-s", "-m", "20", "-H", "Title: SMLM Cloud", "-d", msg, f"https://ntfy.sh/{topic}"],
                       capture_output=True)
    log("ntfy: " + msg)


def pod_hours():
    """Hours since the container started (PID 1), plus a margin for billing before that."""
    if FAKE:
        return float(os.environ.get("SMLM_FAKE_UPTIME_H", 0.5)) + (time.time() - T0) / 3600 * float(
            os.environ.get("SMLM_FAKE_SPEEDUP", 60))
    r = subprocess.run(["ps", "-o", "etimes=", "-p", "1"], capture_output=True, text=True)
    return int(r.stdout.strip()) / 3600 + UPTIME_MARGIN_H


def git_push(message, paths):
    if FAKE:
        log(f"(fake) would push: {message} {[os.path.relpath(p, ROOT) for p in paths]}")
        return True
    for p in paths:
        subprocess.run(["git", "add", "-f", p], cwd=ROOT, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", message + TRAILER], cwd=ROOT, capture_output=True)
    for attempt in range(5):
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash", "origin", "main"], cwd=ROOT,
                       capture_output=True)
        r = subprocess.run(["git", "push", "-q", "origin", "HEAD:main"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return True
        log(f"git push failed (attempt {attempt + 1}): {r.stderr.strip()[:200]}")
        time.sleep(30)
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(paths):
    """Copy files to BOX keeping their path relative to BASE (the repository), then compare sha256 there."""
    paths = [os.path.relpath(p, BASE) for p in paths if os.path.exists(p)]
    if not paths:
        return True
    remote = ":" in BOX
    host, rdir = BOX.split(":", 1) if remote else (None, BOX)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]
    for attempt in range(3):
        # -rt, no owner/group: the box (and /workspace) refuse chown
        cmd = ["rsync", "-rt", "--partial", "-R", *(["-e", " ".join(ssh)] if remote else []), *paths, BOX + "/"]
        if not remote:
            os.makedirs(BOX, exist_ok=True)
        r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"backup rsync failed (attempt {attempt + 1}): {r.stderr.strip()[:300]}")
            time.sleep(60)
            continue
        want = {p: sha256(os.path.join(BASE, p)) for p in paths}
        if remote:
            r = subprocess.run(ssh + [host, "sha256sum", *[f"{rdir}/{p}" for p in paths]], capture_output=True,
                               text=True)
            got = {ln.split(None, 1)[1].strip()[len(rdir) + 1:]: ln.split()[0] for ln in r.stdout.splitlines()
                   if ln.strip()}
        else:
            got = {p: sha256(os.path.join(BOX, p)) for p in paths if os.path.exists(os.path.join(BOX, p))}
        bad = [p for p in paths if got.get(p) != want[p]]
        if not bad:
            log(f"backup ok: {len(paths)} files verified on {BOX}")
            return True
        log(f"backup verification failed (attempt {attempt + 1}): {bad[:5]}")
        time.sleep(60)
    return False


def small_files(run_dir):
    keep = ("run-info.json", "metrics.csv", "train_log.csv", "stdout.log", "gpu_thermal.csv")
    return [os.path.join(run_dir, f) for f in keep if os.path.exists(os.path.join(run_dir, f))]


def read_info(run_dir):
    p = os.path.join(run_dir, "run-info.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def admit(speeds, elapsed_h, cap_h=None, order=None):
    """Budget gate. speeds: run -> preflight train tok/s alone (None = preflight failed).
    Returns (admitted, dropped, projected_hours)."""
    cap_h = CAP_H if cap_h is None else cap_h
    order = order or [n for n, _ in RUNS]
    n_evals = TOKENS / EVAL_EVERY
    work_tokens = TOKENS + n_evals * VAL_TOKENS / 3          # an evaluation token costs ~1/3 of a training token
    admitted, dropped, total, blocked = [], [], 0.0, False
    for name in order:
        s = speeds.get(name)
        if not s:
            dropped.append((name, "preflight failed"))          # does not block the other sizes
            continue
        t = FACTOR * work_tokens / s / 3600
        if not blocked and elapsed_h + total + t + END_H <= cap_h:
            admitted.append(name)
            total += t
        else:
            blocked = True                                       # a larger size would not fit either
            dropped.append((name, f"budget: projected {elapsed_h + total + t + END_H:.2f} h > cap {cap_h:.2f} h"))
    return admitted, dropped, elapsed_h + total + END_H


def train_cmd(model, out, args):
    if FAKE:
        sp = float(os.environ.get("SMLM_FAKE_TOK_S", 4e5)) * {"D-50M": 4, "D-100M": 2, "D-200M": 1, "D-400M": 0.5}[model]
        dur = float(os.environ.get("FAKE_S", 6)) * (1 if "3e6" in args else 3)
        code = ("import json,os,time; time.sleep(%r); open(os.path.join(%r,'model.pt'),'wb').write(os.urandom(4096));"
                "json.dump({'status':'done','results':{'val_ppl':20.0,'train_tok_s_median':%r,"
                "'peak_train_vram_gib':10.0}}, open(os.path.join(%r,'run-info.json'),'w'))") % (dur, out, sp, out)
        return [sys.executable, "-c", code]
    return [sys.executable, "-m", "smlm.train", "--model", model, "--out_dir", out, *args]


class Job:
    def __init__(self, name, model, base, args):
        self.name, self.model, self.args = name, model, args
        self.out = os.path.join(base, name)
        self.proc, self.pid, self.result, self.done = None, None, None, False
        self.reserve_gib = 0.0

    def status(self):
        return read_info(self.out).get("status")

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        cmd = train_cmd(self.model, self.out, self.args)
        log(f"start {self.name}: {' '.join(cmd[1:]) if not FAKE else '(fake)'}")
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        self.fh = open(os.path.join(self.out, "stdout.log"), "w")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.fh, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)       # a crash or signal never reaches the others
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=log_until, args=(os.path.join(self.out, "gpu_thermal.csv"),
                                                               self.stop_evt), daemon=True)
        if not FAKE:
            self.thread.start()

    def adopt(self):
        """Scheduler restarted while this run is still going: watch the existing process."""
        r = subprocess.run(["pgrep", "-f", f"smlm.train .*--out_dir {self.out}( |$)"], capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.split()]
        if pids:
            self.pid = pids[0]
            log(f"adopted running {self.name} (pid {self.pid})")
        return bool(pids)

    def alive(self):
        if self.proc is not None:
            return self.proc.poll() is None
        try:
            with open(f"/proc/{self.pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, TypeError):
            return False

    def stalled(self):
        files = [f for f in small_files(self.out) if not f.endswith("gpu_thermal.csv")]
        return bool(files) and time.time() - max(os.path.getmtime(f) for f in files) > STALL_MIN * 60

    def kill(self, why):
        log(f"{self.name}: {why}, terminating")
        try:
            os.killpg(self.proc.pid if self.proc else self.pid, 15)
        except OSError:
            pass
        self.result = why

    def wait_end(self):
        if self.proc is not None:
            try:
                self.proc.wait(180)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
            self.fh.close()
            self.stop_evt.set()
            if self.thread.is_alive():
                self.thread.join()


def write_status(state):
    path = os.path.join(ROOT, "report", "dense_status.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = {}
    for name, model in RUNS:
        info, pre = read_info(os.path.join(OUT, name)), read_info(os.path.join(PRE, name))
        res, pres = info.get("results", {}), pre.get("results", {})
        runs[name] = {"model": model, "status": info.get("status"), "val_ppl": res.get("val_ppl"),
                      "val2_ppl": res.get("val2_ppl"), "params": info.get("params") or pre.get("params"),
                      "train_tok_s_median": res.get("train_tok_s_median"),
                      "preflight_tok_s": pres.get("train_tok_s_median"),
                      "preflight_peak_vram_gib": pres.get("peak_train_vram_gib")}
    state = {**state, "runs": runs, "pod_hours": round(pod_hours(), 3), "price_usd_h": PRICE, "cap_usd": CAP_USD,
             "est_cost_usd": round(pod_hours() * PRICE, 2), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "w") as f:
        json.dump(state, f, indent=1)
    return path


def finish(job, state, backup_ok):
    job.wait_end()
    job.done = True
    st = job.status()
    info = read_info(job.out)
    ppl = info.get("results", {}).get("val_ppl")
    log(f"end {job.name} status={st}" + (f" ({job.result})" if job.result else "") + (f" val_ppl {ppl:.3f}" if ppl else ""))
    if st != "done":
        log(f"{job.name} did NOT finish; train.py has no mid-run checkpoints, so it is reported, not restarted")
    state["runs_end"][job.name] = st or "failed"
    status_path = write_status(state)
    pushed = git_push(f"Cloud dense: {job.name} {st or 'failed'}" + (f", val PPL {ppl:.3f}" if ppl else ""),
                      small_files(job.out) + [os.path.join(OUT, "queue.log"), status_path])
    ok = backup([os.path.join(job.out, f) for f in sorted(os.listdir(job.out))])
    backup_ok[job.name] = ok
    notify(f"{job.name} {st or 'FAILED (not restarted)'}" + (f", val PPL {ppl:.3f}" if ppl else "")
           + f"; Pod {pod_hours():.2f} h ≈ {pod_hours() * PRICE:.2f} $" + ("" if pushed else "; git push FAILED")
           + ("" if ok else "; Backup FAILED"))


def preflight(state):
    speeds = {}
    for name, model in RUNS:
        job = Job(name, model, PRE, PRE_ARGS)
        if job.status() != "done":
            job.start()
            while job.alive():
                time.sleep(POLL_S / 3)
            job.wait_end()
        info = read_info(job.out)
        res = info.get("results", {})
        if info.get("status") == "done":
            speeds[name] = res.get("train_tok_s_median")
            state["peaks"][name] = res.get("peak_train_vram_gib")
            log(f"preflight {name}: {speeds[name]:.0f} tok/s alone, peak {res.get('peak_train_vram_gib'):.1f} GiB")
        else:
            log(f"preflight {name} FAILED (status {info.get('status')}), see {job.out}/stdout.log")
        pt = os.path.join(job.out, "model.pt")
        if os.path.exists(pt):
            os.remove(pt)
    state["preflight_tok_s"] = speeds
    git_push("Cloud dense: preflight", [f for n, _ in RUNS for f in small_files(os.path.join(PRE, n))]
             + [os.path.join(OUT, "queue.log")])
    return speeds


def gpu_total_gib():
    if FAKE:
        return float(os.environ.get("SMLM_FAKE_GPU_GIB", 79.6))
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                       capture_output=True, text=True)
    return float(r.stdout.strip()) / 1024


def manifest():
    lines = []
    for name, _ in RUNS:
        p = os.path.join(OUT, name, "model.pt")
        if os.path.exists(p):
            lines.append(f"{sha256(p)}  {os.path.relpath(p, ROOT)}")
    path = os.path.join(OUT, "checkpoints.sha256")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def stop_pod(reason):
    if FAKE:
        log(f"(fake) would stop the Pod: {reason}")
        return
    r = subprocess.run(["bash", os.path.join(ROOT, "cloud", "stop_pod.sh")], cwd=ROOT, capture_output=True, text=True)
    log(f"Pod stop ({reason}): rc={r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
    if r.returncode != 0:
        notify("Pod konnte sich NICHT selbst stoppen - bitte in der Runpod-Konsole stoppen!")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PRE, exist_ok=True)
    state = {"phase": "preflight", "peaks": {}, "runs_end": {}, "cap_h": round(CAP_H, 3)}
    log(f"dense queue start: cap {CAP_USD} $ at {PRICE} $/h = {CAP_H:.2f} h, Pod uptime {pod_hours():.2f} h")
    notify(f"Dense-Läufe: Setup fertig, Probelauf startet (Pod {pod_hours():.2f} h)")
    speeds = preflight(state)
    admitted, dropped, proj = admit(speeds, pod_hours())
    state.update({"phase": "runs", "admitted": admitted, "dropped": dropped, "projected_hours": round(proj, 2),
                  "projected_cost_usd": round(proj * PRICE, 2)})
    log(f"budget gate: admitted {admitted}, dropped {dropped}, projected {proj:.2f} h ≈ {proj * PRICE:.2f} $")
    notify(f"Budget-Wächter: laufen {', '.join(a.split('-s0')[0] for a in admitted) or 'nichts'}"
           + (f"; weggelassen {', '.join(d[0].split('-s0')[0] + ' (' + d[1] + ')' for d in dropped)}" if dropped else "")
           + f"; Hochrechnung {proj:.1f} h ≈ {proj * PRICE:.0f} $")
    write_status(state)
    jobs = [Job(n, m, OUT, ARGS) for n, m in RUNS if n in admitted]
    jobs.sort(key=lambda j: -[n for n, _ in RUNS].index(j.name))           # largest first
    for j in jobs:
        if j.status() == "done":
            j.done = True
        elif j.adopt():
            j.reserve_gib = 1.1 * (state["peaks"].get(j.name) or 20) + 1.5
        elif j.status() is not None:            # started earlier, process gone: report, do not restart
            j.done = True
            state["runs_end"][j.name] = f"{j.status()} (process gone when the scheduler restarted, not restarted)"
            log(f"{j.name}: status {j.status()} but no process - reported, not restarted")
    total = gpu_total_gib() - 2.0
    backup_ok = {}
    hard_stop = False
    while not all(j.done for j in jobs):
        for j in jobs:
            if j.done or (j.proc is None and j.pid is None):
                continue
            if not j.alive():
                finish(j, state, backup_ok)
            elif j.stalled():
                j.kill(f"stalled (no log change for {STALL_MIN:.0f} min)")
        if pod_hours() >= CAP_H - HARD_MARGIN_H and not hard_stop:
            hard_stop = True
            notify(f"Kostendeckel erreicht ({pod_hours():.2f} h): laufende Läufe werden beendet")
            for j in jobs:
                if not j.done and (j.proc is not None or j.pid is not None):
                    j.kill("cost cap")
                elif not j.done:
                    j.done = True
                    state["runs_end"][j.name] = "not started (cost cap)"
        for j in jobs:
            if j.done or j.proc is not None or j.pid is not None or hard_stop:
                continue
            need = 1.1 * (state["peaks"].get(j.name) or 20) + 1.5
            used = sum(x.reserve_gib for x in jobs if not x.done and (x.proc is not None or x.pid is not None))
            if used + need <= total:
                j.reserve_gib = need
                j.start()
        time.sleep(POLL_S)
    state["phase"] = "end"
    m = manifest()
    status_path = write_status(state)
    git_push("Cloud dense: queue finished, checkpoint manifest", [m, os.path.join(OUT, "queue.log"), status_path])
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(OUT) for f in fs] + \
            [os.path.join(dp, f) for dp, _, fs in os.walk(PRE) for f in fs]
    ok = backup(files)
    if ok:
        marker = os.path.join(OUT, "BACKUP_OK")
        with open(marker, "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} all files of {os.path.relpath(OUT, ROOT)} verified on {BOX}\n")
        backup([marker])
        log("QUEUE DONE, backup verified")
        notify(f"Fertig: Ergebnisse gepusht, Checkpoints auf der Storage Box geprüft. Claude stoppt und löscht jetzt "
               f"den Pod (Pod {pod_hours():.2f} h ≈ {pod_hours() * PRICE:.2f} $). Falls nicht: Selbst-Stopp in "
               f"{FALLBACK_MIN:.0f} min.")
        time.sleep(FALLBACK_MIN * 60 if not FAKE else 1)
        stop_pod("fallback: not deleted from home within the waiting time")
        notify("Pod hat sich selbst gestoppt (Fallback). Gestoppt kostet er nur noch Speicher; Claude löscht ihn.")
    else:
        log("QUEUE DONE, BACKUP FAILED")
        notify("Fertig, aber das Sichern auf die Storage Box ist FEHLGESCHLAGEN. Der Pod wird nur gestoppt (nicht "
               "gelöscht), die Daten liegen weiter auf seinem Volume.")
        stop_pod("backup failed")


if __name__ == "__main__":
    main()
