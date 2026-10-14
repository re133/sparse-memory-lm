"""Step 1 "Gegenwert der Tabelle": dense models D-50M ... D-400M side by side on one rented GPU (Runpod Pod or a
plain VM such as IONOS); since step 10 also any other runs (SMLM_DENSE_RUNS).

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
5. End: sha256 manifest of the checkpoints (pushed), final copy + verification, then the end message, then queue.log
   once more, so the copy on the box ends with the same lines as here.
   On a Runpod Pod: backup ok -> Claude stops and deletes the Pod from home; if that has not happened after
   FALLBACK_MIN minutes the Pod stops itself. Backup failed -> the Pod stops itself at once (stopped, not deleted:
   /workspace keeps everything).
   On a plain VM (IONOS, SMLM_PLATFORM=vm or no Runpod variables): nothing stops by itself. Claude checks the backup
   on the box from home, then the instance is deleted by hand; backup failed -> ntfy, the instance must not be deleted.
Safety: a run whose logs do not change for STALL_MIN minutes is terminated; when the Pod uptime reaches the cap minus
HARD_MARGIN_H, everything still running is terminated, the results are backed up and the Pod is stopped (a VM is not).
cloud/setup_dense.sh additionally arms an independent watchdog.
Checkpoints (SMLM_DENSE_CKPT_EVERY tokens, default 0 = none, as before): train.py then writes ckpt.pt and every run
starts with --resume. A run that crashes, stalls or whose process is gone after a restart of this scheduler (or of the
VM) is started again and continues from its last checkpoint, at most MAX_RESTARTS times per run (counted across
scheduler restarts); diverged and aborted runs are never restarted. The preflight then also stops each run halfway with
a checkpoint and resumes it, so saving and loading are tried once on this machine before the long runs.

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
OUT = os.path.join(ROOT, os.environ["SMLM_DENSE_OUT"]) if os.environ.get("SMLM_DENSE_OUT") else \
    os.path.join(ROOT, "runs", "cloud_dense")
PRE = OUT + "_preflight"
RUNS = [("D-50M-s0", "D-50M"), ("D-100M-s0", "D-100M"), ("D-200M-s0", "D-200M"), ("D-400M-s0", "D-400M")]
# step 10 (FACTK) on the Pod: SMLM_DENSE_RUNS="D-100M-factk-s0:D-100M" SMLM_DENSE_DATA=wikipedia_factk
# SMLM_DENSE_OUT=runs/factk (everything else, including the training arguments, as in step 1)
# optional third field: extra training arguments of that run, e.g. "B-16M-factk-s0:B-16M-sparse:--mem_impl triton"
EXTRA = {}
if os.environ.get("SMLM_DENSE_RUNS"):
    RUNS = []
    for r in os.environ["SMLM_DENSE_RUNS"].split(","):
        name, model, *extra = r.split(":", 2)
        RUNS.append((name, model))
        EXTRA[name] = extra[0].split() if extra else []
DATA = os.environ.get("SMLM_DENSE_DATA", "wikipedia")
TOKENS_ARG = os.environ.get("SMLM_DENSE_TOKENS", "500e6")
EVAL_ARG = os.environ.get("SMLM_DENSE_EVAL_EVERY", "10e6")
CKPT_ARG = os.environ.get("SMLM_DENSE_CKPT_EVERY", "0")
TOKENS, EVAL_EVERY, CKPT_EVERY = float(TOKENS_ARG), float(EVAL_ARG), float(CKPT_ARG)
COMMON = os.environ.get("SMLM_DENSE_ARGS", "").split()     # extra arguments of every run, e.g. "--skip_nonfinite"
MAX_RESTARTS = int(os.environ.get("SMLM_MAX_RESTARTS", 3))
VAL_TOKENS = 1.48e6 + 0.25e6               # Wikipedia val + WikiText-103 val, evaluated every EVAL_EVERY tokens
# step 1 as literals (tests/test_devcloud_queue.py reads them from this file); the environment changes them below
ARGS = ["--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103", "--eval_every_tokens", "10e6",
        "--seed", "0", "--data_seed", "1234"]
PRE_ARGS = ["--data", "wikipedia", "--tokens", "3e6", "--extra_val", "wikitext103", "--eval_every_tokens", "3e6",
            "--seed", "0", "--data_seed", "1234"]
ARGS, PRE_ARGS = ([DATA if a == "wikipedia" else a for a in x] for x in (ARGS, PRE_ARGS))
ARGS[ARGS.index("--tokens") + 1], ARGS[ARGS.index("--eval_every_tokens") + 1] = TOKENS_ARG, EVAL_ARG
if CKPT_EVERY:
    ARGS += ["--ckpt_every_tokens", CKPT_ARG, "--resume"]
    PRE_ARGS[PRE_ARGS.index("--eval_every_tokens") + 1] = "1.5e6"     # two evaluations: the preflight resumes at 1.5M
ARGS, PRE_ARGS = ARGS + COMMON, PRE_ARGS + COMMON
PRICE = float(os.environ.get("SMLM_PRICE_USD_H", 3.49))
CAP_USD = float(os.environ.get("SMLM_COST_CAP_USD", 18))
CUR = os.environ.get("SMLM_CURRENCY", "$")                 # price and cap are in this currency (IONOS: €)
CAP_H = CAP_USD / PRICE
END_H = float(os.environ.get("SMLM_END_H", 0.4))           # final evaluation, manifest, backup, deletion
UPTIME_MARGIN_H = 0.1          # billing may start before the container (image pull)
# parallel runs on one GPU are not faster than the sum of the single runs; one run at a time needs less margin
FACTOR = float(os.environ.get("SMLM_PARALLEL_FACTOR", 1.15))
MAX_PARALLEL = int(os.environ.get("SMLM_MAX_PARALLEL", 0))     # 0 = as many as fit into GPU memory
# "slowest": the longest run first (the dense ladder: largest first); "listed": in the order of SMLM_DENSE_RUNS
START_ORDER = os.environ.get("SMLM_START_ORDER", "slowest")
HARD_MARGIN_H = 0.3
STALL_MIN = float(os.environ.get("SMLM_STALL_MIN", 30))
FALLBACK_MIN = float(os.environ.get("SMLM_FALLBACK_MIN", 20))
BOX = os.environ.get("SMLM_BOX", "storagebox:smlm")       # "host:dir" (ssh) or a local directory (tests)
BASE = os.environ.get("SMLM_BACKUP_BASE", ROOT)           # paths on the box are relative to this directory
POLL_S = 2 if FAKE else 30
NO_PUSH = os.environ.get("SMLM_NO_PUSH") == "1"
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_019qFk4hJ98tAkLc3kBY5njm")
T0 = time.time()


def on_runpod():
    """Runpod puts RUNPOD_POD_ID into every Pod; ssh sessions may not inherit it, so PID 1's environment counts too
    (as in cloud/stop_pod.sh)."""
    if os.environ.get("RUNPOD_POD_ID"):
        return True
    try:
        with open("/proc/1/environ", "rb") as f:
            return any(v.startswith(b"RUNPOD_POD_ID=") for v in f.read().split(b"\0"))
    except OSError:
        return False


PLATFORM = os.environ.get("SMLM_PLATFORM") or ("runpod" if on_runpod() else "vm")


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


def machine():
    return "Pod" if PLATFORM == "runpod" else "Instanz"


BILLING_START = os.environ.get("SMLM_BILLING_START_FILE", "/workspace/.smlm-setup/billing_start")


def pod_hours():
    """Hours since the container started (PID 1), plus a margin for billing before that. On a VM cloud-init reboots
    once after installing the driver, so the first boot counts (birth time of /var/lib/cloud/instances, if the file
    system records it, and only if it lies less than an hour before the current boot: a directory that came with the
    image would otherwise make the budget gate drop every run). cloud/setup_dense.sh notes that start once in
    BILLING_START, so a later reboot of the VM doesn't reset the count."""
    if FAKE:
        return float(os.environ.get("SMLM_FAKE_UPTIME_H", 0.5)) + (time.time() - T0) / 3600 * float(
            os.environ.get("SMLM_FAKE_SPEEDUP", 60))
    try:
        with open(BILLING_START) as f:
            return (time.time() - int(f.read().strip())) / 3600 + UPTIME_MARGIN_H
    except (OSError, ValueError):
        pass
    r = subprocess.run(["ps", "-o", "etimes=", "-p", "1"], capture_output=True, text=True)
    up_s = int(r.stdout.strip())
    if PLATFORM == "vm":
        b = subprocess.run(["stat", "-c", "%W", "/var/lib/cloud/instances"], capture_output=True, text=True)
        born = int(b.stdout.strip()) if b.returncode == 0 and b.stdout.strip().isdigit() else 0
        if born > 0 and 0 <= int(time.time()) - born - up_s < 3600:
            up_s = int(time.time()) - born
    return up_s / 3600 + UPTIME_MARGIN_H


def git_push(message, paths):
    if NO_PUSH:
        # the Pod has no write access to the repository: results come home through the storage box backup
        log(f"push skipped (SMLM_NO_PUSH): {message}")
        return True
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


def backup(paths, quiet=False):
    """Copy files to BOX keeping their path relative to BASE (the repository), then compare sha256 there.
    quiet: no log line on success (for queue.log itself, so the copy on the box ends with the same line as here)."""
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
            if not quiet:
                log(f"backup ok: {len(paths)} files verified on {BOX}")
            return True
        log(f"backup verification failed (attempt {attempt + 1}): {bad[:5]}")
        time.sleep(60)
    return False


def run_files(run_dir, with_ckpt):
    """Files of a run for the backup: never half-written ones (<file>.tmp<pid>), the 77 GB checkpoint only when the
    run is not done (then it is the only way to continue it elsewhere)."""
    out = []
    for f in sorted(os.listdir(run_dir)):
        if ".tmp" in f and f.rsplit(".tmp", 1)[1].isdigit():
            continue
        if f == "ckpt.pt" and not with_ckpt:
            continue
        out.append(os.path.join(run_dir, f))
    return out


def small_files(run_dir):
    keep = ("run-info.json", "metrics.csv", "train_log.csv", "stdout.log", "gpu_thermal.csv")
    return [os.path.join(run_dir, f) for f in keep if os.path.exists(os.path.join(run_dir, f))]


def read_info(run_dir):
    p = os.path.join(run_dir, "run-info.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def remaining(name):
    """Share of a run still to train: 0 when done, from its last checkpoint when it has one, else 1."""
    info = read_info(os.path.join(OUT, name))
    if info.get("status") == "done" and os.path.exists(os.path.join(OUT, name, "model.pt")):
        return 0.0
    ck = info.get("checkpoint")
    return max(0.0, 1 - ck["tokens"] / TOKENS) if ck and CKPT_EVERY else 1.0


def admit(speeds, elapsed_h, cap_h=None, order=None, left=None, ckpt_s=None):
    """Budget gate. speeds: run -> preflight train tok/s alone (None = preflight failed); left: run -> share still to
    train (default 1, see remaining()); ckpt_s: run -> seconds per checkpoint save (preflight).
    Returns (admitted, dropped, projected_hours)."""
    cap_h = CAP_H if cap_h is None else cap_h
    order = order or [n for n, _ in RUNS]
    n_evals = TOKENS / EVAL_EVERY
    work_tokens = TOKENS + n_evals * VAL_TOKENS / 3          # an evaluation token costs ~1/3 of a training token
    n_ckpts = TOKENS / CKPT_EVERY + 1 if CKPT_EVERY else 0
    admitted, dropped, total, blocked = [], [], 0.0, False
    for name in order:
        s = speeds.get(name)
        if not s:
            dropped.append((name, "preflight failed"))          # does not block the other sizes
            continue
        frac = 1.0 if left is None else left.get(name, 1.0)
        t = frac * (FACTOR * work_tokens / s + n_ckpts * (ckpt_s or {}).get(name, 0)) / 3600
        if not blocked and elapsed_h + total + t + END_H <= cap_h:
            admitted.append(name)
            total += t
        else:
            blocked = True                                       # a larger size would not fit either
            dropped.append((name, f"budget: projected {elapsed_h + total + t + END_H:.2f} h > cap {cap_h:.2f} h"))
    return admitted, dropped, elapsed_h + total + END_H


FAKE_TRAIN = """
import json, os, sys, time
out, sp, dur, args = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]
info_p, ck_p = os.path.join(out, "run-info.json"), os.path.join(out, "ckpt.pt")
crash = (os.path.basename(out) in os.environ.get("SMLM_FAKE_CRASH", "").split(",")
         and not os.path.dirname(out).endswith("_preflight"))                 # main runs only
info = {"status": "running", "checkpoint": None}
if "--resume" in args and os.path.exists(ck_p):
    info.update(json.load(open(info_p)), status="running")
    info.setdefault("resumes", []).append({"at_step": 1})
json.dump(info, open(info_p, "w"))
time.sleep(dur)
if "--ckpt_every_tokens" in args:
    open(ck_p, "wb").write(os.urandom(2048))
    info["checkpoint"] = {"step": 1, "tokens": 1e6, "save_s": 0.5, "gib": 0.0}
    json.dump(info, open(info_p, "w"))
if crash and not info.get("resumes"):
    sys.exit(1)                                                   # dies once after its first checkpoint
open(os.path.join(out, "model.pt"), "wb").write(os.urandom(4096))
info.update(status="done", results={"val_ppl": 20.0, "train_tok_s_median": sp, "peak_train_vram_gib": 10.0})
json.dump(info, open(info_p, "w"))
if os.path.exists(ck_p) and "--keep_ckpt" not in args:
    os.remove(ck_p)
"""


def train_cmd(model, out, args):
    if FAKE:
        sp = float(os.environ.get("SMLM_FAKE_TOK_S", 4e5)) * {"D-50M": 4, "D-100M": 2, "D-200M": 1,
                                                                "D-400M": 0.5}.get(model, 1)
        dur = float(os.environ.get("FAKE_S", 6)) * (1 if "3e6" in args else 3)
        return [sys.executable, "-c", FAKE_TRAIN, out, str(sp), str(dur), *args]
    return [sys.executable, "-m", "smlm.train", "--model", model, "--out_dir", out, *args]


class Job:
    def __init__(self, name, model, base, args):
        self.name, self.model, self.args = name, model, args
        self.out = os.path.join(base, name)
        self.proc, self.pid, self.result, self.done = None, None, None, False
        self.reserve_gib = 0.0
        self.no_restart = False                 # set when terminated for the cost cap

    def status(self):
        return read_info(self.out).get("status")

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        cmd = train_cmd(self.model, self.out, self.args)
        log(f"start {self.name}: {' '.join(cmd[1:]) if not FAKE else '(fake)'}")
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        # appended: after a restart the first part (and its traceback) stays readable
        self.fh = open(os.path.join(self.out, "stdout.log"), "a")
        self.fh.write(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} start {' '.join(cmd[1:]) if not FAKE else '(fake)'}\n")
        self.fh.flush()
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

    def reset(self):
        """Ready to be started again (restart from its checkpoint)."""
        self.proc, self.pid, self.result, self.reserve_gib = None, None, None, 0.0


def restarts_used():
    try:
        return json.load(open(os.path.join(OUT, "restarts.json")))
    except (OSError, ValueError):
        return {}


def may_restart(job):
    """A run that ended without being done is started again (it resumes from its checkpoint) unless it diverged or
    was aborted by its own check, was stopped for the cost cap, or has used up its restarts."""
    st = job.status()
    return (bool(CKPT_EVERY) and not job.no_restart and st not in ("done", "diverged", "aborted")
            and restarts_used().get(job.name, 0) < MAX_RESTARTS)


def note_restart(job, why):
    used = restarts_used()
    used[job.name] = used.get(job.name, 0) + 1
    with open(os.path.join(OUT, "restarts.json"), "w") as f:
        json.dump(used, f)
    ck = read_info(job.out).get("checkpoint")
    where = f"checkpoint at {ck['tokens'] / 1e6:.0f}M tokens" if ck else "the beginning (no checkpoint yet)"
    notify(f"{job.name}: {why}, Neustart {used[job.name]}/{MAX_RESTARTS} ab {where}")


def write_status(state):
    # fake mode (local test) writes next to its runs, not over the real report
    default_out = OUT == os.path.join(ROOT, "runs", "cloud_dense")
    path = os.path.join(ROOT, "report", "dense_status.json") if default_out and not FAKE else \
        os.path.join(OUT, "dense_status.json")
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
    st = job.status()
    if st != "done" and may_restart(job):
        log(f"end {job.name} status={st} ({job.result or 'process exited'}), restarting from its checkpoint")
        note_restart(job, job.result or f"Prozess beendet (Status {st})")
        job.reset()
        return
    job.done = True
    info = read_info(job.out)
    ppl = info.get("results", {}).get("val_ppl")
    log(f"end {job.name} status={st}" + (f" ({job.result})" if job.result else "") + (f" val_ppl {ppl:.3f}" if ppl else ""))
    if st != "done":
        log(f"{job.name} did NOT finish; reported, not restarted"
            + (" (no checkpoints in this queue)" if not CKPT_EVERY else ""))
    state["runs_end"][job.name] = st or "failed"
    status_path = write_status(state)
    pushed = git_push(f"Cloud dense: {job.name} {st or 'failed'}" + (f", val PPL {ppl:.3f}" if ppl else ""),
                      small_files(job.out) + [os.path.join(OUT, "queue.log"), status_path])
    ok = backup(run_files(job.out, with_ckpt=st != "done"))
    backup_ok[job.name] = ok
    notify(f"{job.name} {st or 'FAILED (not restarted)'}" + (f", val PPL {ppl:.3f}" if ppl else "")
           + f"; {machine()} {pod_hours():.2f} h ≈ {pod_hours() * PRICE:.2f} {CUR}"
           + ("" if pushed else "; git push FAILED")
           + ("" if ok else "; Backup FAILED"))


def run_to_end(job):
    job.start()
    while job.alive():
        time.sleep(POLL_S / 3)
    job.wait_end()


def preflight(state):
    """Every run alone for 3 M tokens. With checkpoints in two parts: stopped after 1.5 M tokens with a checkpoint,
    then resumed to 3 M, so saving, loading and continuing have worked once on this machine."""
    speeds, ckpt_s = {}, {}
    for name, model in RUNS:
        args = PRE_ARGS + EXTRA.get(name, [])                            # extra last: argparse keeps the last value
        job = Job(name, model, PRE, args)
        done = job.status() == "done" and (not CKPT_EVERY or read_info(job.out).get("resumes"))
        if not done and CKPT_EVERY:
            first = Job(name, model, PRE, args + ["--ckpt_every_tokens", "1.5e6", "--stop_after_tokens", "1.5e6",
                                                  "--keep_ckpt"])
            if not os.path.exists(os.path.join(job.out, "ckpt.pt")):
                run_to_end(first)
            job.args = args + ["--ckpt_every_tokens", "1.5e6", "--resume"]
            if os.path.exists(os.path.join(job.out, "ckpt.pt")):
                run_to_end(job)
            else:
                log(f"preflight {name}: first part left no checkpoint, see {job.out}/stdout.log")
        elif not done:
            run_to_end(job)
        info = read_info(job.out)
        res = info.get("results", {})
        if info.get("status") == "done" and (not CKPT_EVERY or info.get("resumes")):
            speeds[name] = res.get("train_tok_s_median")
            state["peaks"][name] = res.get("peak_train_vram_gib")
            ck = info.get("checkpoint") or {}
            ckpt_s[name] = ck.get("save_s", 0)
            log(f"preflight {name}: {speeds[name]:.0f} tok/s alone, peak {res.get('peak_train_vram_gib'):.1f} GiB"
                + (f", checkpoint {ck.get('gib')} GiB saved in {ck.get('save_s')} s, resumed ok" if CKPT_EVERY else ""))
        else:
            log(f"preflight {name} FAILED (status {info.get('status')}), see {job.out}/stdout.log")
        for f in ("model.pt", "ckpt.pt"):
            if os.path.exists(os.path.join(job.out, f)):
                os.remove(os.path.join(job.out, f))
    state["preflight_tok_s"] = speeds
    state["preflight_ckpt_s"] = ckpt_s
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
    if PLATFORM != "runpod" and not os.environ.get("SMLM_STOP_CMD"):
        log(f"no self-stop ({reason}): not a Runpod Pod and no SMLM_STOP_CMD, the instance is deleted by hand")
        return
    r = subprocess.run(["bash", os.path.join(ROOT, "cloud", "stop_pod.sh")], cwd=ROOT, capture_output=True, text=True)
    log(f"Pod stop ({reason}): rc={r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
    if r.returncode != 0:
        notify(f"{machine()} konnte sich NICHT selbst stoppen - bitte in der "
               f"{'Runpod' if PLATFORM == 'runpod' else 'Cloud'}-Konsole stoppen!")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PRE, exist_ok=True)
    state = {"phase": "preflight", "peaks": {}, "runs_end": {}, "cap_h": round(CAP_H, 3)}
    log(f"dense queue start ({PLATFORM}): cap {CAP_USD} {CUR} at {PRICE} {CUR}/h = {CAP_H:.2f} h, "
        f"uptime {pod_hours():.2f} h" + (f", checkpoints every {CKPT_ARG} tokens" if CKPT_EVERY else ""))
    notify(f"Dense-Läufe: Setup fertig, Probelauf startet ({machine()} {pod_hours():.2f} h)")
    speeds = preflight(state)
    left = {n: remaining(n) for n, _ in RUNS}
    admitted, dropped, proj = admit(speeds, pod_hours(), left=left, ckpt_s=state["preflight_ckpt_s"])
    state.update({"phase": "runs", "admitted": admitted, "dropped": dropped, "projected_hours": round(proj, 2),
                  "projected_cost_usd": round(proj * PRICE, 2)})
    log(f"budget gate: admitted {admitted}, dropped {dropped}, projected {proj:.2f} h ≈ {proj * PRICE:.2f} {CUR}"
        + (f", still to train {({n: round(v, 3) for n, v in left.items() if v < 1})}" if any(v < 1 for v in left.values())
           else ""))
    notify(f"Budget-Wächter: laufen {', '.join(a.split('-s0')[0] for a in admitted) or 'nichts'}"
           + (f"; weggelassen {', '.join(d[0].split('-s0')[0] + ' (' + d[1] + ')' for d in dropped)}" if dropped else "")
           + f"; Hochrechnung {proj:.1f} h ≈ {proj * PRICE:.0f} {CUR}")
    write_status(state)
    jobs = [Job(n, m, OUT, ARGS + EXTRA.get(n, [])) for n, m in RUNS if n in admitted]
    if START_ORDER != "listed":
        jobs.sort(key=lambda j: speeds.get(j.name) or 0)                   # slowest (= longest) first
    for j in jobs:
        has_ckpt = os.path.exists(os.path.join(j.out, "model.pt"))
        if j.status() == "done" and has_ckpt:
            j.done = True
        elif j.adopt():
            j.reserve_gib = 1.1 * (state["peaks"].get(j.name) or 20) + 1.5
        elif j.status() is not None and j.status() != "done" and may_restart(j):
            note_restart(j, f"Prozess nach Neustart der Queue weg (Status {j.status()})")
            log(f"{j.name}: status {j.status()}, process gone when the scheduler restarted - resumed from its checkpoint")
        elif j.status() is not None:
            # started earlier and the process is gone, or "done" without model.pt (e.g. a run-info.json from git
            # in a fresh clone): report, do not restart
            j.done = True
            why = "model.pt missing" if j.status() == "done" else "process gone when the scheduler restarted"
            state["runs_end"][j.name] = f"{j.status()} ({why}, not restarted)"
            log(f"{j.name}: status {j.status()}, {why} - reported, not restarted")
    total = gpu_total_gib() - 2.0
    backup_ok = {}
    hard_stop = False
    while not all(j.done for j in jobs):
        for j in jobs:
            if j.done or (j.proc is None and j.pid is None):
                continue
            if not j.alive():
                finish(j, state, backup_ok)                                # restarts it if it may
            elif j.stalled():
                j.kill(f"stalled (no log change for {STALL_MIN:.0f} min)")
        if pod_hours() >= CAP_H - HARD_MARGIN_H and not hard_stop:
            hard_stop = True
            notify(f"Kostendeckel erreicht ({pod_hours():.2f} h): laufende Läufe werden beendet")
            for j in jobs:
                if not j.done and (j.proc is not None or j.pid is not None):
                    j.no_restart = True
                    j.kill("cost cap")
                elif not j.done:
                    j.done = True
                    state["runs_end"][j.name] = "not started (cost cap)"
        for j in jobs:
            if j.done or j.proc is not None or j.pid is not None or hard_stop:
                continue
            need = 1.1 * (state["peaks"].get(j.name) or 20) + 1.5
            running = [x for x in jobs if not x.done and (x.proc is not None or x.pid is not None)]
            used = sum(x.reserve_gib for x in running)
            if MAX_PARALLEL and len(running) >= MAX_PARALLEL:
                break
            if START_ORDER == "listed" and running and used + need > total:
                break                                       # keep the order: the next one waits for room
            if used + need <= total or not running:         # alone it always starts (it ran alone in the preflight)
                j.reserve_gib = need
                j.start()
        time.sleep(POLL_S)
    state["phase"] = "end"
    m = manifest()
    status_path = write_status(state)
    git_push("Cloud dense: queue finished, checkpoint manifest", [m, os.path.join(OUT, "queue.log"), status_path])
    # an unfinished run keeps its checkpoint on the box (the only way to continue it elsewhere), half-written files don't
    files = [os.path.join(dp, f) for top in (OUT, PRE) for dp, _, fs in os.walk(top) for f in fs
             if not (".tmp" in f and f.rsplit(".tmp", 1)[1].isdigit())]
    ok = backup(files)
    if ok:
        marker = os.path.join(OUT, "BACKUP_OK")
        with open(marker, "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} all files of {os.path.relpath(OUT, ROOT)} verified on {BOX}\n")
        backup([marker])
        # BACKUP_OK only says the backup worked; "Fertig" only if every admitted run finished
        missing = [f"{j.name} ({state['runs_end'][j.name]})" for j in jobs
                   if state["runs_end"].get(j.name, "done") != "done"]
        log("QUEUE DONE, backup verified" + (f", NOT finished: {', '.join(missing)}" if missing else ""))
        head = ("Fertig" if not missing else f"UNVOLLSTÄNDIG, nicht fertig: {', '.join(missing)}. Der Rest") + ": " \
            + ("" if NO_PUSH else "Ergebnisse gepusht, ") + "Checkpoints auf der Storage Box geprüft"
        cost = f"{machine()} {pod_hours():.2f} h ≈ {pod_hours() * PRICE:.2f} {CUR}"
        if PLATFORM == "runpod":
            notify(f"{head}. Claude stoppt und löscht jetzt den Pod ({cost}). Falls nicht: Selbst-Stopp in "
                   f"{FALLBACK_MIN:.0f} min.")
        else:
            notify(f"{head} ({cost}). Claude prüft das Backup von zu Hause und meldet sich; erst dann die Instanz "
                   "löschen.")
        backup([os.path.join(OUT, "queue.log")], quiet=True)
        if PLATFORM == "runpod":
            time.sleep(FALLBACK_MIN * 60 if not FAKE else 1)
            stop_pod("fallback: not deleted from home within the waiting time")
            notify("Pod hat sich selbst gestoppt (Fallback). Gestoppt kostet er nur noch Speicher; Claude löscht ihn.")
    else:
        log("QUEUE DONE, BACKUP FAILED")
        if PLATFORM == "runpod":
            notify("Fertig, aber das Sichern auf die Storage Box ist FEHLGESCHLAGEN. Der Pod wird nur gestoppt (nicht "
                   "gelöscht), die Daten liegen weiter auf seinem Volume.")
            stop_pod("backup failed")
        else:
            notify("Fertig, aber das Sichern auf die Storage Box ist FEHLGESCHLAGEN. Instanz NICHT löschen, die "
                   "Daten liegen nur auf ihr.")


if __name__ == "__main__":
    main()
