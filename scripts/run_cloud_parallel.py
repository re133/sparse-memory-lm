"""Parallel continuation of the cloud queue (decided 2026-10-03 23:5x, user rules below).

run_cloud.py runs the three configurations one after the other, but on the H200 a single run is CPU-bound
(one Python thread at ~90 %, GPU ~24 % busy). This script takes over from it while B-1M is running:

  * B-1M-s0  keeps running untouched (started by run_cloud.py); it is only adopted: watched, finished, pushed.
             run_cloud.py itself is frozen with SIGSTOP before this script starts (killing it would close its tmux
             pane and SIGHUP its process group, which includes B-1M); once B-1M is finished this script kills it.
  * B-16M-s0 starts now on the same GPU (B-1M 10.4 GiB + B-16M 100.9 GiB peak, card 140 GiB).
  * B-4M-s0  starts only when B-1M is finished AND (B-16M is finished OR, right before the start, the GPU has
             at least max(15 GB, B-4M peak 28.5 GiB + 2 GiB) free). If the free memory cannot be read: wait.
  * A run that crashes does not touch the others (separate processes). train.py writes no mid-run checkpoints,
    so a crashed run is reported (log, push, notification) and NOT restarted.
  * At the end: checkpoint manifest, push, stop request (cloud/stop_pod.sh) - as in run_cloud.py.

Same training code, arguments, data and seeds as run_cloud.py; only the scheduling differs. Speed numbers
(tok/s, train time) of runs that shared the GPU are not comparable with single-run numbers.

  python scripts/run_cloud_parallel.py           (SMLM_FAKE_TRAIN=1: test the scheduling with dummy runs)
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_cloud as rc  # noqa: E402
from smlm.gpu_monitor import log_until  # noqa: E402

FAKE = os.environ.get("SMLM_FAKE_TRAIN") == "1"
NEED_B4M_GIB = 28.5 + 2.0
MIN_FREE_GIB = 15.0
POLL_S = 5 if FAKE else 30


def gpu_free_gib():
    if FAKE:
        return float(os.environ.get("SMLM_FAKE_FREE_GIB", "40"))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return float(out) / 1024
    except Exception:
        return None


class Job:
    def __init__(self, name, model):
        self.name, self.model = name, model
        self.out = os.path.join(rc.OUT, name)
        self.proc, self.pid = None, None
        self.stop_evt, self.thread, self.done = None, None, False
        self.result = None

    def alive(self):
        if self.proc is not None:
            return self.proc.poll() is None
        if self.pid is None:
            return False
        try:                                    # adopted: its parent (stopped run_cloud.py) cannot reap it, so a
            with open(f"/proc/{self.pid}/stat") as f:      # finished run stays as zombie ("Z")
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False

    def thermal(self, fname):
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=log_until, args=(os.path.join(self.out, fname), self.stop_evt),
                                       daemon=True)
        self.thread.start()

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        if FAKE:
            cmd = [sys.executable, "-c", "import json,sys,time,os; time.sleep(float(os.environ.get('FAKE_S','10')));"
                   f"json.dump({{'status':'done','results':{{'val_ppl':1.0}}}}, open(r'{self.out}/run-info.json','w'))"]
        else:
            cmd = [sys.executable, "-m", "smlm.train", "--model", self.model, "--out_dir", self.out, *rc.ARGS]
        rc.log(f"start {self.name} (parallel scheduler, free GPU memory {gpu_free_gib()} GiB): " + " ".join(cmd[1:]))
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        self.fh = open(os.path.join(self.out, "stdout.log"), "w")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.fh, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)       # own session: no signal reaches the others
        self.thermal("gpu_thermal.csv")

    def adopt(self):
        r = subprocess.run(["pgrep", "-f", f"smlm.train .*--out_dir {self.out}( |$)"], capture_output=True, text=True)
        pids = [int(p) for p in r.stdout.split()]
        if not pids:
            return False
        self.pid = pids[0]
        rc.log(f"adopted running {self.name} (pid {self.pid}); its first GPU log stopped with run_cloud.py, "
               "continued in gpu_thermal_cont.csv")
        self.thermal("gpu_thermal_cont.csv")
        return True

    def stalled(self):
        files = [f for f in rc.small_files(self.out) if not f.endswith(".csv") or f.endswith("train_log.csv")]
        files = [f for f in files if os.path.exists(f)]
        return bool(files) and time.time() - max(os.path.getmtime(f) for f in files) > rc.STALL_MIN * 60

    def kill(self, why):
        rc.log(f"{self.name}: {why}, terminating")
        try:
            os.kill(self.proc.pid if self.proc else self.pid, 15)
        except OSError:
            pass
        self.result = why

    def finish(self):
        self.done = True
        if self.stop_evt:
            self.stop_evt.set()
            self.thread.join()
        st = rc.status(self.name)
        rc_code = self.proc.returncode if self.proc else "n/a (adopted)"
        rc.log(f"end {self.name} rc={rc_code} status={st}" + (f" ({self.result})" if self.result else ""))
        if st != "done":
            rc.log(f"{self.name} did NOT finish (status {st}); train.py has no mid-run checkpoints, "
                   "so it is reported and not restarted")
        if not FAKE:
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "cloud_status.py")], cwd=ROOT)
        p = os.path.join(self.out, "run-info.json")
        info = json.load(open(p)) if os.path.exists(p) else {}
        ppl = info.get("results", {}).get("val_ppl")
        extra = [os.path.join(self.out, "gpu_thermal_cont.csv")] if self.proc is None else []
        files = rc.small_files(self.out) + [f for f in extra if os.path.exists(f)]
        pushed = (True if FAKE else
                  rc.git_push(f"Cloud: {self.name} {st or 'failed'}" + (f", val PPL {ppl:.3f}" if ppl else ""),
                              files + [os.path.join(rc.OUT, "queue.log"), "REPORT.md",
                                       os.path.join("report", "cloud_status.json")]))
        rc.notify(f"SMLM cloud: {self.name} {st or 'FAILED (not restarted)'}" + (f", val PPL {ppl:.3f}" if ppl else "")
                  + ("" if pushed else " (git push FAILED)"))


def main():
    os.makedirs(rc.OUT, exist_ok=True)
    rc.log("parallel scheduler start (takes over from run_cloud.py)")
    jobs = {name: Job(name, model) for name, model in rc.RUNS}
    b1, b4, b16 = jobs["B-1M-s0"], jobs["B-4M-s0"], jobs["B-16M-s0"]
    if rc.status(b1.name) == "done":
        b1.done = True
    elif not b1.adopt():
        if FAKE:
            b1.start()
        else:
            rc.log("B-1M-s0 is neither done nor running - starting it")
            b1.start()
    if rc.status(b16.name) == "done":
        b16.done = True
    else:
        b16.start()
    if rc.status(b4.name) == "done":
        b4.done = True
    waiting_logged = False
    while True:
        for j in jobs.values():
            if j.done or (j.proc is None and j.pid is None):
                continue
            if not j.alive():
                j.finish()
            elif j.stalled():
                j.kill(f"stalled (no log change for {rc.STALL_MIN:.0f} min)")
            elif time.time() - rc.T_START > rc.MAX_HOURS * 3600:
                j.kill("time limit")
        if b1.done and not getattr(b1, "old_queue_killed", False):
            # the frozen run_cloud.py (and its tmux pane) can go now: B-1M is finished, nothing else of it runs
            subprocess.run(["pkill", "-KILL", "-f", r"scripts/run_cloud\.py"], capture_output=True)
            b1.old_queue_killed = True
        if not b4.done and b4.proc is None and b1.done:
            free = gpu_free_gib()
            if b16.done:
                b4.start()
            elif free is not None and free >= max(MIN_FREE_GIB, NEED_B4M_GIB):
                b4.start()
            elif not waiting_logged:
                rc.log(f"B-4M waits: B-16M still running and free GPU memory {free} GiB < "
                       f"{max(MIN_FREE_GIB, NEED_B4M_GIB)} GiB (or unreadable)")
                waiting_logged = True
        if all(j.done for j in jobs.values()):
            break
        if time.time() - rc.T_START > rc.MAX_HOURS * 3600 and b4.proc is None and not b4.done:
            rc.log("time limit reached before B-4M could start")
            b4.done = True
        time.sleep(POLL_S)
    if FAKE:
        rc.log("QUEUE DONE (fake)")
        return
    m = rc.manifest()
    pushed = rc.git_push("Cloud: queue finished, checkpoint manifest", [m, os.path.join(rc.OUT, "queue.log")])
    rc.log("QUEUE DONE" + ("" if pushed else " (final git push FAILED - results only on this disk)"))
    r = subprocess.run(["bash", os.path.join(ROOT, "cloud", "stop_pod.sh")], cwd=ROOT, capture_output=True,
                       text=True)
    if r.returncode == 0:
        rc.notify("SMLM cloud: queue done, results pushed; Pod stop requested via the Runpod API")
        rc.log("Runpod stop requested: " + r.stdout.strip()[:300])
    else:
        rc.notify("SMLM cloud: queue done, but the Pod could NOT be stopped automatically - stop it now!")
        rc.log("Runpod stop FAILED: " + (r.stdout + r.stderr).strip()[:300])


if __name__ == "__main__":
    main()
