"""Train in the background and step aside for games (HEIM 1): the run pauses itself and continues exactly later.

  python scripts/heim.py [--resume_min 10] -- <smlm.train arguments, including --out_dir>
  python scripts/heim.py --probe                 (prints what the watcher sees right now, starts nothing)
  touch ~/.smlm-pause                            (pause by hand; remove the file to let the run continue)

Runs `python -m smlm.train <arguments> --resume` and looks at the PC every 5 s. A pause is due when
  - another process keeps the GPU's graphics engine at least 25 % busy for 15 s (a game; read from /proc/*/fdinfo of
    the amdgpu driver; ROCm training runs on the compute queues and doesn't count there), or
  - a Steam game is running (Steam starts every game through `reaper SteamLaunch ...`), or
  - ~/.smlm-pause exists.
Then the run gets SIGUSR1: it writes ckpt.pt at its next log row and exits with code 75 (smlm/train.py). Once nothing
of that has been seen for --resume_min minutes, the run is started again with --resume and continues bit for bit.
Exit code 0 of the run ends heim.py; any other code is reported and ends it too. Stopping heim.py itself (SIGTERM or
SIGINT, e.g. `systemctl --user stop`) pauses the run first, so nothing is lost. ntfy message on pause, continue and
end (NTFY_TOPIC or ~/smlm-cloud-kit/cloud.env); log in <out_dir>/heim.log, the run's output in <out_dir>/stdout.log.

A pause costs one checkpoint write and the steps since the last log row (--log_every, default 10). Checkpoint size:
B-1M ~5 GB with Adam, ~3.4 GB with rowrms; B-16M with its table in RAM ~77 GB with Adam, ~26 GB with rowrms. The
checkpoint goes to /dev/shm/smlm-heim/<run> (RAM, not the SSD; --ckpt_dir overrides it), so while paused it holds
that much RAM, and a reboot during a pause loses the run's progress since its start.
"""
import os
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PAUSED = 75                                  # smlm.train.PAUSED (not imported: that would load torch here)
PAUSE_FILE = os.path.expanduser("~/.smlm-pause")
INTERVAL = 5.0
GFX_BUSY = 0.25
GFX_SECONDS = 15.0


def descendants(pid):
    """pid and all processes below it."""
    children = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                ppid = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(p))
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        out.add(p)
        todo += children.get(p, [])
    return out


def gfx_time():
    """{pid: (name, graphics engine ns summed over its DRM clients)} of every process using the GPU."""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{p}/fdinfo")
        except OSError:
            continue
        clients, total = set(), 0
        for fd in fds:
            try:
                text = open(f"/proc/{p}/fdinfo/{fd}").read()
            except OSError:
                continue
            client = re.search(r"drm-client-id:\s*(\d+)", text)
            if "drm-driver" not in text or not client or client.group(1) in clients:
                continue
            clients.add(client.group(1))
            total += sum(int(n) for n in re.findall(r"drm-engine-gfx:\s*(\d+) ns", text))
        if clients:
            try:
                name = open(f"/proc/{p}/comm").read().strip()
            except OSError:
                name = "?"
            out[int(p)] = (name, total)
    return out


def steam_game():
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if b"SteamLaunch" in cmd:
                return " ".join(a.decode(errors="replace") for a in cmd if a.startswith(b"AppId="))
    return None


class Watcher:
    """Remembers graphics engine times between looks; why() says why a pause is due now, or returns None."""

    def __init__(self, interval=INTERVAL):
        self.interval, self.last, self.busy_since = interval, None, {}

    def why(self, own=()):
        if os.path.exists(PAUSE_FILE):
            return f"{PAUSE_FILE} exists"
        game = steam_game()
        if game is not None:
            return f"Steam game running ({game or 'SteamLaunch'})"
        now, times = time.time(), gfx_time()
        last, self.last = self.last, (now, times)
        if last is None:
            return None
        dt = max(now - last[0], 1e-3)
        for pid, (name, ns) in times.items():
            if pid in own or pid not in last[1]:
                continue
            busy = (ns - last[1][pid][1]) / 1e9 / dt
            if busy >= GFX_BUSY:
                since = self.busy_since.setdefault(pid, last[0])
                if now - since >= GFX_SECONDS:
                    return f"{name} (pid {pid}) keeps the GPU {100 * busy:.0f} % busy"
            else:
                self.busy_since.pop(pid, None)
        return None


def ntfy_topic():
    topic = os.environ.get("NTFY_TOPIC")
    kit = os.path.expanduser("~/smlm-cloud-kit/cloud.env")
    if not topic and os.path.exists(kit):
        for line in open(kit):
            if line.startswith("NTFY_TOPIC="):
                topic = line.split("=", 1)[1].strip().strip('"')
    return topic


def supervise(cmd, out_dir, why, resume_s, interval=INTERVAL, notify=None, sleep=time.sleep):
    """Run cmd (it must pause on SIGUSR1 with exit code 75) until it ends with another code; returns that code.
    why(own_pids) -> reason or None; resume_s: quiet seconds before continuing after a pause."""
    os.makedirs(out_dir, exist_ok=True)

    def log(msg, push=False):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(os.path.join(out_dir, "heim.log"), "a") as f:
            f.write(line + "\n")
        if push and notify:
            notify(msg)

    stop = []
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_: stop.append(1))
    quiet_since, starts = None, 0
    while True:
        if stop:
            log("heim stopped while paused")
            return PAUSED
        reason = why(set())
        if reason:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        if reason is None and (starts == 0 or time.time() - quiet_since >= resume_s):
            with open(os.path.join(out_dir, "stdout.log"), "a") as fh:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            log(("start" if starts == 0 else "continue") + f" (pid {p.pid})", push=starts > 0)
            starts += 1
            asked = None
            while p.poll() is None:
                sleep(interval)
                reason = "heim.py was stopped" if stop else why(descendants(p.pid))
                if reason and asked is None:
                    log(f"pause: {reason}", push=True)
                    os.kill(p.pid, signal.SIGUSR1)
                    asked = time.time()
            if p.returncode != PAUSED:
                log(f"run ended with exit code {p.returncode}", push=True)
                return p.returncode
            log(f"paused after {time.time() - asked:.0f} s" if asked else "the run paused by itself")
            quiet_since = None
        sleep(interval)


def main():
    argv = sys.argv[1:]
    if argv == ["--probe"]:
        w = Watcher()
        w.why()
        time.sleep(INTERVAL)
        print("pause due:", w.why() or "no")
        last = w.last[1]
        for pid, (name, ns) in sorted(last.items()):
            print(f"  {pid:>7} {name:<20} graphics engine {ns / 1e9:10.1f} s in total")
        return
    resume_min = 10.0
    if len(argv) >= 2 and argv[0] == "--resume_min":
        resume_min, argv = float(argv[1]), argv[2:]
    if not argv or argv[0] != "--" or "--out_dir" not in argv:
        sys.exit(__doc__)
    train_args = argv[1:]
    out_dir = os.path.abspath(train_args[train_args.index("--out_dir") + 1])
    cmd = [sys.executable, "-m", "smlm.train", *train_args] + ([] if "--resume" in train_args else ["--resume"])
    if "--ckpt_dir" not in train_args:
        cmd += ["--ckpt_dir", os.path.join("/dev/shm", "smlm-heim", os.path.basename(out_dir))]
    topic = ntfy_topic()

    def notify(msg):
        if topic:
            subprocess.run(["curl", "-s", "-m", "20", "-H", "Title: SMLM heim", "-d",
                            f"{os.path.basename(out_dir)}: {msg}", f"https://ntfy.sh/{topic}"], capture_output=True)

    watcher = Watcher()
    sys.exit(supervise(cmd, out_dir, watcher.why, resume_min * 60, notify=notify))


if __name__ == "__main__":
    main()
