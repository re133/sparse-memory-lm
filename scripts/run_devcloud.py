"""AMD Developer Cloud queue; results stay on the VM, finished steps are skipped.

  python scripts/run_devcloud.py --only env,tests,speed
  python scripts/run_devcloud.py --only all --dry_run --gpus 0,1,2,3,4,5,6,7
  python scripts/run_devcloud.py --only experiments --approval approval.json

Experiments require a recorded approval of written criteria. Interrupted steps restart from scratch in a new
attempt; train.py has no intermediate checkpoints. No git writes, remote backup, billing API or VM shutdown.
"""
import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DEFAULT_OUT = ROOT / "runs" / "devcloud"
COMMON = ["--data", "wikipedia", "--tokens", "500e6", "--extra_val", "wikitext103",
          "--eval_every_tokens", "10e6", "--seed", "1", "--data_seed", "1234"]
MEM = ["--mem_impl", "triton", "--value_lr", "2.4e-3"]
SPEED_ARGS = ["--data", "wikipedia", "--tokens", "3e6", "--extra_val", "wikitext103", "--eval_every_tokens", "3e6",
              "--seed", "0", "--data_seed", "1234", "--no_save", "--sample_windows", "0"]
SPEED = [("speed_A", "A", []), ("speed_D-100M", "D-100M", []), ("speed_D-400M", "D-400M", []),
         ("speed_B-1M-torch", "B-1M-sparse", ["--mem_impl", "torch", "--value_lr", "2.4e-3"]),
         ("speed_B-1M", "B-1M-sparse", MEM), ("speed_B-4M", "B-4M-sparse", MEM),
         ("speed_B-16M", "B-16M-sparse", MEM)]
EXPERIMENTS = [("B-16M-s1", "B-16M-sparse", MEM), ("B-4M-s1", "B-4M-sparse", MEM),
               ("D-100M-s1", "D-100M", []), ("D-200M-s1", "D-200M", []),
               ("BE-1M-s0", "BE-1M", [*MEM, "--eng_value_lr", "3e-3"])]


@dataclasses.dataclass
class Step:
    name: str
    command: list
    kind: str
    timeout_min: float


def plan(out):
    steps = [Step("env", [PY, str(Path(__file__).resolve()), "--record_env", str(out / "env" / "env.json")],
                  "env", 5),
             Step("tests", [PY, "-m", "pytest", "-q", "-rs", "tests"], "tests", 45)]
    for name, model, extra in SPEED:
        steps.append(Step(name, [PY, "-m", "smlm.train", "--model", model, "--out_dir", str(out / name),
                                 *extra, *SPEED_ARGS], "speed", 30))
    for name, model, extra in EXPERIMENTS:
        args = COMMON.copy()
        if name == "BE-1M-s0":
            args[args.index("--seed") + 1] = "0"
        steps.append(Step(name, [PY, "-m", "smlm.train", "--model", model, "--out_dir", str(out / name),
                                 *extra, *args], "experiment", 480))
    return steps


def select_steps(steps, only):
    names = {s.name for s in steps}
    groups = {"speed": {s.name for s in steps if s.kind == "speed"},
              "experiments": {s.name for s in steps if s.kind == "experiment"}, "all": names}
    wanted = set()
    for item in only.split(","):
        item = item.strip()
        if item not in names and item not in groups:
            raise ValueError(f"unknown step/group: {item}; choose {', '.join(sorted(names | groups.keys()))}")
        wanted.update(groups.get(item, {item}))
    return [s for s in steps if s.name in wanted]


def clean_env(gpu=None):
    env = dict(os.environ)
    # Physical ROCr ordinals are the only device mask; mixing masks renumbers cards twice.
    for key in list(env):
        if (key.startswith("PYTORCH_") and key.endswith("ALLOC_CONF")) or key in (
                "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                "HSA_OVERRIDE_GFX_VERSION", "TRITON_INTERPRET"):
            env.pop(key)
    env["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / "triton")
    env["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    env["TIKTOKEN_CACHE_DIR"] = str(ROOT / ".cache" / "tiktoken")
    env["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / ".cache" / "torchinductor")
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["ROCR_VISIBLE_DEVICES"] = str(gpu)
    return env


def inventory():
    code = ("import json, platform, torch, triton, transformers; "
            "from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig; "
            "print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__, "
            "hip=torch.version.hip, triton=triton.__version__, transformers=transformers.__version__, "
            "kernel=platform.release(), devices=[dict(index=i, name=p.name, "
            "arch=getattr(p,'gcnArchName',''), memory_bytes=p.total_memory) "
            "for i in range(torch.cuda.device_count()) for p in [torch.cuda.get_device_properties(i)]])))")
    result = subprocess.run([PY, "-c", code], env=clean_env(), capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout)


def gpu_ids(spec, info=None):
    if spec == "auto":
        ids = [d["index"] for d in info["devices"]] if info else [0]
    else:
        try:
            ids = [int(x) for x in spec.split(",")]
        except ValueError:
            raise ValueError("--gpus must be auto or comma-separated physical GPU ordinals") from None
    if not ids or len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("--gpus must contain distinct non-negative GPU ordinals")
    if info is not None:
        devices = {d["index"]: d for d in info["devices"]}
        if not str(info.get("hip", "")).startswith("7.2"):
            raise ValueError("ROCm 7.2 PyTorch required; run cloud/setup_devcloud.sh")
        for i in ids:
            if i not in devices or devices[i]["arch"].split(":")[0] != "gfx942":
                raise ValueError(f"GPU {i} is absent or not gfx942; refusing to run this queue")
    return ids


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    try:
        with open(path) as f:
            value = json.load(f)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_hash():
    paths = sorted([*ROOT.glob("smlm/*.py"), *ROOT.glob("tests/*.py"), Path(__file__).resolve(),
                    ROOT / "requirements.txt", ROOT / "cloud" / "data_sha256.txt"])
    return hashlib.sha256("".join(str(p.relative_to(ROOT)) + sha256(p) for p in paths).encode()).hexdigest()


def signature(step, source):
    return hashlib.sha256(json.dumps([step.command, source]).encode()).hexdigest()


def artifacts_ok(step, out):
    d = out / step.name
    if step.kind == "env":
        return bool(read_json(d / "env.json").get("devices"))
    if step.kind == "tests":
        return (d / "stdout.log").is_file()
    if read_json(d / "run-info.json").get("status") != "done":
        return False
    if step.kind == "experiment":
        return (d / "model.pt").is_file() and (d / "model.pt").stat().st_size > 0
    return True


def done(step, out, source):
    marker = read_json(out / step.name / "DONE")
    return marker.get("signature") == signature(step, source) and artifacts_ok(step, out)


def verify_data(root=ROOT):
    entries = []
    for line in (root / "cloud" / "data_sha256.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split()
        if name.startswith(("wikipedia_en_gpt2/", "wikitext103_gpt2/")):
            path = root / "data" / name
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"missing/corrupt uploaded data: {path}; rsync again, do not rebuild it")
            entries.append(name)
    if not entries:
        raise ValueError("no prepared token files in cloud/data_sha256.txt")
    return entries


def approval_record(path, steps):
    selected = [s.name for s in steps if s.kind == "experiment"]
    if not selected:
        return None
    if path is None:
        raise ValueError("experiments need --approval FILE with approved written criteria; see cloud/DEVCLOUD.md")
    path = Path(path).resolve()
    record = read_json(path)
    criteria = Path(record.get("criteria_file", ""))
    if not criteria.is_absolute():
        criteria = path.parent / criteria
    if (record.get("approved") is not True or not isinstance(record.get("steps"), list)
            or not all(isinstance(name, str) for name in record["steps"])
            or not set(selected) <= set(record["steps"]) or not criteria.is_file()
            or not criteria.read_text().strip() or sha256(criteria) != record.get("criteria_sha256")):
        raise ValueError("approval missing, incomplete or criteria hash changed; see cloud/DEVCLOUD.md")
    return {"approval": record, "criteria": criteria.read_text()}


def log(out, text):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + text
    print(line, flush=True)
    with open(out / "queue.log", "a") as f:
        f.write(line + "\n")


class Job:
    def __init__(self, step, gpu, out, source, stall_min):
        self.step, self.gpu, self.out, self.source = step, gpu, out, source
        self.stall_min, self.proc, self.fh = stall_min, None, None
        self.started = time.monotonic()
        self.d = out / step.name

    def start(self):
        if self.d.exists():
            # Retain partial logs/checkpoints instead of mixing them with the new attempt.
            old = read_json(self.d / "attempt.json")
            if old.get("pid") and Path(f"/proc/{old['pid']}").exists() and old.get("status") == "running":
                raise ValueError(f"{self.step.name}: previous PID {old['pid']} still exists; inspect it first")
            if (self.d / "DONE").exists():
                raise ValueError(f"{self.step.name}: stale/incomplete DONE; use a new --out_dir or inspect the result")
            attempts = self.out / "attempts"
            attempts.mkdir(exist_ok=True)
            self.d.rename(attempts / f"{self.step.name}-{time.time_ns()}")
        self.d.mkdir(parents=True)
        log(self.out, f"start {self.step.name} GPU {self.gpu}: {shlex.join(self.step.command)}")
        self.fh = open(self.d / "stdout.log", "w")
        self.proc = subprocess.Popen(self.step.command, cwd=ROOT, env=clean_env(self.gpu), stdout=self.fh,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        self.started = time.monotonic()
        write_json(self.d / "attempt.json", {"status": "running", "pid": self.proc.pid, "gpu": self.gpu,
                   "command": self.step.command, "signature": signature(self.step, self.source)})

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()

    def poll(self):
        elapsed = time.monotonic() - self.started
        reason = None
        if self.proc.poll() is None:
            if elapsed > self.step.timeout_min * 60:
                reason = "timeout"
            elif self.step.kind in ("speed", "experiment"):
                newest = max(p.stat().st_mtime for p in self.d.iterdir()
                             if p.name in ("stdout.log", "metrics.csv", "train_log.csv"))
                if time.time() - newest > self.stall_min * 60:
                    reason = "stalled"
            if reason is None:
                return None
            self.stop()
        self.fh.close()
        ok = reason is None and self.proc.returncode == 0 and artifacts_ok(self.step, self.out)
        result = "done" if ok else reason or "failed"
        record = {"status": result, "returncode": self.proc.returncode, "duration_s": elapsed,
                  "gpu": self.gpu, "command": self.step.command, "signature": signature(self.step, self.source)}
        write_json(self.d / "attempt.json", record)
        if ok:
            write_json(self.d / "DONE", record)
        log(self.out, f"end {self.step.name}: {result}, rc={self.proc.returncode}, {elapsed:.1f} s")
        return result


def execute(steps, gpus, out, source, stall_min=30, parallel=False):
    pending, active, results = list(steps), {}, {}
    slots = gpus if parallel else gpus[:1]
    try:
        while pending or active:
            for gpu in slots:
                if gpu in active:
                    continue
                while pending and done(pending[0], out, source):
                    step = pending.pop(0)
                    log(out, f"skip {step.name} (done)")
                    results[step.name] = "done"
                if pending:
                    job = Job(pending.pop(0), gpu, out, source, stall_min)
                    active[gpu] = job
                    job.start()
            for gpu, job in list(active.items()):
                result = job.poll()
                if result is not None:
                    results[job.step.name] = result
                    del active[gpu]
                    if result != "done" and not parallel:
                        return results
            if active:
                time.sleep(0.25)
    finally:
        for job in active.values():
            job.stop()
            if job.fh:
                job.fh.close()
    return results


def bundle(out):
    """Uncompressed tar avoids spending GPU rental time compressing large floating-point checkpoints."""
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    if any(p.is_symlink() for p in out.rglob("*")):
        raise ValueError("refusing to package symlinks in results")
    manifest = out / "SHA256SUMS"
    manifest.write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in paths))
    target = out.with_suffix(out.suffix + ".tar")
    tmp = target.with_name(target.name + ".tmp")
    with tarfile.open(tmp, "w") as archive:
        archive.add(out, arcname=out.name)
    tmp.replace(target)
    target.with_name(target.name + ".sha256").write_text(f"{sha256(target)}  {target.name}\n")
    print(f"Pull {target} and {target}.sha256; or rsync {out}/ (verify SHA256SUMS there).", flush=True)


def preview(steps, gpus, out, source):
    print("DRY RUN: no processes, GPU probes, data reads, directories, notifications or archives are created.")
    print("At execution: verify gfx942/ROCm 7.2, uploaded data hashes, approval and completed preflight gates.")
    experiments = 0
    for step in steps:
        gpu = gpus[experiments % len(gpus)] if step.kind == "experiment" else gpus[0]
        if step.kind == "experiment":
            experiments += 1
        status = "SKIP done" if done(step, out, source) else "RUN"
        print(f"{status} {step.name}: env ROCR_VISIBLE_DEVICES={gpu} "
              f"TRITON_CACHE_DIR={shlex.quote(str(ROOT / '.cache' / 'triton'))} {shlex.join(step.command)}")
    print("Experimental GPU assignment shown for initial slots; later jobs use the first free GPU.")
    print(shlex.join([PY, str(Path(__file__).resolve()), "--out_dir", str(out), "--pack_only"]))


def main(argv=None):
    global PY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="env,tests,speed", help="names, speed, experiments or all; priority order is fixed")
    ap.add_argument("--gpus", default="auto", help="physical ROCr ordinals, e.g. 0 or 0,1,2,3,4,5,6,7")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--approval", type=Path)
    ap.add_argument("--dry_run", action="store_true", help="print commands only; auto assumes GPU 0")
    ap.add_argument("--python", help="interpreter path for a setup preview (requires --dry_run)")
    ap.add_argument("--stall_min", type=float, default=30)
    ap.add_argument("--pack_only", action="store_true", help="rebuild result manifest/tar without any GPU work")
    ap.add_argument("--record_env", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.record_env and args.dry_run:
        raise ValueError("--record_env is internal and cannot be combined with --dry_run")
    if args.python:
        if not args.dry_run:
            raise ValueError("--python is only for --dry_run; run the queue with the intended Python directly")
        PY = args.python
    else:
        PY = sys.executable
    if args.record_env:
        info = inventory()
        gpu_ids("auto", info)
        write_json(args.record_env, info)
        print(json.dumps(info, indent=2))
        return 0
    out = args.out_dir.resolve()
    steps = select_steps(plan(out), args.only)
    source = source_hash()
    if args.dry_run:
        if args.pack_only:
            print(shlex.join([PY, str(Path(__file__).resolve()), "--out_dir", str(out), "--pack_only"]))
            return 0
        preview(steps, gpu_ids(args.gpus), out, source)
        return 0
    if args.stall_min <= 0:
        raise ValueError("--stall_min must be positive")
    out.mkdir(parents=True, exist_ok=True)
    with open(out.parent / f".{out.name}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another queue owns {out}") from None
        if args.pack_only:
            bundle(out)
            return 0
        approval = approval_record(args.approval, steps)
        info = inventory()
        gpus = gpu_ids(args.gpus, info)
        recorded = read_json(out / "inventory.json")
        if recorded and recorded != info:
            raise ValueError("GPU/software environment changed; use a new --out_dir and repeat the preflight")
        if any(s.kind in ("speed", "experiment") for s in steps):
            verify_data()
        write_json(out / "inventory.json", info)
        if approval:
            name = hashlib.sha256(json.dumps(approval, sort_keys=True).encode()).hexdigest()
            saved = out / "approvals"
            saved.mkdir(exist_ok=True)
            write_json(saved / f"{name}.json", approval)
        def interrupted(*_):
            raise KeyboardInterrupt

        previous = signal.signal(signal.SIGTERM, interrupted)
        results = {}
        try:
            selected = {s.name for s in steps}
            # Omitting a prerequisite means reusing a completed result, never bypassing the gate.
            for step in plan(out):
                if step.kind == "experiment":
                    break
                needed = (step.name in selected or any(s.kind == "experiment" for s in steps)
                          or (step.kind in ("env", "tests") and any(s.kind == "speed" for s in steps))
                          or (step.kind == "env" and "tests" in selected))
                if not needed:
                    continue
                if step.name not in selected:
                    if not done(step, out, source):
                        raise ValueError(f"missing prerequisite {step.name}; run --only env,tests,speed first")
                    continue
                results.update(execute([step], gpus, out, source, args.stall_min))
                if results[step.name] != "done":
                    break
            if all(r == "done" for r in results.values()):
                results.update(execute([s for s in steps if s.kind == "experiment"], gpus, out, source,
                                       args.stall_min, parallel=True))
        except KeyboardInterrupt:
            log(out, "queue interrupted; child processes stopped")
            results["queue"] = "interrupted"
        except (OSError, ValueError) as e:
            log(out, str(e))
            results["queue"] = "failed"
        finally:
            signal.signal(signal.SIGTERM, previous)
        ok = all(results.get(s.name) == "done" for s in steps)
        write_json(out / "selection.json", {"selected": [s.name for s in steps], "results": results,
                                           "complete": ok, "source_sha256": source, "gpus": gpus})
        log(out, "SELECTED STEPS COMPLETE" if ok else "SELECTED STEPS INCOMPLETE; inspect selection.json")
        bundle(out)
        topic = os.environ.get("NTFY_TOPIC")
        if topic:
            subprocess.run(["curl", "-sS", "-m", "20", "-d", "SMLM DevCloud: " +
                            ("selected steps complete" if ok else "INCOMPLETE") + "; pull results from VM",
                            f"https://ntfy.sh/{topic}"], capture_output=True)
        return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"DevCloud: {e}", file=sys.stderr)
        sys.exit(1)
