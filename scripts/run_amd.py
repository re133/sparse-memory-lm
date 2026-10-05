"""AMD Instinct MI300X check (Runpod, 1 GPU): do the kernels run correctly on a data-centre AMD GPU, and how fast?

  python scripts/run_amd.py        (started by cloud/setup_amd.sh in tmux session "queue")

Steps (results in runs/amd_mi300x/<step>/, everything backed up to the storage box; nothing is pushed to GitHub):
  env         versions (PyTorch, HIP, Triton, ROCm, GPU) -> env.json
  tests       the whole test suite on the GPU (pytest; the Qwen tests need transformers and are skipped)
  speed_*     every model alone for 3 M tokens, same arguments as the H100 preflight of step 1: train tok/s, peak
              memory, decode / prefill speed (train.py's inference benchmark), val PPL after 3 M tokens
              A, D-100M, D-400M, B-1M with PyTorch reference and with Triton kernels, B-4M, B-16M (fits on one card)
  crosscheck  B-1M (Triton kernels), first 20 M tokens of the real 500 M schedule, eval every 2 M tokens, seed 0:
              the same run exists from the RX 9070 (runs/kernel_check/triton_500msched) and the H200 (runs/cloud,
              eval every 10 M) -> same numbers on three different GPUs?
GPU temperature / power / clocks every 10 s per step (gpu_thermal.csv). A step that has finished is skipped when the
script is started again. Before each step the cost cap is checked (Pod uptime); at the end: sha256 manifest, copy to
the storage box, verification there, BACKUP_OK, ntfy; Claude deletes the Pod, fallback: self-stop after 20 min.
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
import run_dense as rd  # noqa: E402  (log / notify / backup / stop_pod / pod_hours helpers)
from smlm.gpu_monitor import log_until  # noqa: E402

OUT = os.path.join(ROOT, "runs", "amd_mi300x")
rd.OUT = OUT
PY = sys.executable
PRICE = rd.PRICE
CAP_H = float(os.environ.get("SMLM_COST_CAP_USD", 6)) / PRICE
SPEED_ARGS = ["--data", "wikipedia", "--tokens", "3e6", "--extra_val", "wikitext103", "--eval_every_tokens", "3e6",
              "--seed", "0", "--data_seed", "1234", "--no_save", "--sample_windows", "0"]
MEM = ["--mem_impl", "triton", "--value_lr", "2.4e-3"]
SPEED = [("speed_A", "A", []), ("speed_D-100M", "D-100M", []), ("speed_D-400M", "D-400M", []),
         ("speed_B-1M-torch", "B-1M-sparse", ["--mem_impl", "torch", "--value_lr", "2.4e-3"]),
         ("speed_B-1M", "B-1M-sparse", MEM), ("speed_B-4M", "B-4M-sparse", MEM), ("speed_B-16M", "B-16M-sparse", MEM)]
CROSS = ["--model", "B-1M-sparse", *MEM, "--data", "wikipedia", "--tokens", "500e6", "--stop_after_tokens", "20e6",
         "--eval_every_tokens", "2e6", "--seed", "0", "--sample_windows", "0", "--no_save"]


def done(d):
    return os.path.exists(os.path.join(d, "DONE"))


def step(name, cmd, timeout_min=60):
    d = os.path.join(OUT, name)
    if done(d):
        rd.log(f"skip {name} (done)")
        return True
    if rd.pod_hours() >= CAP_H - rd.HARD_MARGIN_H:
        rd.log(f"skip {name}: cost cap ({rd.pod_hours():.2f} h of {CAP_H:.2f} h)")
        return False
    os.makedirs(d, exist_ok=True)
    rd.log(f"start {name}: {' '.join(cmd)}")
    stop = threading.Event()
    th = threading.Thread(target=log_until, args=(os.path.join(d, "gpu_thermal.csv"), stop), daemon=True)
    th.start()
    t0 = time.time()
    with open(os.path.join(d, "stdout.log"), "w") as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        try:
            rc = p.wait(timeout_min * 60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 15)
            rc = "timeout"
    stop.set()
    th.join()
    ok = rc == 0
    rd.log(f"end {name} rc={rc} ({(time.time() - t0) / 60:.1f} min)")
    if ok:
        open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    return ok


def env_info():
    d = os.path.join(OUT, "env")
    os.makedirs(d, exist_ok=True)
    code = ("import json, torch, triton, platform; p = torch.cuda.get_device_properties(0); print(json.dumps({"
            "'torch': torch.__version__, 'hip': torch.version.hip, 'triton': triton.__version__, "
            "'python': platform.python_version(), 'gpu': p.name, 'arch': getattr(p, 'gcnArchName', ''), "
            "'gpu_mem_gib': round(p.total_memory / 2**30, 1), 'gpus': torch.cuda.device_count()}))")
    info = json.loads(subprocess.run([PY, "-c", code], capture_output=True, text=True).stdout)
    for name, cmd in (("rocm_smi", ["rocm-smi", "--showproductname", "--showdriverversion", "--showvbios"]),
                      ("amd_smi", ["amd-smi", "static", "--asic", "--driver", "-g", "0"]),
                      ("cpu", ["lscpu"]), ("rocm_version", ["cat", "/opt/rocm/.info/version"])):
        try:
            info[name] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout[-3000:]
        except Exception as e:
            info[name] = f"unavailable: {e}"
    json.dump(info, open(os.path.join(d, "env.json"), "w"), indent=1)
    rd.log(f"env: torch {info['torch']} hip {info['hip']} triton {info['triton']} {info['gpu']} {info['arch']} "
           f"{info['gpu_mem_gib']} GiB")
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    rd.log(f"amd queue start: cap {CAP_H * PRICE:.1f} $ = {CAP_H:.2f} h, Pod uptime {rd.pod_hours():.2f} h")
    info = env_info()
    rd.notify(f"MI300X: Setup fertig ({info['gpu']}, {info['arch']}), Tests starten")
    ok_tests = step("tests", [PY, "-m", "pytest", "-q", "-rs", "tests", "--ignore=tests/test_qwen_memory.py"],
                    timeout_min=45)
    rd.notify(f"MI300X: Tests {'alle grün' if ok_tests else 'mit FEHLERN (siehe Log)'}; Messungen starten "
              f"(Pod {rd.pod_hours():.2f} h ≈ {rd.pod_hours() * PRICE:.2f} $)")
    for name, model, extra in SPEED:
        step(name, [PY, "-m", "smlm.train", "--model", model, "--out_dir", os.path.join(OUT, name), *extra,
                    *SPEED_ARGS], timeout_min=30)
    step("crosscheck", [PY, "-m", "smlm.train", *CROSS, "--out_dir", os.path.join(OUT, "crosscheck")],
         timeout_min=45)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(OUT) for f in fs]
    ok = rd.backup(files)
    if ok:
        marker = os.path.join(OUT, "BACKUP_OK")
        open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " all files verified on the box\n")
        rd.backup([marker])
        rd.log("QUEUE DONE, backup verified")
        rd.notify(f"MI300X fertig, Ergebnisse auf der Storage Box geprüft. Claude löscht den Pod "
                  f"(Pod {rd.pod_hours():.2f} h ≈ {rd.pod_hours() * PRICE:.2f} $).")
        time.sleep(rd.FALLBACK_MIN * 60)
        rd.stop_pod("fallback")
    else:
        rd.log("QUEUE DONE, BACKUP FAILED")
        rd.notify("MI300X fertig, aber das Sichern auf die Storage Box ist FEHLGESCHLAGEN - Pod wird nur gestoppt.")
        rd.stop_pod("backup failed")


if __name__ == "__main__":
    main()
