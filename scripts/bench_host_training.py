"""Measure host-table training on reproducible synthetic or read-only prepared token batches.

  python scripts/bench_host_training.py --model B-1M-sparse --compare --steps 3 --warmup 1
  python scripts/bench_host_training.py --model B-16M-sparse --value_device host --value_state int8
  python scripts/bench_host_training.py --device cpu --tiny --compare --steps 2 --no_profile_gpu
  python scripts/bench_host_training.py --memory_budget --ram_gb 128 --reserve_gib 16

Synthetic tokens isolate implementation parity; these losses are not a language-model quality result.
CUDA kernel time comes from one extra profiled step, separate from the unprofiled timing steps.
"""
import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
import resource
import subprocess
import sys
import time

# Set these before importing numerical libraries, including in comparison subprocesses.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(ROOT, ".cache", "triton"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import numpy as np  # noqa: E402

from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.optim import build_optimizer, clip_grads, value_table_memory  # noqa: E402
from smlm.train import MODELS  # noqa: E402


def memory_budget(args):
    """Arithmetic only: no table allocation and no claim that a theoretical limit is runnable."""
    cfg = ModelConfig(**MODELS[args.model])
    dim = cfg.mem_v_dim if cfg.mem_v_dim > 0 else cfg.d_model
    ram = int(args.ram_gib * 2**30 if args.ram_gib is not None else args.ram_gb * 10**9)
    reserve = int(args.reserve_gib * 2**30)
    if ram <= reserve:
        raise ValueError("the RAM budget must exceed the reserve")
    rows = cfg.mem_n_keys ** 2
    modes = {}
    for mode, moment_bytes in (("fp32", 8), ("bf16", 4), ("int8", 2)):
        per_row = dim * (4 + 4 + moment_bytes) + 1 + (8 if mode == "int8" else 0)
        table_bytes = rows * per_row + 4
        max_rows = (ram - reserve - 4) // per_row
        n_keys = math.isqrt(max_rows)
        modes[mode] = {
            "bytes_per_row": per_row, "preset_persistent_bytes": table_bytes,
            "preset_persistent_gib": table_bytes / 2**30,
            "preset_fits_after_reserve": table_bytes <= ram - reserve,
            "maximum_rows_arithmetic": max_rows, "maximum_pkm_n_keys_arithmetic": n_keys,
            "maximum_square_rows_arithmetic": n_keys**2,
            "maximum_square_parameters_arithmetic": n_keys**2 * dim,
            "maximum_square_persistent_gib": (n_keys**2 * per_row + 4) / 2**30,
            "maximum_power_of_two_n_keys_arithmetic": 1 << (n_keys.bit_length() - 1) if n_keys else 0,
        }
    return {
        "kind": "arithmetic_only", "model": args.model, "rows": rows, "width": dim,
        "table_parameters": rows * dim, "ram_bytes": ram, "ram_gib": ram / 2**30,
        "reserve_bytes": reserve, "reserve_gib": reserve / 2**30, "states": modes,
        "assumptions": [
            "One shared fp32 value table, fp32 gradient accumulator, bool touched mask, and one fp32 step.",
            "Int8 has two fp32 scales per row. Values remain fp32 for every state mode.",
            "Reserve is an explicit budget assumption for OS, dense model, staging, row indices, and scratch.",
            "Optimizer/clip row indices cost eight bytes per touched row; update scratch is chunk-bounded.",
            "Pinned staging and retained GPU rows depend on actual per-layer unique lookups and batch size.",
            "Largest square table is an arithmetic upper bound, not measured allocation or throughput.",
        ],
    }


def rss_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def peak_rss_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Batches:
    def __init__(self, args, vocab_size):
        self.args, self.vocab_size = args, vocab_size
        self.generator = torch.Generator().manual_seed(args.seed + 1)
        self.tokens = (np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
                       if args.data_dir else None)
        if self.tokens is not None and len(self.tokens) <= args.seq_len:
            raise ValueError("train.bin must contain at least seq_len + 1 tokens")

    def batch(self):
        args = self.args
        if self.tokens is None:
            return torch.randint(self.vocab_size, (args.micro_batch, args.seq_len + 1), generator=self.generator)
        windows = (len(self.tokens) - 1) // args.seq_len
        indices = torch.randint(windows, (args.micro_batch,), generator=self.generator).tolist()
        tokens = torch.from_numpy(np.stack([self.tokens[i * args.seq_len:(i + 1) * args.seq_len + 1]
                                           for i in indices]).astype(np.int64))
        if int(tokens.max()) >= self.vocab_size:
            raise ValueError("dataset token id exceeds model vocabulary")
        return tokens


def step(model, opt, args, generator, step_index):
    from smlm.host_values import host_timings

    device = next(model.parameters()).device
    # Transfer input tokens before the timing window, so H2D row timings contain no input transfer.
    batches = []
    for _ in range(args.accum):
        tokens = generator.batch()
        batches.append((tokens[:, :-1].contiguous().to(device), tokens[:, 1:].contiguous().to(device)))
    synchronize(device)
    if args.value_device == "host":
        host_timings(model, reset=True)
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    losses = []
    for x, y in batches:
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.amp else nullcontext()
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.detach())
        (loss / args.accum).backward()
    synchronize(device)
    forward_backward_wall = time.perf_counter() - start
    clip_start = time.perf_counter()
    norm, value_norm = clip_grads(model, args.clip)
    synchronize(device)
    clip_wall = time.perf_counter() - clip_start
    cpu_optimizer, device_optimizer = 0.0, 0.0
    for part in getattr(opt, "opts", [opt]):
        params = [p for group in part.param_groups for p in group["params"]]
        on_cpu = all(p.device.type == "cpu" for p in params)
        synchronize(device)
        opt_start = time.perf_counter()
        part.step()
        synchronize(device)
        elapsed = time.perf_counter() - opt_start
        if on_cpu:
            cpu_optimizer += elapsed
        else:
            device_optimizer += elapsed
    wall = time.perf_counter() - start
    phases = host_timings(model) if args.value_device == "host" else {}
    return {
        "step": step_index, "loss": float(torch.stack(losses).mean()),
        "wall_s": wall, "forward_backward_wall_s": forward_backward_wall,
        "clip_wall_s": clip_wall, "cpu_optimizer_wall_s": cpu_optimizer,
        "device_optimizer_wall_s": device_optimizer, "host_phases_s": phases,
        "grad_norm": float(norm), "value_grad_norm": float(value_norm) if value_norm is not None else None,
        "rss_bytes": rss_bytes(),
    }


def profile_step(model, opt, args, generator):
    """Kernel activity excludes CPU stalls; CUDA event spans across gathers would include those stalls."""
    if args.device != "cuda" or not args.profile_gpu:
        return {"available": False, "reason": "disabled or CPU execution", "gpu_compute_s": None}
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return {"available": False, "reason": "CUDA/HIP profiler activity unavailable", "gpu_compute_s": None}
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            result = step(model, opt, args, generator, args.warmup + args.steps + 1)
    except RuntimeError as exc:
        # Some HIP builds advertise activity support but cannot load the tracing runtime.
        if not any(name in str(exc).lower() for name in ("kineto", "cupti", "roctracer", "rocprofiler")):
            raise
        return {"available": False, "reason": str(exc), "gpu_compute_s": None}
    kernels, copies, memset = [], [], []
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.name.lower()
        if "memcpy" in name or "memory copy" in name:
            copies.append(event)
        elif "memset" in name or "memory set" in name:
            memset.append(event)
        else:
            kernels.append(event)
    seconds = lambda events: sum(e.time_range.elapsed_us() for e in events) / 1e6  # noqa: E731
    return {
        "available": bool(kernels), "gpu_compute_s": seconds(kernels) if kernels else None,
        "gpu_memcpy_s": seconds(copies) if kernels else None,
        "gpu_memset_s": seconds(memset) if kernels else None,
        "kernel_events": len(kernels), "copy_events": len(copies), "step": result,
        "reason": None if kernels else "profiler returned no CUDA/HIP kernel events",
        "scope": "One extra training step after timing; aggregate kernel durations, including optimizer and metrics.",
    }


def run(args):
    from smlm.host_optim import value_table_memory_by_device
    from smlm.host_values import enable_host_values

    if args.threads:                         # 0: PyTorch's default, as in smlm.train
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is unavailable; use --device cpu --tiny for a light smoke check")
    if device.type == "cpu" and not args.tiny:
        raise ValueError("CPU benchmarks require --tiny; large presets are reserved for the reviewer GPU run")
    if device.type == "cpu" and args.mem_impl != "torch":
        raise ValueError("--device cpu requires --mem_impl torch")
    torch.manual_seed(args.seed)
    cfg = (ModelConfig(vocab_size=64, d_model=16, n_layers=3, n_heads=2, ffn_hidden=32,
                       max_seq_len=args.seq_len, mem_layers=[0, 2], mem_n_keys=8, mem_heads=2,
                       mem_knn=2, mem_k_dim=8, mem_share_values=True, mem_value_grad="row_sparse")
           if args.tiny else ModelConfig(**MODELS[args.model], max_seq_len=args.seq_len))
    cfg.mem_impl = args.mem_impl
    start = time.perf_counter()
    model = Transformer(cfg)
    if args.value_device == "host":
        enable_host_values(model, profile=True)
    model.to(device).train()
    opt = build_optimizer(model, args.lr, args.value_lr, args.weight_decay, value_state=args.value_state)
    synchronize(device)
    init_s = time.perf_counter() - start
    generator = Batches(args, cfg.vocab_size)
    warmup = []
    for i in range(args.warmup):
        result = step(model, opt, args, generator, i + 1)
        warmup.append(result["loss"])
        print(f"{args.value_device} warmup {i + 1}: loss={result['loss']:.8f}", file=sys.stderr, flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for i in range(args.steps):
        result = step(model, opt, args, generator, args.warmup + i + 1)
        records.append(result)
        print(f"{args.value_device} step {i + 1}: loss={result['loss']:.8f}, wall={result['wall_s']:.6f}s",
              file=sys.stderr, flush=True)
    memory = {
        "rss_bytes": rss_bytes(), "peak_process_rss_bytes": peak_rss_bytes(),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }
    average = {key: sum(r[key] for r in records) / len(records)
               for key in ("wall_s", "forward_backward_wall_s", "clip_wall_s", "cpu_optimizer_wall_s",
                           "device_optimizer_wall_s")}
    phase_keys = set().union(*(r["host_phases_s"] for r in records))
    average["host_phases_s"] = {key: sum(r["host_phases_s"].get(key, 0) for r in records) / len(records)
                                for key in sorted(phase_keys)}
    result = {
        "args": vars(args), "torch": torch.__version__, "hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "model_config": cfg.to_dict(), "params": model.param_counts(), "initialization_wall_s": init_s,
        "data_source": ({"kind": "read_only_uint16_memmap", "directory": os.path.abspath(args.data_dir),
                         "sampling": "uniform windows with replacement, seed + 1"} if args.data_dir
                        else {"kind": "synthetic_uniform_tokens", "seed": args.seed + 1}),
        "value_table_memory": value_table_memory(model, args.value_state), "memory": memory,
        "value_table_memory_by_device": value_table_memory_by_device(model, args.value_state),
        "tokens_per_step": args.micro_batch * args.seq_len * args.accum,
        "warmup_losses": warmup, "losses": [r["loss"] for r in records], "steps": records, "mean": average,
        "profiled_extra_step": profile_step(model, opt, args, generator),
        "measurement_notes": [
            "Next-token batches and initialization match across modes; warmup updates are retained.",
            "Timed steps exclude initialization, input loading/generation/transfers, and the extra profile step.",
            "Host transfer phases are synchronized staging wall time, including allocation and launch overhead.",
            "Host phases lie inside forward/backward; do not add them to forward_backward_wall_s.",
            "Kernel sum is GPU active work in the extra profiled step, not a wall-time residual or a throughput estimate.",
            "Kernel durations may overlap; kernel, transfer, and CPU times are not an additive wall-time partition.",
            "Profiler unavailability yields null compute time, never a guessed GPU compute measurement.",
            "CPU optimizer timing includes dense Adam on CPU; on CUDA only host-resident optimizer parts are counted.",
            "Peak RSS covers this process from startup; --compare uses fresh subprocesses per mode.",
            "VRAM peaks include persistent model/optimizer tensors but exclude the extra profiled step.",
        ],
    }
    del opt, model
    gc.collect()
    return result


def compare(args):
    results = {}
    for mode in ("gpu", "host"):
        command = [sys.executable, os.path.abspath(__file__), "--value_device", mode]
        for key in ("model", "value_state", "device", "mem_impl", "steps", "warmup", "micro_batch", "seq_len",
                    "accum", "seed", "threads", "lr", "value_lr", "weight_decay", "clip"):
            command.extend(["--" + key, str(getattr(args, key))])
        if args.tiny:
            command.append("--tiny")
        if args.data_dir:
            command.extend(["--data_dir", args.data_dir])
        if not args.amp:
            command.append("--no_amp")
        if not args.profile_gpu:
            command.append("--no_profile_gpu")
        child = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True, cwd=ROOT)
        results[mode] = json.loads(child.stdout)
    curves = {mode: result["warmup_losses"] + result["losses"] for mode, result in results.items()}
    errors = [abs(a - b) for a, b in zip(curves["gpu"], curves["host"])]
    matches = all(math.isclose(a, b, abs_tol=args.loss_atol, rel_tol=args.loss_rtol)
                  for a, b in zip(curves["gpu"], curves["host"]))
    return {"runs": results, "comparison": {
        "losses_match": matches, "max_absolute_loss_difference": max(errors),
        "absolute_tolerance": args.loss_atol, "relative_tolerance": args.loss_rtol,
        "includes_warmup": True, "same_seed": args.seed,
        "note": "This checks losses only; parameter/state/gradient parity is covered by the host-value tests.",
    }}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["B-1M-sparse", "B-4M-sparse", "B-16M-sparse"], default="B-1M-sparse")
    ap.add_argument("--value_device", choices=["gpu", "host"], default="host")
    ap.add_argument("--value_state", choices=["fp32", "bf16", "int8"], default="fp32")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--mem_impl", choices=["torch", "triton"], default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--micro_batch", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads; 0 = PyTorch's default as in smlm.train (use that to estimate real training)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--value_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--no_amp", dest="amp", action="store_false", help="disable CUDA bf16 autocast")
    ap.add_argument("--no_profile_gpu", dest="profile_gpu", action="store_false")
    ap.add_argument("--tiny", action="store_true", help="small model for a light CPU/GPU smoke check")
    ap.add_argument("--data_dir", help="read-only prepared dataset directory with uint16 train.bin; otherwise synthetic")
    ap.add_argument("--compare", action="store_true", help="fresh subprocesses, same seed, fail on loss mismatch")
    ap.add_argument("--loss_atol", type=float, default=1e-4)
    ap.add_argument("--loss_rtol", type=float, default=1e-5)
    ap.add_argument("--out", help="JSON output path; the same JSON is printed to stdout")
    ap.add_argument("--memory_budget", action="store_true", help="allocation-free host RAM arithmetic")
    ram = ap.add_mutually_exclusive_group()
    ram.add_argument("--ram_gb", type=float, default=128, help="decimal GB, default 128")
    ram.add_argument("--ram_gib", type=float, default=None, help="binary GiB instead of decimal GB")
    ap.add_argument("--reserve_gib", type=float, default=16, help="explicit OS/model/staging/scratch assumption")
    args = ap.parse_args()
    args.mem_impl = args.mem_impl or ("torch" if args.device == "cpu" else "triton")
    args.micro_batch = args.micro_batch if args.micro_batch is not None else (2 if args.tiny else 4)
    args.seq_len = args.seq_len if args.seq_len is not None else (8 if args.tiny else 1024)
    if min(args.steps, args.micro_batch, args.seq_len, args.accum) < 1 or args.warmup < 0:
        ap.error("steps, batch, sequence length and accumulation must be positive; warmup must be nonnegative")
    if args.loss_atol < 0 or args.loss_rtol < 0 or args.reserve_gib < 0:
        ap.error("tolerances and reserve must be nonnegative")
    if args.tiny and args.data_dir:
        ap.error("--tiny has a small synthetic vocabulary and cannot be used with --data_dir")
    result = memory_budget(args) if args.memory_budget else compare(args) if args.compare else run(args)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(encoded + "\n")
    print(encoded, flush=True)
    if args.compare and not args.memory_budget and not result["comparison"]["losses_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
