"""fp32 tl.atomic_add vs plain stores in Triton on this GPU: throughput and the generated ISA.

Three kernels write N fp32 values with the same addressing:
  store       plain tl.store (each address once)
  atomic      tl.atomic_add, each address once (no contention)
  atomic_hot  tl.atomic_add, 64 programs' worth of lanes hitting the same 4096 addresses (contention)
Prints GB/s-equivalent and which atomic instructions appear in the AMDGCN (or PTX) of the atomic kernel.
"""
import re

import torch
import triton
import triton.language as tl


@triton.jit
def k_store(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(out_ptr + o, tl.load(x_ptr + o, mask=m), mask=m)


@triton.jit
def k_atomic(x_ptr, out_ptr, n, HOT: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tgt = o % 4096 if HOT else o
    tl.atomic_add(out_ptr + tgt, tl.load(x_ptr + o, mask=m), mask=m, sem="relaxed")


def bench(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    dev = torch.cuda.get_device_properties(0)
    print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__, dev.name,
          getattr(dev, "gcnArchName", ""))
    n = 1 << 24                                    # 16 M fp32 = 64 MB
    x = torch.randn(n, device="cuda")
    out = torch.zeros(n, device="cuda")
    B = 1024
    grid = (triton.cdiv(n, B),)
    t_store = bench(lambda: k_store[grid](x, out, n, BLOCK=B))
    t_atomic = bench(lambda: k_atomic[grid](x, out, n, HOT=False, BLOCK=B))
    t_hot = bench(lambda: k_atomic[grid](x, out, n, HOT=True, BLOCK=B))
    gb = 2 * n * 4 / 1e9
    print(f"plain store          {t_store:7.3f} ms  ({gb / t_store * 1e3:6.1f} GB/s)")
    print(f"atomic_add, no clash {t_atomic:7.3f} ms  ({gb / t_atomic * 1e3:6.1f} GB/s)  {t_atomic / t_store:5.1f}x store")
    print(f"atomic_add, 4096 hot {t_hot:7.3f} ms  ({gb / t_hot * 1e3:6.1f} GB/s)  {t_hot / t_store:5.1f}x store")
    compiled = k_atomic.warmup(x, out, n, HOT=False, BLOCK=B, grid=grid)
    asm = compiled.asm.get("amdgcn") or compiled.asm.get("ptx") or ""
    ops = sorted(set(re.findall(r"\b(global_atomic\w*|buffer_atomic\w*|flat_atomic\w*|atom\.\S+|red\.\S+)", asm)))
    loops = len(re.findall(r"cmpswap|atom\.cas", asm))
    print("atomic instructions in the atomic kernel:", ops or "none found", "| compare-and-swap occurrences:", loops)


if __name__ == "__main__":
    main()
