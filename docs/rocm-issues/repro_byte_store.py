"""Minimal check: masked uint8 stores from Triton programs must not touch bytes of other programs.

Each program owns BR consecutive bytes of a 0/1 flag array. It loads its flags, writes how many were set to
count[pid], then clears its own set flags with a masked byte store (the pattern of a fused optimizer step that clears
a 'touched' mask). Expected: count[pid] == number of set flags in the original slice of pid, and all flags 0 at the end.
If a byte store is widened beyond its mask/range, a neighbouring program can read 0 for a flag that was 1.
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def clear_kernel(flags_ptr, count_ptr, n, BR: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BR + tl.arange(0, BR)
    inb = offs < n
    f = tl.load(flags_ptr + offs, mask=inb, other=0)
    t = f != 0
    tl.store(count_ptr + pid, tl.sum(t.to(tl.int32), axis=0))
    if MODE == 0:      # clear only the flags that were set (data-dependent mask)
        tl.store(flags_ptr + offs, tl.zeros([BR], dtype=tl.uint8), mask=inb & t)
    else:              # clear the whole own slice (bounds mask only)
        tl.store(flags_ptr + offs, tl.zeros([BR], dtype=tl.uint8), mask=inb)


def run(n, BR, mode, density, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    flags = (torch.rand(n, device="cuda", generator=g) < density).to(torch.uint8)
    want = flags.view(-1)[: (n // BR) * BR].view(-1, BR).sum(1).int()
    if n % BR:
        want = torch.cat([want, flags[(n // BR) * BR:].sum().int().view(1)])
    count = torch.empty(triton.cdiv(n, BR), dtype=torch.int32, device="cuda")
    clear_kernel[(triton.cdiv(n, BR),)](flags, count, n, BR=BR, MODE=mode)
    torch.cuda.synchronize()
    bad_counts = int((count != want).sum())
    left = int(flags.sum())
    return bad_counts, left


def main():
    print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__,
          torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    fails = 0
    for BR in (1, 4, 8, 16, 32, 64, 128):
        for mode in (0, 1):
            for density in (0.01, 0.3, 0.9):
                tot_bad, tot_left = 0, 0
                for seed in range(5):
                    b, l = run(1 << 22, BR, mode, density, seed)
                    tot_bad += b
                    tot_left += l
                status = "OK" if tot_bad == tot_left == 0 else "FAIL"
                fails += status == "FAIL"
                print(f"BR={BR:4d} mode={'data-mask' if mode == 0 else 'bounds  '} density={density:4.2f}: "
                      f"wrong counts {tot_bad:7d}, flags left {tot_left:7d}  {status}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
