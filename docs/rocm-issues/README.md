# ROCm notes and reproducers (RX 9070, gfx1201)

Environment of the measurements: Radeon RX 9070 (gfx1201), ROCm 7.2.4, PyTorch 2.14 (HIP 7.2), Triton 3.5.1,
Linux 7.2 (CachyOS).

| Script | What it checks | Result | Bug? |
|---|---|---|---|
| `repro_atomic_add.py` | fp32 `tl.atomic_add` against plain stores, plus the generated ISA | the native `global_atomic_add_f32`, non-returning, is used; ≈ 7.9× slower than plain stores (48 vs 383 GB/s equivalent), same with or without address contention | no, a hardware throughput property. Workaround in `smlm/kernels.py` (kernel 1): segmented scan, plain stores for rows owned by one program, atomics only at program borders |
| `repro_byte_store.py` | masked `uint8` stores of many programs to adjacent bytes (clearing a flag mask) | correct in all 42 configurations (slice sizes 1–128 bytes, data-dependent and bounds masks) | no. The corruption seen in an early version of our lazy-Adam kernel is not reproducible here and was most likely a bug in that kernel. The mask is now cleared with `zero_()` after the kernel anyway |

Not reduced to a minimal case (so not reported): two crashes of Qwen3.5-0.8B with the PyTorch fallback of Gated
DeltaNet under ROCm.
- `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` during lm-eval log-likelihood batches.
- A GPU memory access fault with gradient checkpointing.

Data-centre AMD: on an AMD Instinct MI350X (gfx950, ROCm 7.1, PyTorch 2.13, Triton 3.7.1) the whole test suite passes
(107/107) without code changes (`runs/amd_mi350x`, REPORT.md section "AMD Instinct MI350X").
