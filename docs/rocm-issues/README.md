# ROCm notes and reproducers (RX 9070, gfx1201)

Environment of the measurements: Radeon RX 9070 (gfx1201), ROCm 7.2.4, PyTorch 2.14 (HIP 7.2), Triton 3.5.1,
Linux 7.2 (CachyOS). Also checked on the same card: the official PyTorch wheels for `rocm7.1` and `rocm7.2`.

| Script | What it checks | Result | Bug? |
|---|---|---|---|
| `repro_atomic_add.py` | fp32 `tl.atomic_add` against plain stores, plus the generated ISA | the native `global_atomic_add_f32`, non-returning, is used; ≈ 7.9× slower than plain stores (48 vs 383 GB/s equivalent), same with or without address contention | no, a hardware throughput property. Workaround in `smlm/kernels.py` (kernel 1): segmented scan, plain stores for rows owned by one program, atomics only at program borders |
| `repro_byte_store.py` | masked `uint8` stores of many programs to adjacent bytes (clearing a flag mask) | correct in all 42 configurations (slice sizes 1–128 bytes, data-dependent and bounds masks) | no. The corruption seen in an early version of our lazy-Adam kernel is not reproducible here and was most likely a bug in that kernel. The mask is now cleared with `zero_()` after the kernel anyway |

Not reduced to a minimal case (so not reported):
- Two crashes of Qwen3.5-0.8B with the PyTorch fallback of Gated DeltaNet:
  - `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` during lm-eval log-likelihood batches.
  - A GPU memory access fault with gradient checkpointing.
- **`expandable_segments`:** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `scripts/kernel_speedup.py`
  died in the first prefill after the training steps of the PyTorch-reference model
  (`HSA_STATUS_ERROR_EXCEPTION`, code 0x1016). That path is plain PyTorch, no Triton. Without the setting it runs.
  The script now ignores the variable.
- **Official `rocm7.1` wheel (PyTorch 2.13.0, HIP 7.1, Triton 3.7.1) on the RX 9070:**
  - What fails: the full test suite aborts in `test_decode_graph_bit_identical[fp32]` with
    `HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed` (code 0x1009).
  - When: every time when it runs after the other tests. Alone, or with only `tests/test_kernels.py`, it passes.
  - What I tried: waiting for the GPU before the captured graphs are freed didn't change anything.
  - What works: the `rocm7.2` wheel (PyTorch 2.14.1, Triton 3.8.0) and the CachyOS ROCm 7.2 build pass everything,
    twice in a row. The same `rocm7.1` wheel also passed everything on the MI350X.
  - So on Radeon: use the `rocm7.2` wheel.

Data-centre AMD: on an AMD Instinct MI350X (gfx950, ROCm 7.1, PyTorch 2.13, Triton 3.7.1) the whole test suite passes
(107/107) without code changes (`runs/amd_mi350x`, REPORT.md section "AMD Instinct MI350X").
