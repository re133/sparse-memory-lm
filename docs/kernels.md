# The kernels

In plain PyTorch the memory layer was the bottleneck. On my RX 9070 one training step of B-1M took 848 ms against
357 ms for the same model without the table, and reading a prompt was 2.6x slower. So I wrote the hot parts in Triton
(`smlm/kernels.py`). The PyTorch version stays in the code as the reference (`mem_impl="torch"`), and the kernels are
switched on with `mem_impl="triton"`.

## One code base, three very different GPUs

| GPU | Architecture | Stack | Tests |
|---|---|---|---|
| AMD Radeon RX 9070 | RDNA4 (gfx1201), consumer | ROCm 7.2, Triton 3.5 | all pass |
| AMD Instinct MI350X | CDNA4 (gfx950), data centre | ROCm 7.1, Triton 3.7 | 107 / 107 |
| NVIDIA H200 (and H100) | Hopper | CUDA 12.8, Triton 3.6 (3.7.1 on H100) | all pass on the H200; on the H100 the kernels ran in training and in the Qwen add-on checks |

The same code gives the same results on all of them. The plot below is the same training run (same seed, same data)
on all three:

![B-1M on three GPUs](../report/amd_crosscheck.png)

The early curves differ by up to ~1%, which is the normal noise from ties in the top-k and float sums in a different
order. Over a full 500M-token run, kernels and reference end up 0.002% apart, and the H200 and my RX 9070 land on
21.8365 vs 21.8369.

## What the kernels do

| Kernel | Replaces | The trick |
|---|---|---|
| Row-gradient accumulation (`bag_backward_rows`) | materialising `w * grad` for every lookup (~0.8 GB per call) plus `index_put_` | The lookups are sorted by table row, and a segmented scan sums equal rows in registers. Runs that stay inside one program get a plain read-add-write; only the at most two runs at the program borders use atomics. Atomics are ~8x slower than stores on AMD cards, so this took one call from 17 ms to 2.7 ms. The gradient of the lookup weights comes out of the same pass. |
| Product-key selection (`pk_select`, `bag_forward`) | two `topk`s, a 32x32 candidate grid, another `topk`, softmax, gather | Everything happens in one kernel. Each score becomes an order-preserving 16-bit code packed together with its index into one int32, so a single `tl.topk` returns values and indices. Only 130 of the 1,024 candidate pairs can make the top 32 at all, since pair (i, j) needs (i+1)(j+1) ≤ 32. bf16 rounding is done with integer ops, so the selected scores match the reference bit for bit. |
| Lazy Adam (`lazy_adam_step`) | Adam on the whole table, which backs up and restores the rows that weren't read | Works in place and only touches the rows that were read in the step. The reference needed ~4.5 KB of temporary memory per unread row, tens of GB at 16M rows. Without this kernel B-16M would not fit into the 101 GB it trained in. |
| Inference lookup (`bag_infer`) | gather + weighted sum on an fp32 table | Reads fp32, bf16 or 4-bit tables directly, with the swilu product fused in. |
| Decode graphs | ~20 small launches per memory layer per token | For batch-1 generation the memory layer is replayed as a CUDA/HIP graph. Generating text was limited by the CPU launching kernels, not by the GPU. |

## How much it brings

| | RX 9070 | MI350X |
|---|---|---|
| Training, kernels vs PyTorch reference | 1.47x (848 → 577 ms per step) | 1.63x (77k → 126k tok/s) |
| Reading a prompt, vs reference | 1.79x (109 → 61 ms, bf16 table) | 1.88x |
| Generating text (batch 1) | from 21% slower than the plain model to 2% faster (with graphs) | 1.14x vs reference |
| B-1M training speed relative to the same model without table | 0.62x | 0.76x |

**Correctness:**
- Every kernel is tested against the PyTorch reference, forward and gradients, with tables of 262k, 1M and 4M rows.
- Also covered: tables with more than 2³¹ elements, ties in the top-k, the 4-bit format and the decode graphs.
- The tests also run on the CPU with `TRITON_INTERPRET=1`.

## What it isn't

- **No outside benchmark.** "Faster" means faster than my own plain PyTorch version. I haven't benchmarked it against
  other optimised memory-layer implementations, for example Meta's code for *Memory Layers at Scale*.
- **The table still costs time.** Even with the kernels, a model with a table trains at 62% (RX 9070) to 76%
  (MI350X) of the speed of the same model without one.
- **Not tuned per GPU:** a few fixed block sizes, no autotuning. The rest of the model is plain eager PyTorch.
- **Small models only.** Everything was measured on small models. With much larger models other bottlenecks may
  take over.

## Things I learned on AMD

- **Avoid atomics where you can.** fp32 `tl.atomic_add` compiles to the right native instruction, but on RDNA4 it's
  ~8x slower than a plain store, with or without contention
  ([repro](rocm-issues/repro_atomic_add.py)).
- **Small-batch decoding is launch-bound.** Graph replay fixes it; faster kernels don't.
- **For bit-exact tests, round in integer ops.** The CPU interpreter rounds fp32 → bf16 differently from the GPU, so
  where bits must match, the kernels do the rounding (round-to-nearest-even) with integer ops.
- **Byte masks are fine.** An early kernel lost updates when it cleared a byte mask itself. That turned out to be my
  bug, not ROCm's ([repro](rocm-issues/repro_byte_store.py)). The mask is now cleared with one memset after the
  kernel.
