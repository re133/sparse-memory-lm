"""Triton kernels for the product-key memory (run on ROCm and CUDA; on the CPU via TRITON_INTERPRET=1).

The PyTorch code in pkm.py / sparse_values.py stays the reference and fallback (ModelConfig.mem_impl =
"torch"); mem_impl = "triton" switches to the kernels below.

Kernel 1: value-bag backward with row-sparse accumulation (`bag_backward_rows`)
  Reference: per-sample-weight gradients via aten, then contrib = w * grad_out[bag] materialised for every
  lookup (N*K x D, ~0.8 GB per call at training shape) and index_put_(accumulate=True) into the accumulator.
  Kernel: the lookups are sorted by table row once (torch.sort on N*K int32 keys). Each program takes P
  consecutive positions of the sorted list, loads w * grad_out[n] as a (P x BLOCK_D) tile, sums the runs of
  equal rows with a segmented scan and adds each run once: runs that lie entirely inside the program with a
  plain read-add-write, the (at most two) runs at the program borders with an atomic add. Atomics for every
  run were 6x slower on the RX 9070 (17 ms vs 2.7 ms per call at training shape). The per-sample-weight gradient dot(grad_out[n], table[row]) is
  computed in the same pass (consecutive equal rows hit the cache). Rows are marked in the `touched` mask.
  Summation order differs from the reference (floating-point rounding only).
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                     # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:
    @triton.jit
    def _seg_add(f1, v1, f2, v2):
        # segmented sum: a segment start (f2) restarts the running sum
        return f1 | f2, tl.where(f2, v2, v1 + v2)

    @triton.jit
    def _bag_bwd_rows_kernel(rows_ptr, src_ptr, w_ptr, g_ptr, table_ptr, acc_ptr, touched_ptr, gw_ptr,
                             n_pos, K: tl.constexpr, D: tl.constexpr, P: tl.constexpr, BLOCK_D: tl.constexpr):
        # one program = P consecutive positions of the row-sorted lookup list, all of D in BLOCK_D slices
        pid = tl.program_id(0)
        pos = pid * P + tl.arange(0, P)
        pm = pos < n_pos
        rows = tl.load(rows_ptr + pos, mask=pm, other=-1)
        prev = tl.load(rows_ptr + pos - 1, mask=pm & (pos > 0), other=-1)
        nxt = tl.load(rows_ptr + pos + 1, mask=(pos + 1) < n_pos, other=-1)
        lane = tl.arange(0, P)
        is_start = (rows != prev) | (lane == 0)
        is_end = ((rows != nxt) | (lane == P - 1)) & pm
        # a run of equal rows can continue in the neighbouring program only if it is this program's first or
        # last row; those runs are added atomically, all others belong to this program alone (plain add)
        first_row = tl.load(rows_ptr + pid * P)
        last_row = tl.load(rows_ptr + tl.minimum(pid * P + P - 1, n_pos - 1))
        shared = (rows == first_row) | (rows == last_row)
        src = tl.load(src_ptr + pos, mask=pm, other=0)
        n = (src // K).to(tl.int64)
        w = tl.load(w_ptr + src, mask=pm, other=0.0).to(tl.float32)
        r64 = tl.where(pm, rows, 0).to(tl.int64)
        gw = tl.zeros([P], dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            m2 = pm[:, None] & (d < D)[None, :]
            g = tl.load(g_ptr + n[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            t = tl.load(table_ptr + r64[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
            gw += tl.sum(g * t, axis=1)
            flags = tl.broadcast_to(is_start[:, None], (P, BLOCK_D))
            _, seg = tl.associative_scan((flags, w[:, None] * g), 0, _seg_add)
            ptrs = acc_ptr + r64[:, None] * D + d[None, :]
            own = (is_end & ~shared)[:, None] & m2
            cur = tl.load(ptrs, mask=own, other=0.0)
            tl.store(ptrs, cur + seg, mask=own)
            tl.atomic_add(ptrs, seg, mask=(is_end & shared)[:, None] & m2, sem="relaxed")
        tl.store(gw_ptr + src, gw, mask=pm)
        tl.store(touched_ptr + r64, tl.full([P], 1, tl.uint8), mask=pm)


def bag_backward_rows(grad_out, indices, weights, table, acc, touched, P=32, block_d=64, num_warps=2):
    """grad_out (N, D), indices / weights (N, K) -> per-sample-weight gradient (N, K) (fp32);
    acc[indices[n, k]] += weights[n, k] * grad_out[n] and touched[indices] = True, in place."""
    N, K = indices.shape
    D = table.shape[1]
    flat = indices.reshape(-1)
    rows, src = torch.sort(flat.to(torch.int32))
    g = grad_out.contiguous()
    w = weights.reshape(-1).contiguous()
    gw = torch.empty(N * K, dtype=torch.float32, device=grad_out.device)
    n_pos = flat.numel()
    grid = (triton.cdiv(n_pos, P),)
    _bag_bwd_rows_kernel[grid](rows, src, w, g, table, acc, touched.view(torch.uint8), gw, n_pos, K, D,
                               P=P, BLOCK_D=block_d, num_warps=num_warps)
    return gw.view(N, K)
