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


# ---------------------------------------------------------------------------------------------------------
# Kernel 2: product-key selection (both half top-k, cartesian top-k, softmax) and the weighted value bag
#
# The sub-key scores s1, s2 (N, H, n_keys) still come from the same einsum as in the reference, so they are
# bit-identical. The kernel selects on them like the reference: top-knn of each half, the pairwise sums
# rounded to the scores' dtype (bf16 under autocast, as `s1 + s2` in PyTorch), top-knn of those, softmax in
# fp32. Selection sorts integer keys: the score's order-preserving 16-bit code in the high bits, the
# candidate (key index or final table index, inverted so that ties prefer the smaller index) in the low bits,
# so one tl.topk returns values and indices. Of the knn x knn pairs only the 130 (knn = 32) with
# (i+1)(j+1) <= knn can be in the top-knn (all others are dominated). Selected scores are exactly the
# reference's; indices can differ only between entries with exactly equal scores (tie order).
# ---------------------------------------------------------------------------------------------------------
_LOW = 0xFFFFFFFF

if HAVE_TRITON:
    @triton.jit
    def _bf16_bits_rtne(x):
        """fp32 tensor -> bit pattern (int32, 0..0xFFFF) of x rounded to bf16, round-to-nearest-even as in
        PyTorch. Done with integer ops so that the GPU and the CPU interpreter round identically."""
        u = x.to(tl.int32, bitcast=True)
        u = u + 0x7FFF + ((u >> 16) & 1)
        return (u >> 16) & 0xFFFF

    @triton.jit
    def _order16(b):
        """bf16 bit pattern (int32, 0..0xFFFF) -> int64 key part: order-preserving signed 16-bit code in the high
        half."""
        o = tl.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)
        return (o - 0x8000).to(tl.int64) << 32

    @triton.jit
    def _decode16(key):
        """inverse of _order16 for the high half of an int64 key -> fp32 value of the 16-bit float."""
        o = (key >> 32).to(tl.int32) + 0x8000
        b = tl.where(o >= 0x8000, o ^ 0x8000, (~o) & 0xFFFF)
        return (b << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _half_topk(x, NK: tl.constexpr, KNN: tl.constexpr, CH: tl.constexpr):
        """top-KNN of a (NK,) 16-bit-float row -> (values fp32, indices int32), descending. int32 keys:
        order-preserving 16-bit code << 16 | (0xFFFF - index); two stages (top-KNN of each CH-chunk, then of
        the survivors), which selects exactly the same set as one top-KNN because all keys are distinct."""
        k = tl.arange(0, NK)
        b = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        o = tl.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)
        key = ((o - 0x8000) << 16) | (0xFFFF - k)
        if NK > CH:
            part = tl.topk(tl.reshape(key, (NK // CH, CH)), KNN, dim=1)
            top = tl.topk(tl.reshape(part, (NK // CH * KNN,)), KNN, dim=0)
        else:
            top = tl.topk(key, KNN, dim=0)
        o2 = (top >> 16) + 0x8000
        bb = tl.where(o2 >= 0x8000, o2 ^ 0x8000, (~o2) & 0xFFFF)
        return (bb << 16).to(tl.float32, bitcast=True), 0xFFFF - (top & 0xFFFF)

    @triton.jit
    def _pk_select_kernel(s1_ptr, s2_ptr, pi_ptr, pj_ptr, scores_ptr, idx_ptr, w_ptr,
                          NK: tl.constexpr, KNN: tl.constexpr, NP: tl.constexpr, CH: tl.constexpr,
                          IS_BF16: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, NK)
        v1, i1 = _half_topk(tl.load(s1_ptr + row * NK + k), NK, KNN, CH)
        v2, i2 = _half_topk(tl.load(s2_ptr + row * NK + k), NK, KNN, CH)
        # candidates (i, j) with (i+1)(j+1) <= KNN: every other pair is dominated by >= KNN pairs that are at
        # least as large (both lists sorted descending), so it can at most tie with the KNN-th score
        p = tl.arange(0, NP)
        pi = tl.load(pi_ptr + p)
        pj = tl.load(pj_ptr + p)
        valid = pi >= 0
        pi = tl.where(valid, pi, 0)
        pj = tl.where(valid, pj, 0)
        cand = _bf16_bits_rtne(tl.gather(v1, pi, 0) + tl.gather(v2, pj, 0))   # as `s1 + s2` in bf16
        full = tl.gather(i1, pi, 0).to(tl.int64) * NK + tl.gather(i2, pj, 0).to(tl.int64)
        LOW = tl.full([1], 0xFFFFFFFF, tl.int64)
        ck = tl.where(valid, _order16(cand) | (LOW - full), -9223372036854775807)
        top = tl.topk(ck, KNN, dim=0)
        sc = _decode16(top)
        idx = LOW - (top & LOW)
        e = tl.exp(sc - tl.max(sc, axis=0))
        w = e / tl.sum(e, axis=0)
        j = tl.arange(0, KNN)
        if IS_BF16:
            tl.store(scores_ptr + row * KNN + j, sc.to(tl.bfloat16))
        else:
            tl.store(scores_ptr + row * KNN + j, sc.to(tl.float16))
        tl.store(idx_ptr + row * KNN + j, idx)
        tl.store(w_ptr + row * KNN + j, w)

    @triton.jit
    def _bag_fwd_kernel(idx_ptr, w_ptr, table_ptr, out_ptr, K: tl.constexpr, D: tl.constexpr,
                        BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = d < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j0 in range(0, K, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            r = tl.load(idx_ptr + n * K + j).to(tl.int64)
            w = tl.load(w_ptr + n * K + j).to(tl.float32)
            v = tl.load(table_ptr + r[:, None] * D + d[None, :], mask=dm[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w[:, None] * v, axis=0)
        tl.store(out_ptr + n * D + d, acc, mask=dm)


_PAIRS = {}


def _pairs(knn, device):
    """(i, j) with (i+1)(j+1) <= knn, padded with -1 to a power of two (130 pairs -> 256 for knn = 32)."""
    key = (knn, str(device))
    if key not in _PAIRS:
        pairs = [(i, j) for i in range(knn) for j in range(knn) if (i + 1) * (j + 1) <= knn]
        n = triton.next_power_of_2(len(pairs))
        pi = torch.full((n,), -1, dtype=torch.int32)
        pj = torch.full((n,), -1, dtype=torch.int32)
        pi[:len(pairs)] = torch.tensor([a for a, _ in pairs], dtype=torch.int32)
        pj[:len(pairs)] = torch.tensor([b for _, b in pairs], dtype=torch.int32)
        _PAIRS[key] = (pi.to(device), pj.to(device), n)
    return _PAIRS[key]


def pk_select(s1, s2, knn, ch=128, num_warps=4):
    """s1, s2 (N, H, n_keys) bf16/fp16 sub-key scores -> scores (N, H, knn) (same dtype), indices (N, H, knn)
    int64 into the n_keys^2 table, softmax weights (N, H, knn) fp32."""
    assert s1.dtype == torch.bfloat16 and s1.shape == s2.shape     # 16-bit codes decode as bf16
    N, H, NK = s1.shape
    s1, s2 = s1.contiguous(), s2.contiguous()
    scores = torch.empty(N, H, knn, dtype=s1.dtype, device=s1.device)
    idx = torch.empty(N, H, knn, dtype=torch.int64, device=s1.device)
    w = torch.empty(N, H, knn, dtype=torch.float32, device=s1.device)
    pi, pj, n_pairs = _pairs(knn, s1.device)
    _pk_select_kernel[(N * H,)](s1, s2, pi, pj, scores, idx, w, NK=NK, KNN=knn, NP=n_pairs, CH=min(NK, ch),
                                IS_BF16=s1.dtype == torch.bfloat16, num_warps=num_warps)
    return scores, idx, w


def bag_forward(indices, weights, table, block_j=32, block_d=128):
    """out[n] = sum_k weights[n, k] * table[indices[n, k]] (fp32); table fp32 / bf16 / fp16."""
    N, K = indices.shape
    D = table.shape[1]
    out = torch.empty(N, D, dtype=torch.float32, device=table.device)
    bj = min(block_j, K)
    assert K % bj == 0
    _bag_fwd_kernel[(N, triton.cdiv(D, block_d))](indices.contiguous(), weights.contiguous(), table, out,
                                                  K=K, D=D, BLOCK_J=bj, BLOCK_D=block_d, num_warps=4)
    return out


class PKSelect(torch.autograd.Function):
    """Differentiable wrapper of pk_select. Backward = softmax backward, then the gradient of every selected
    score goes to its two sub-key scores (sum over the cartesian partners, accumulated in fp32 and rounded once,
    as the reference's autograd does), returned as dense (N, H, n_keys) gradients for the einsum's backward."""

    @staticmethod
    def forward(ctx, s1, s2, knn):
        scores, idx, w = pk_select(s1, s2, knn)
        ctx.save_for_backward(idx, w)
        ctx.nk, ctx.dtype = s1.shape[-1], s1.dtype
        ctx.mark_non_differentiable(idx)
        return scores, idx, w

    @staticmethod
    def backward(ctx, g_scores, g_idx, g_w):
        idx, w = ctx.saved_tensors
        dl = torch.zeros_like(w)
        if g_w is not None:
            dl = w * (g_w - (w * g_w).sum(-1, keepdim=True))
        if g_scores is not None:
            dl = dl + g_scores.float()
        ds = dl.to(ctx.dtype).float()                   # the reference's grad of scores.float() is bf16
        N, H, _ = idx.shape
        z = torch.zeros(N, H, ctx.nk, dtype=torch.float32, device=idx.device)
        ds1 = z.scatter_add(-1, idx // ctx.nk, ds).to(ctx.dtype)
        ds2 = z.scatter_add_(-1, idx % ctx.nk, ds).to(ctx.dtype)
        return ds1, ds2, None


# ---------------------------------------------------------------------------------------------------------
# Kernel 4: inference lookups on a bf16 or 4-bit copy of the value table, with the swilu product fused in
#
# 4 bit: the "int4" scheme of scripts/quantize_table_eval.py (measured +0.12 % val PPL for B-1M): per row
# d = (value with the largest |w|) / -8 as fp16, codes c = clamp(round(w / d), -8, 7), w ~ c * d. Packed two
# codes per byte: byte i of a row holds column i (low nibble) and column i + D/2 (high nibble), stored as
# c + 8. The bag dequantises c * d in fp32, i.e. exactly the table that quantize_table_eval evaluated.
# Optional epilogue (inference): out = bag * bf16(silu(pre)), as `out * F.silu(swilu_proj(x)).to(out.dtype)`
# with `pre` = swilu_proj(x) in bf16; optionally the result is rounded to bf16 (the cast autocast would do
# before value_proj).
# ---------------------------------------------------------------------------------------------------------
def quantize_q4(table, chunk=1 << 18):
    """table (rows, D) -> packed (rows, D/2) uint8, scales (rows,) fp16 (see above)."""
    rows, D = table.shape
    assert D % 2 == 0
    packed = torch.empty(rows, D // 2, dtype=torch.uint8, device=table.device)
    scales = torch.empty(rows, dtype=torch.float16, device=table.device)
    for a in range(0, rows, chunk):
        w = table[a:a + chunk].float()
        m = w.gather(1, w.abs().argmax(dim=1, keepdim=True))
        d = (m / -8).half()
        d = torch.where(d == 0, torch.ones_like(d), d)
        c = (torch.clamp(torch.round(w / d.float()), -8, 7) + 8).to(torch.uint8)
        packed[a:a + chunk] = c[:, :D // 2] | (c[:, D // 2:] << 4)
        scales[a:a + chunk] = d.squeeze(1)
    return packed, scales


def dequantize_q4(packed, scales):
    lo = (packed & 15).float() - 8
    hi = (packed >> 4).float() - 8
    return torch.cat([lo, hi], dim=1) * scales.float()[:, None]


if HAVE_TRITON:
    @triton.jit
    def _silu_bf16(p):
        s = p / (1.0 + tl.exp(-p))
        return (_bf16_bits_rtne(s) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _round_bf16(x):
        return (_bf16_bits_rtne(x) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _bag_infer_kernel(idx_ptr, w_ptr, table_ptr, scale_ptr, pre_ptr, out_ptr, K: tl.constexpr,
                          D: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_D: tl.constexpr, Q4: tl.constexpr,
                          SWILU: tl.constexpr, OUT_BF16: tl.constexpr):
        n = tl.program_id(0).to(tl.int64)
        if Q4:
            DH: tl.constexpr = D // 2
            d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)          # byte column (< D/2)
            dm = d < DH
            acc_lo = tl.zeros([BLOCK_D], dtype=tl.float32)
            acc_hi = tl.zeros([BLOCK_D], dtype=tl.float32)
            for j0 in range(0, K, BLOCK_J):
                j = j0 + tl.arange(0, BLOCK_J)
                r = tl.load(idx_ptr + n * K + j).to(tl.int64)
                ws = tl.load(w_ptr + n * K + j).to(tl.float32) * tl.load(scale_ptr + r).to(tl.float32)
                byte = tl.load(table_ptr + r[:, None] * DH + d[None, :], mask=dm[None, :], other=0).to(tl.int32)
                lo = ((byte & 15) - 8).to(tl.float32)
                hi = ((byte >> 4) - 8).to(tl.float32)
                acc_lo += tl.sum(ws[:, None] * lo, axis=0)
                acc_hi += tl.sum(ws[:, None] * hi, axis=0)
            cols = (d, d + DH)
            accs = (acc_lo, acc_hi)
        else:
            d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
            dm = d < D
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)
            for j0 in range(0, K, BLOCK_J):
                j = j0 + tl.arange(0, BLOCK_J)
                r = tl.load(idx_ptr + n * K + j).to(tl.int64)
                w = tl.load(w_ptr + n * K + j).to(tl.float32)
                v = tl.load(table_ptr + r[:, None] * D + d[None, :], mask=dm[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(w[:, None] * v, axis=0)
            cols = (d,)
            accs = (acc,)
        for i in tl.static_range(len(cols)):
            c = cols[i]
            a = accs[i]
            if SWILU:
                p = tl.load(pre_ptr + n * D + c, mask=dm, other=0.0).to(tl.float32)
                a = a * _silu_bf16(p)
            if OUT_BF16:
                tl.store(out_ptr + n * D + c, _round_bf16(a).to(tl.bfloat16), mask=dm)
            else:
                tl.store(out_ptr + n * D + c, a, mask=dm)


def bag_infer(indices, weights, table, scales=None, pre=None, out_bf16=False, block_j=64, block_d=None):
    """Inference value bag. table: fp32 / bf16 (scales None) or q4-packed uint8 (rows, D/2) with fp16 scales.
    pre: optional (N, D) swilu pre-activation (bf16) -> out = bag * bf16(silu(pre)). Returns (N, D) fp32 or bf16."""
    N, K = indices.shape
    q4 = scales is not None
    D = table.shape[1] * (2 if q4 else 1)
    out = torch.empty(N, D, dtype=torch.bfloat16 if out_bf16 else torch.float32, device=table.device)
    bj = min(block_j, K)
    block_d = block_d or (64 if q4 else 128)
    width = D // 2 if q4 else D
    grid = (N, triton.cdiv(width, block_d))
    _bag_infer_kernel[grid](indices.contiguous(), weights.contiguous(), table, scales if q4 else table,
                            pre.contiguous() if pre is not None else out, out, K=K, D=D, BLOCK_J=bj,
                            BLOCK_D=block_d, Q4=q4, SWILU=pre is not None, OUT_BF16=out_bf16, num_warps=4)
    return out


# ---------------------------------------------------------------------------------------------------------
# Kernel 3: lazy Adam on the value table, fused and in place
#
# Same update as sparse_values.LazyRowAdam (torch Adam formula, global step for the bias correction, no
# weight decay) for the rows marked in `touched`; afterwards their accumulator rows are zero and the mask is
# cleared (the mask with one memset after the kernel). Rows that were not read are not loaded at all, and no
# copies are made: the reference
# implementation saves and restores the unread rows (values + both moments), which costs 3 x 1.5 KB per
# unread row of temporary memory - tens of GB for 16M rows.
# ---------------------------------------------------------------------------------------------------------
if HAVE_TRITON:
    @triton.jit
    def _lazy_adam_kernel(p_ptr, m_ptr, v_ptr, acc_ptr, touched_ptr, scale_ptr, n_rows, lr, beta1, beta2, eps,
                          step_size, bc2_sqrt, D: tl.constexpr, BR: tl.constexpr, BLOCK_D: tl.constexpr):
        rows = tl.program_id(0) * BR + tl.arange(0, BR)
        rm = rows < n_rows
        t = tl.load(touched_ptr + rows, mask=rm, other=0) != 0
        scale = tl.load(scale_ptr).to(tl.float32)
        r64 = rows.to(tl.int64)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            msk = t[:, None] & (d < D)[None, :]
            off = r64[:, None] * D + d[None, :]
            g = tl.load(acc_ptr + off, mask=msk, other=0.0) * scale
            m = tl.load(m_ptr + off, mask=msk, other=0.0)
            v = tl.load(v_ptr + off, mask=msk, other=0.0)
            p = tl.load(p_ptr + off, mask=msk, other=0.0)
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g * g
            p = p - step_size * m / (tl.sqrt(v) / bc2_sqrt + eps)
            tl.store(m_ptr + off, m, mask=msk)
            tl.store(v_ptr + off, v, mask=msk)
            tl.store(p_ptr + off, p, mask=msk)
            tl.store(acc_ptr + off, tl.zeros_like(g), mask=msk)


def lazy_adam_step(param, exp_avg, exp_avg_sq, acc, touched, grad_scale, step, lr, beta1, beta2, eps,
                   rows_per_program=8, block_d=128):
    """In-place lazy Adam step (see above). grad_scale: 0-d tensor (clip factor) or float; step: the step
    number after this update (1 for the first)."""
    rows, D = param.shape
    if not torch.is_tensor(grad_scale):
        grad_scale = torch.tensor(float(grad_scale), device=param.device)
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    grid = (triton.cdiv(rows, rows_per_program),)
    _lazy_adam_kernel[grid](param, exp_avg, exp_avg_sq, acc, touched.view(torch.uint8),
                            grad_scale.reshape(1).float(), rows, lr, beta1, beta2, eps, lr / bc1, bc2 ** 0.5,
                            D=D, BR=rows_per_program, BLOCK_D=block_d, num_warps=4)
    # cleared here, not in the kernel: a masked byte store of the mask from inside the kernel cleared rows
    # of neighbouring programs before they had read them (seen on the RX 9070 with 8+ rows per program)
    touched.zero_()
