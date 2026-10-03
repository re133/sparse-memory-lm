"""Triton kernels against the PyTorch reference (smlm/kernels.py).

GPU (ROCm or CUDA):  pytest tests/test_kernels.py
CPU (interpreter):   TRITON_INTERPRET=1 pytest tests/test_kernels.py      (small sizes only)

Tolerances (fixed): fp32 outputs / gradients rtol=1e-5, atol=1e-5 relative to the reference's scale; the
kernels only change the summation order.
"""
import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if INTERPRET:
    DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    pytest.skip("Triton kernels need a GPU or TRITON_INTERPRET=1", allow_module_level=True)
pytest.importorskip("triton")

from smlm.kernels import PKSelect, bag_forward, pk_select  # noqa: E402
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.sparse_values import RowStore, _RowSparseBag  # noqa: E402

GPU_MEM_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE == "cuda" else 0
# (rows, D, tokens N): table sizes 262k / 1M / 4M on a GPU; small shapes in the CPU interpreter
if INTERPRET:
    SIZES = [(512, 32, 16), (4096, 48, 24)]
else:
    # 4M rows x 384 needs ~20 GB in this test; on smaller GPUs the 4M table gets 64 columns (same row count,
    # same int64 offsets up to 4M x 64, ~3 GB)
    SIZES = [(512 ** 2, 384, 4096), (1024 ** 2, 384, 4096), (2048 ** 2, 384 if GPU_MEM_GIB > 40 else 64, 4096)]
TOL = dict(rtol=1e-5, atol=1e-5)


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    if DEVICE == "cuda":
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def close(a, b, msg="", chunk=1 << 18):
    """assert_close relative to the reference's scale, in row chunks (no full-size temporaries)."""
    scale = max(1.0, float(b.abs().max()))
    for s in range(0, a.shape[0], chunk):
        torch.testing.assert_close(a[s:s + chunk] / scale, b[s:s + chunk] / scale, **TOL, msg=msg)


def skewed_indices(rows, N, K, gen):
    """Mix of uniform rows and a few hot rows (runs of equal rows that cross kernel program borders)."""
    idx = torch.randint(0, rows, (N, K), generator=gen)
    hot = torch.randint(0, rows, (8,), generator=gen)
    mask = torch.rand(N, K, generator=gen) < 0.2
    idx[mask] = hot[torch.randint(0, 8, (int(mask.sum()),), generator=gen)]
    return idx


@pytest.mark.parametrize("rows,D,N", SIZES)
def test_bag_backward_rows_matches_reference(rows, D, N):
    gen = torch.Generator().manual_seed(rows)
    K = 128
    table = (torch.randn(rows, D, generator=gen) * D ** -0.5).to(DEVICE)
    idx = skewed_indices(rows, N, K, gen).to(DEVICE)
    w = torch.softmax(torch.randn(N, K, generator=gen), -1).to(DEVICE)
    g = torch.randn(N, D, generator=gen).to(DEVICE)
    out = {}
    for impl in ("torch", "triton"):
        st = RowStore(table)
        st.acc.add_(1.0)                                    # the accumulator already holds earlier gradients
        wr = w.clone().requires_grad_()
        y = _RowSparseBag.apply(wr, idx, table, st, impl)
        y.backward(g)
        out[impl] = (y.detach(), wr.grad, st.acc, st.touched)
        del st, y
    (y0, gw0, acc0, t0), (y1, gw1, acc1, t1) = out.pop("torch"), out.pop("triton")
    close(y1, y0, "bag output (kernel 2 forward)")
    close(gw1, gw0, "per-sample-weight gradient")
    close(acc1, acc0, "row accumulator")
    assert torch.equal(t1, t0)
    del table, acc0, acc1
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def tiny(impl, seed=0):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[0, 2, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16,
                      mem_share_values=True, mem_value_grad="row_sparse", mem_impl=impl)
    torch.manual_seed(seed)
    return Transformer(cfg).to(DEVICE)


def test_model_gradients_triton_equal_torch():
    """Whole model (3 memory layers sharing one table, 2 micro-batches): every gradient and the row
    accumulator agree between mem_impl='torch' and 'triton'."""
    ref, ker = tiny("torch"), tiny("triton")
    ker.load_state_dict(ref.state_dict())
    gen = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 128, (2, 17), generator=gen).to(DEVICE) for _ in range(2)]
    for m in (ref, ker):
        for b in batches:
            _, loss = m(b[:, :-1], b[:, 1:])
            (loss / 2).backward()
    for (n, p0), p1 in zip(ref.named_parameters(), ker.parameters()):
        if p0.grad is None:
            continue
        close(p1.grad, p0.grad, n)
    t0, t1 = ref.memory_layers()[0].values.weight, ker.memory_layers()[0].values.weight
    close(t1.row_store.acc, t0.row_store.acc, "table accumulator")
    assert torch.equal(t1.row_store.touched, t0.row_store.touched)


# ----------------------------------------------------------------------------------------- kernel 2
# n_keys per half: 512 (262k), 1024 (1M), 2048 (4M), 4096 (16M, the cloud's B-16M)
SELECT_SHAPES = ([(64, 2, 16, 4), (96, 4, 64, 8), (4, 2, 4096, 32)] if INTERPRET else
                 [(4096, 4, 512, 32), (4096, 4, 1024, 32), (4096, 4, 2048, 32), (4096, 4, 4096, 32),
                  (1, 4, 1024, 32)])


def reference_select(s1, s2, knn):
    N, H, NK = s1.shape
    a1, i1 = s1.topk(knn, -1)
    a2, i2 = s2.topk(knn, -1)
    alls = (a1.unsqueeze(-1) + a2.unsqueeze(-2)).view(N, H, -1)
    alli = (i1.unsqueeze(-1) * NK + i2.unsqueeze(-2)).view(N, H, -1)
    sc, best = alls.topk(knn, -1)
    return sc, alli.gather(-1, best), torch.softmax(sc.float(), -1)


@pytest.mark.parametrize("N,H,NK,knn", SELECT_SHAPES)
def test_pk_select_matches_reference(N, H, NK, knn, dtype=torch.bfloat16):
    """Scores exactly equal; every selected index really has its reported score (so the selection is an exact
    top-k and can differ from the reference only between equal scores); weights within 1e-6."""
    gen = torch.Generator().manual_seed(N + NK)
    s1 = torch.randn(N, H, NK, generator=gen).to(DEVICE, dtype)
    s2 = torch.randn(N, H, NK, generator=gen).to(DEVICE, dtype)
    rs, ri, rw = reference_select(s1, s2, knn)
    sc, idx, w = pk_select(s1, s2, knn)
    assert torch.equal(sc, rs)
    own = (s1.gather(-1, idx // NK).float() + s2.gather(-1, idx % NK).float()).to(dtype)
    assert torch.equal(own, sc)
    srt = idx.view(-1, knn).sort(-1).values
    assert bool((srt[:, 1:] != srt[:, :-1]).all())                       # no index twice in a row
    torch.testing.assert_close(w, rw, rtol=0, atol=1e-6)


@pytest.mark.parametrize("N,H,NK,knn", SELECT_SHAPES[:2])
def test_pk_select_backward(N, H, NK, knn):
    """PKSelect's backward: softmax backward, then every selected score's gradient summed into its two sub-key
    scores. The reference's autograd sums these in fp32 and rounds once to bf16 (sum over the cartesian
    partners), so the kernel must equal the exactly summed (fp64) gradient rounded to bf16, within 1 bf16 ulp."""
    gen = torch.Generator().manual_seed(7)
    s1 = torch.randn(N, H, NK, generator=gen).to(DEVICE, torch.bfloat16).requires_grad_()
    s2 = torch.randn(N, H, NK, generator=gen).to(DEVICE, torch.bfloat16).requires_grad_()
    g = torch.randn(N, H, knn, generator=gen).to(DEVICE)
    sc, idx, w = PKSelect.apply(s1, s2, knn)
    (w * g).sum().backward()
    d1, d2 = s1.grad.clone(), s2.grad.clone()
    w64, g64 = w.double(), g.double()
    ds = (w64 * (g64 - (w64 * g64).sum(-1, keepdim=True))).float().to(torch.bfloat16).double()
    z = torch.zeros(N, H, NK, dtype=torch.float64, device=DEVICE)
    e1 = z.scatter_add(-1, idx // NK, ds).to(torch.bfloat16).float()
    e2 = z.scatter_add(-1, idx % NK, ds).to(torch.bfloat16).float()
    # gradient flows only into selected sub-keys
    assert torch.equal(d1 != 0, e1 != 0) or bool(((d1 != 0) & (e1 == 0)).sum() == 0)
    # 1 bf16 ulp, absolute floor at the scale of the largest term (cancellation in fp32 vs fp64)
    torch.testing.assert_close(d1.float(), e1, rtol=2 ** -7, atol=2 ** -8 * float(ds.abs().max()))
    torch.testing.assert_close(d2.float(), e2, rtol=2 ** -7, atol=2 ** -8 * float(ds.abs().max()))


@pytest.mark.parametrize("rows,D,N", SIZES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bag_forward_matches_embedding_bag(rows, D, N, dtype):
    gen = torch.Generator().manual_seed(3)
    table = (torch.randn(rows, D, generator=gen) * D ** -0.5).to(DEVICE, dtype)
    idx = skewed_indices(rows, N, 128, gen).to(DEVICE)
    w = torch.softmax(torch.randn(N, 128, generator=gen), -1).to(DEVICE)
    ref = torch.nn.functional.embedding_bag(idx, table.float(), per_sample_weights=w, mode="sum")
    close(bag_forward(idx, w, table), ref, "bag forward")


# ----------------------------------------------------------------------- kernel 4 / decode graphs
@pytest.mark.parametrize("rows,D,N", SIZES)
@pytest.mark.parametrize("kind", ["fp32", "bf16", "q4"])
def test_bag_infer_matches_reference(rows, D, N, kind):
    """Inference bag on fp32 / bf16 / 4-bit tables with the swilu product fused, against embedding_bag on the
    same (dequantised) table times bf16(silu(pre)); the bf16 output against the reference's bf16 cast."""
    from smlm.kernels import bag_infer, dequantize_q4, quantize_q4
    gen = torch.Generator().manual_seed(5)
    table = (torch.randn(rows, D, generator=gen) * D ** -0.5).to(DEVICE)
    idx = skewed_indices(rows, N, 128, gen).to(DEVICE)
    w = torch.softmax(torch.randn(N, 128, generator=gen), -1).to(DEVICE)
    pre = torch.randn(N, D, generator=gen).to(DEVICE, torch.bfloat16)
    if kind == "fp32":
        t, sc, ref_t = table, None, table
    elif kind == "bf16":
        t, sc = table.to(torch.bfloat16), None
        ref_t = t.float()
    else:
        t, sc = quantize_q4(table)
        ref_t = dequantize_q4(t, sc)
    ref = torch.nn.functional.embedding_bag(idx, ref_t, per_sample_weights=w, mode="sum")
    close(bag_infer(idx, w, t, sc), ref, "bag")
    ref2 = ref * torch.nn.functional.silu(pre).float()
    close(bag_infer(idx, w, t, sc, pre=pre), ref2, "bag * silu")
    out = bag_infer(idx, w, t, sc, pre=pre, out_bf16=True)
    assert out.dtype == torch.bfloat16
    scale = float(ref2.abs().max())                        # bf16 output: within 1 bf16 ulp of the reference
    torch.testing.assert_close(out.float(), ref2.to(torch.bfloat16).float(), rtol=2 ** -7, atol=2 ** -8 * scale)


def test_q4_equals_quantisation_study():
    """The packed 4-bit table dequantises exactly to the int4 scheme of scripts/quantize_table_eval.py."""
    import importlib.util
    from smlm.kernels import dequantize_q4, quantize_q4
    spec = importlib.util.spec_from_file_location(
        "qte", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "quantize_table_eval.py"))
    qte = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qte)
    table = torch.randn(1000, 64, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    assert torch.equal(dequantize_q4(*quantize_q4(table)), qte.quantize(table, "int4"))


@pytest.mark.skipif(DEVICE != "cuda", reason="graphs need a GPU")
@pytest.mark.parametrize("kind", ["fp32", "bf16", "q4"])
def test_decode_graph_bit_identical(kind):
    """Decoding with captured memory-layer graphs gives bit-identical logits to the eager kernel path."""
    ker = tiny("triton").eval()
    ker.set_memory_inference_table(kind)
    x = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(2)).to(DEVICE)

    def decode():
        caches = [dict() for _ in range(ker.cfg.n_layers)]
        outs = [ker(x[:, :4], kv_caches=caches, pos0=0).float()]
        for i in range(4, 8):
            outs.append(ker(x[:, i:i + 1], kv_caches=caches, pos0=i).float())
        return torch.cat(outs, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = decode()
        ker.set_memory_decode_graphs(True)
        graph = decode()
        graph2 = decode()                                   # replay of the captured graphs
        ker.set_memory_decode_graphs(False)
    assert torch.equal(graph, eager) and torch.equal(graph2, eager)


# ----------------------------------------------------------------------------------------- kernel 3
@pytest.mark.parametrize("rows,D,N", SIZES)
def test_lazy_adam_kernel_matches_reference(rows, D, N):
    if DEVICE == "cuda" and GPU_MEM_GIB < 40:
        # two tables + accumulators + Adam states + the reference's copies of the unread rows: 4M x 384 would
        # need > 40 GB, 1M x 384 ~13 GB; on small GPUs the rows stay, the columns shrink
        D = 16
    """Three steps of the fused lazy Adam against LazyRowAdam (reference): read rows updated identically up to
    rounding, unread rows (values and both moments) bit-identical, accumulator and mask cleared."""
    from smlm.sparse_values import LazyRowAdam, row_store
    gen = torch.Generator().manual_seed(11)
    base = (torch.randn(rows, D, generator=gen) * D ** -0.5).to(DEVICE)
    tabs = {impl: torch.nn.Parameter(base.clone()) for impl in ("torch", "triton")}
    opts = {impl: LazyRowAdam([tabs[impl]], lr=2.4e-3, betas=(0.9, 0.95), eps=1e-8, impl=impl)
            for impl in ("torch", "triton")}
    for step in range(3):
        idx = skewed_indices(rows, min(N, 1024), 128, gen).to(DEVICE)
        grad_rows = torch.randn(idx.numel(), D, generator=gen).to(DEVICE)
        scale = 0.5 if step == 1 else 1.0
        before = {impl: (tabs[impl].detach().clone()) for impl in tabs}
        for impl in ("torch", "triton"):
            st = row_store(tabs[impl])
            st.acc.index_put_((idx.reshape(-1),), grad_rows, accumulate=True)
            st.touched[idx.reshape(-1)] = True
            st.grad_scale = torch.tensor(scale, device=DEVICE) if impl == "triton" else scale
            opts[impl].step()
            assert not st.touched.any() and torch.count_nonzero(st.acc) == 0
        t0, t1 = tabs["torch"].detach(), tabs["triton"].detach()
        read = torch.zeros(rows, dtype=torch.bool, device=DEVICE)
        read[idx.reshape(-1)] = True
        assert torch.equal(t1[~read], before["triton"][~read])                 # unread rows untouched
        close(t1, t0, "values")
        s0, s1 = opts["torch"].state[tabs["torch"]], opts["triton"].state[tabs["triton"]]
        close(s1["exp_avg"], s0["exp_avg"], "exp_avg")
        close(s1["exp_avg_sq"], s0["exp_avg_sq"], "exp_avg_sq")
        assert int(s1["step"]) == step + 1
    del tabs, opts
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------- offsets beyond 2^31 elements (B-16M: 16.8M x 384 = 6.4 G)
@pytest.mark.skipif(DEVICE != "cuda" or GPU_MEM_GIB < 60, reason="needs > 60 GB GPU memory (runs on the H200)")
def test_large_table_offsets():
    """Table with 6M x 384 = 2.3 G elements (> 2^31) and lookups of rows beyond 5.6M (row * 384 > 2^31):
    bag forward, kernel-1 backward and the fused lazy Adam against PyTorch on the touched rows only."""
    from smlm.kernels import bag_backward_rows, lazy_adam_step
    rows, D, N, K = 6_000_000, 384, 1024, 128
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    table = torch.randn(rows, D, device=DEVICE, generator=gen) * D ** -0.5
    idx = torch.randint(rows - 400_000, rows, (N, K), device=DEVICE, generator=gen)   # rows > 5.6M
    idx[:, :8] = torch.randint(0, rows, (N, 8), device=DEVICE, generator=gen)
    w = torch.softmax(torch.randn(N, K, device=DEVICE, generator=gen), -1)
    g = torch.randn(N, D, device=DEVICE, generator=gen)
    rows_read = table[idx.reshape(-1)].view(N, K, D)                                  # (N, K, D) gather
    close(bag_forward(idx, w, table), (w[..., None] * rows_read).sum(1), "bag forward")
    acc = torch.zeros_like(table)
    touched = torch.zeros(rows, dtype=torch.bool, device=DEVICE)
    gw = bag_backward_rows(g, idx, w, table, acc, touched)
    close(gw, (rows_read * g[:, None, :]).sum(-1), "per-sample-weight gradient")
    uniq, inv = torch.unique(idx.reshape(-1), return_inverse=True)
    ref_acc = torch.zeros(uniq.numel(), D, device=DEVICE).index_add_(0, inv, (w[..., None] * g[:, None, :]).view(-1, D))
    close(acc[uniq], ref_acc, "accumulator rows")
    assert int(touched.sum()) == uniq.numel() and bool(touched[uniq].all())
    del rows_read
    m, v = torch.zeros_like(table), torch.zeros_like(table)
    p0 = table[uniq].clone()
    untouched_probe = torch.tensor([0, rows // 2, rows - 1], device=DEVICE)
    untouched_probe = untouched_probe[~touched[untouched_probe]]
    before = table[untouched_probe].clone()
    lazy_adam_step(table, m, v, acc, touched, 1.0, 1, 2.4e-3, 0.9, 0.95, 1e-8)
    gr = ref_acc
    m_ref, v_ref = 0.1 * gr, 0.05 * gr * gr
    p_ref = p0 - (2.4e-3 / 0.1) * m_ref / (v_ref.sqrt() / 0.05 ** 0.5 + 1e-8)
    close(table[uniq], p_ref, "lazy Adam values")
    assert torch.equal(table[untouched_probe], before)
    assert not touched.any() and torch.count_nonzero(acc[uniq]) == 0
