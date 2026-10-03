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
    torch.testing.assert_close(y1, y0, rtol=0, atol=0)
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
