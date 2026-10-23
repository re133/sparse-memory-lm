"""Row-wise table optimizers (smlm/row_optim.py): formulas, C against torch, GPU against CPU, state, wiring.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_row_optim.py
"""
import math

import pytest
import torch
from torch import nn

from smlm.optim import build_optimizer, value_table_memory
from smlm.row_optim import RowWiseOptimizer
from smlm.sparse_values import clip_row_sparse, row_store


def table(initial, host=False, device="cpu"):
    p = nn.Parameter(initial.clone().to(device))
    p.host_values = host
    p.pk_value_param = True
    return p


def add_grad(p, rows, gradient):
    st = row_store(p)
    st.acc.index_add_(0, rows.to(st.acc.device), gradient.to(st.acc.device))
    st.touched[rows.to(st.acc.device)] = True


def reference(kind, values, grads_by_step, lr, b2=0.95, eps=1e-8):
    """The formulas of the module docstring, row by row in float64."""
    p = values.double().clone()
    v = torch.zeros(p.shape[0], dtype=torch.float64)
    n = torch.zeros(p.shape[0], dtype=torch.int64)
    for grads in grads_by_step:
        for r, g in grads.items():
            g = g.double()
            q = g.square().mean()
            if kind == "rowrms":
                v[r] = b2 * v[r] + (1 - b2) * q
                n[r] += 1
                denom = math.sqrt(v[r] / (1 - b2 ** int(n[r]))) + eps
            else:
                v[r] += q
                denom = math.sqrt(v[r]) + eps
            p[r] -= lr / denom * g
    return p, v, n


@pytest.mark.parametrize("c", ["1", "0"])
@pytest.mark.parametrize("kind", ["rowrms", "rowadagrad"])
def test_host_path_matches_formula_and_keeps_unread_rows(kind, c, monkeypatch):
    monkeypatch.setenv("SMLM_HOST_C", c)
    gen = torch.Generator().manual_seed(1)
    initial = torch.randn(300, 24, generator=gen)
    p = table(initial, host=True)
    opt = RowWiseOptimizer([p], lr=0.01, kind=kind)
    visits = [torch.arange(300), torch.tensor([1, 2, 7]), torch.tensor([4, 8, 299]), torch.tensor([1, 7, 150])]
    history = []
    for rows in visits:
        before = p.detach().clone()
        grads = {}
        for _ in range(2):                                     # two micro-batches add up in the accumulator
            g = torch.randn(rows.numel(), 24, generator=gen)
            add_grad(p, rows, g)
            for r, gr in zip(rows.tolist(), g):
                grads[r] = grads.get(r, 0) + gr
        history.append(grads)
        opt.step()
        unread = torch.ones(300, dtype=torch.bool)
        unread[rows] = False
        assert torch.equal(p.detach()[unread], before[unread])
        assert torch.count_nonzero(row_store(p).acc) == 0 and not row_store(p).touched.any()
    want_p, want_v, want_n = reference(kind, initial, history, 0.01)
    torch.testing.assert_close(p.detach().double(), want_p, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(opt.state[p]["v"].double(), want_v, rtol=1e-5, atol=1e-9)
    if kind == "rowrms":
        assert torch.equal(opt.state[p]["n"].long(), want_n)
    assert int(opt.state[p]["step"]) == len(visits)


def test_rowrms_counts_each_rows_own_updates(monkeypatch):
    """A row first read after many steps takes a first step of size lr (in RMS over the row), not lr / sqrt(1 - b2)."""
    p = table(torch.zeros(4, 16), host=True)
    opt = RowWiseOptimizer([p], lr=0.01, kind="rowrms")
    gen = torch.Generator().manual_seed(2)
    for _ in range(50):
        add_grad(p, torch.tensor([0]), torch.randn(1, 16, generator=gen))
        opt.step()
    add_grad(p, torch.tensor([3]), torch.randn(1, 16, generator=gen))
    opt.step()
    rms = p.detach()[3].square().mean().sqrt()
    torch.testing.assert_close(rms, torch.tensor(0.01), rtol=1e-5, atol=0)


def test_clip_factor_is_applied(monkeypatch):
    p, q = table(torch.randn(50, 8), host=True), table(torch.randn(50, 8), host=True)
    q.data.copy_(p.data)
    opts = [RowWiseOptimizer([t], lr=0.01, kind="rowadagrad") for t in (p, q)]
    g = torch.randn(10, 8, generator=torch.Generator().manual_seed(3))
    rows = torch.arange(10)
    add_grad(p, rows, g)
    add_grad(q, rows, g * 0.5)
    clip_row_sparse([p], g.norm().item() * 0.5)                # p's gradient is scaled to q's
    for o in opts:
        o.step()
    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU path")
@pytest.mark.parametrize("kind", ["rowrms", "rowadagrad"])
def test_gpu_path_matches_host_path(kind, monkeypatch):
    import smlm.row_optim as row_optim
    monkeypatch.setattr(row_optim, "CHUNK_ROWS", 70)          # several chunks, the last one partial
    gen = torch.Generator().manual_seed(4)
    initial = torch.randn(300, 24, generator=gen)
    host, gpu = table(initial, host=True), table(initial, device="cuda")
    opts = [RowWiseOptimizer([t], lr=0.01, kind=kind) for t in (host, gpu)]
    for rows in (torch.arange(300), torch.tensor([1, 2, 7, 140]), torch.tensor([4, 8, 299])):
        before = gpu.detach().clone()
        g = torch.randn(rows.numel(), 24, generator=gen)
        for t in (host, gpu):
            add_grad(t, rows, g)
        norms = [clip_row_sparse([t], 2.0) for t in (host, gpu)]
        torch.testing.assert_close(norms[0], norms[1].cpu(), rtol=1e-5, atol=1e-6)
        for o in opts:
            o.step()
        torch.testing.assert_close(gpu.detach().cpu(), host.detach(), rtol=1e-5, atol=1e-6)
        unread = torch.ones(300, dtype=torch.bool, device="cuda")
        unread[rows.cuda()] = False
        assert torch.equal(gpu.detach()[unread], before[unread])
        assert torch.count_nonzero(row_store(gpu).acc) == 0 and not row_store(gpu).touched.any()
    for k in opts[0].state[host]:
        if k != "step":
            torch.testing.assert_close(opts[1].state[gpu][k].cpu(), opts[0].state[host][k], rtol=1e-5, atol=1e-9)


def test_c_loop_on_many_rows_and_threads(monkeypatch):
    """C against torch on enough rows that several threads share the work."""
    from smlm.host_optim import rows_library
    if rows_library() is None:
        pytest.skip("no C compiler")
    initial = torch.randn(5000, 384, generator=torch.Generator().manual_seed(5))
    out = {}
    for c in ("1", "0"):
        monkeypatch.setenv("SMLM_HOST_C", c)
        gen = torch.Generator().manual_seed(6)
        p = table(initial, host=True)
        opt = RowWiseOptimizer([p], lr=0.01, kind="rowrms")
        for _ in range(3):
            rows = torch.randperm(5000, generator=gen)[:2000].sort().values
            add_grad(p, rows, torch.randn(2000, 384, generator=gen))
            opt.step()
        out[c] = (p.detach().clone(), opt.state[p]["v"].clone(), opt.state[p]["n"].clone())
    torch.testing.assert_close(out["1"][0], out["0"][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out["1"][1], out["0"][1], rtol=1e-5, atol=1e-9)
    assert torch.equal(out["1"][2], out["0"][2])


def test_state_dict_round_trip():
    gen = torch.Generator().manual_seed(7)
    a, b = table(torch.randn(20, 6, generator=gen), host=True), None
    opt = RowWiseOptimizer([a], lr=0.01, kind="rowrms")
    add_grad(a, torch.tensor([1, 5]), torch.randn(2, 6, generator=gen))
    opt.step()
    b = table(a.detach(), host=True)
    opt2 = RowWiseOptimizer([b], lr=0.01, kind="rowrms")
    opt2.load_state_dict(opt.state_dict())
    g = torch.randn(3, 6, generator=gen)
    for t, o in ((a, opt), (b, opt2)):
        add_grad(t, torch.tensor([1, 2, 5]), g)
        o.step()
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        RowWiseOptimizer([b], lr=0.01, kind="rowadagrad").load_state_dict(opt.state_dict())


def test_build_optimizer_routes_tables_and_counts_state():
    from smlm.model import ModelConfig, Transformer
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=16,
                      mem_layers=[1], mem_n_keys=8, mem_heads=2, mem_knn=4, mem_k_dim=16, mem_v_dim=32,
                      mem_value_grad="row_sparse")
    model = Transformer(cfg)
    opt = build_optimizer(model, 1e-3, 2e-3, 0.1, value_optim="rowrms")
    rowwise = [o for o in opt.opts if isinstance(o, RowWiseOptimizer)]
    assert len(rowwise) == 1 and rowwise[0].param_groups[0]["lr"] == 2e-3
    rows = model.memory_layers()[0].values.weight.shape[0]
    assert value_table_memory(model, value_optim="rowrms")["moment_bytes"] == rows * 8
    assert value_table_memory(model, value_optim="rowadagrad")["moment_bytes"] == rows * 4
    with pytest.raises(ValueError):
        build_optimizer(model, 1e-3, 2e-3, 0.1, value_optim="rowrms", value_state="int8")
