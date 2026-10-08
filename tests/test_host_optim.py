"""Touched-row CPU Adam and mixed-device clipping for host tables.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_host_optim.py
"""
import pytest
import torch
from torch import nn

from smlm.host_optim import HostLazyRowAdam, HostRowStore, value_table_memory_by_device
from smlm.optim import build_optimizer, value_table_memory
from smlm.sparse_values import LazyRowAdam, clip_row_sparse, row_store


def table(initial, host=False):
    p = nn.Parameter(initial.clone())
    p.host_values = host
    p.pk_value_param = True
    return p


@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_host_adam_matches_cpu_reference_with_accumulation_and_clipping(mode, monkeypatch):
    import smlm.host_optim as host_optim
    monkeypatch.setattr(host_optim, "CHUNK_ELEMENTS", 26)
    generator = torch.Generator().manual_seed(42)
    initial = torch.randn(19, 13, generator=generator)
    normal, host = table(initial), table(initial, host=True)
    optimizers = [LazyRowAdam([normal], lr=0.01, state_dtype=mode),
                  HostLazyRowAdam([host], lr=0.01, state_dtype=mode)]
    visits = [torch.arange(19), torch.tensor([1, 2, 7]), torch.tensor([4, 8]), torch.tensor([1, 7, 18])]
    for step, rows in enumerate(visits):
        before = host.detach().clone()
        previous = {k: v.clone() for k, v in optimizers[1].state[host].items() if k != "step"}
        for _ in range(3):
            gradient = torch.randn(rows.numel(), 13, generator=generator)
            for p in (normal, host):
                st = row_store(p)
                st.acc.index_add_(0, rows, gradient)
                st.touched[rows] = True
        norms = [clip_row_sparse([p], 0.25) for p in (normal, host)]
        torch.testing.assert_close(*norms, rtol=2e-7, atol=1e-6)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(host, normal, rtol=2e-6, atol=2e-7)
        for key, expected in optimizers[0].state[normal].items():
            actual = optimizers[1].state[host][key]
            if actual.is_floating_point():
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
            else:
                assert torch.equal(actual, expected), key
        unread = torch.ones(19, dtype=torch.bool)
        unread[rows] = False
        assert torch.equal(host[unread], before[unread])
        for key, values in previous.items():
            assert torch.equal(optimizers[1].state[host][key][unread], values[unread]), key
        assert optimizers[1].state[host]["step"] == step + 1
        assert host.grad is None and torch.count_nonzero(row_store(host).acc) == 0
        assert not row_store(host).touched.any()


def test_c_row_loops_match_torch_path(monkeypatch):
    """smlm/host_rows.c against the torch path: scatter-add and gather exactly, Adam within float rounding (torch's
    lerp may fuse), on enough rows that several threads share the work; unread rows stay bit-identical."""
    from smlm.host_optim import rows_library
    from smlm.host_values import HostEmbedding
    if rows_library() is None:
        pytest.skip("no C compiler")
    initial = torch.randn(5000, 384, generator=torch.Generator().manual_seed(3))
    runs = {}
    for c in ("1", "0"):
        monkeypatch.setenv("SMLM_HOST_C", c)
        generator = torch.Generator().manual_seed(4)            # same rows and gradients in both runs
        emb = HostEmbedding(nn.Embedding(5000, 384, _weight=initial.clone()))
        p = emb.weight
        p.pk_value_param = True
        opt = HostLazyRowAdam([p], lr=0.01)
        steps = []
        for step in range(3):
            for _ in range(2):
                rows = torch.randperm(5000, generator=generator)[:2000].sort().values
                emb.accumulate(rows, torch.randn(2000, 384, generator=generator))
            acc = row_store(p).acc.clone()
            row_store(p).grad_scale = 0.5
            staged = torch.empty(2000, 384)
            if c == "1":
                rows_library().host_rows_gather(staged.data_ptr(), p.data_ptr(), rows.data_ptr(), 2000, 384, 4)
            else:
                torch.index_select(p.detach(), 0, rows, out=staged)
            touched = row_store(p).touched.clone()
            opt.step()
            steps.append((acc, staged, touched, p.detach().clone(), opt.state[p]["exp_avg"].clone(),
                          opt.state[p]["exp_avg_sq"].clone()))
            assert torch.count_nonzero(row_store(p).acc) == 0 and not row_store(p).touched.any()
            assert row_store(p).grad_scale == 1.0
            assert torch.equal(p.detach()[~touched], (steps[-2][3] if step else initial)[~touched])
        runs[c] = steps
    for fast, ref in zip(runs["1"], runs["0"]):
        assert torch.equal(fast[0], ref[0]) and torch.equal(fast[2], ref[2])      # accumulate, touched
        torch.testing.assert_close(fast[1], ref[1], rtol=2e-6, atol=2e-7)          # gather of the moved values
        for a, r in zip(fast[3:], ref[3:]):
            torch.testing.assert_close(a, r, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_empty_host_step_keeps_torch_semantics(mode):
    p = table(torch.ones(7, 3), host=True)
    opt = HostLazyRowAdam([p], lr=0.01, state_dtype=mode)
    opt.step()
    assert not opt.state[p]
    st = row_store(p)
    st.acc[1].fill_(0.2)
    st.touched[1] = True
    opt.step()
    before = p.detach().clone()
    state = {k: v.clone() for k, v in opt.state[p].items()}
    opt.step()
    assert torch.equal(before, p)
    assert all(torch.equal(v, opt.state[p][k]) for k, v in state.items())


def test_host_store_promotes_existing_accumulator_and_touches_only_visited_rows(monkeypatch):
    import smlm.host_optim as host_optim
    monkeypatch.setattr(host_optim, "CHUNK_ELEMENTS", 6)
    p = table(torch.zeros(11, 3))
    st = row_store(p)
    st.acc.fill_(float("nan"))
    rows = torch.tensor([1, 3, 6, 10])
    st.acc[rows] = 2.0
    st.touched[rows] = True
    p.host_values = True
    assert row_store(p) is st and isinstance(st, HostRowStore)
    torch.testing.assert_close(st.norm(), torch.full((4, 3), 2.0).norm())
    st.reset()
    assert not st.touched.any() and st.grad_scale == 1.0
    assert torch.count_nonzero(st.acc[rows]) == 0
    assert st.acc[0].isnan().all()
    assert st.norm() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_clipping_combines_host_and_device_accumulators(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable")
    host = table(torch.zeros(5, 7), host=True)
    normal = table(torch.zeros(9, 7, device=device))
    for p, magnitude in ((host, 2.0), (normal, 3.0)):
        st = row_store(p)
        st.acc[1].fill_(magnitude)
        st.touched[1] = True
    expected = torch.cat((row_store(host).acc.flatten(), row_store(normal).acc.cpu().flatten())).norm()
    for tables in ([host, normal], [normal, host]):
        total = clip_row_sparse(tables, 0.1)
        torch.testing.assert_close(total.cpu(), expected)
        for p in tables:
            scale = row_store(p).grad_scale
            assert scale.device == p.device
            torch.testing.assert_close(scale.cpu(), 0.1 / (expected + 1e-6))


@pytest.mark.parametrize("mode", ["fp32", "bf16", "int8"])
def test_optimizer_routing_and_memory_split(mode):
    model = nn.Module()
    model.dense = nn.Linear(3, 3)
    model.memories = nn.ModuleList()
    for host in (True, False):
        memory = nn.Module()
        memory.value_grad, memory.impl = "row_sparse", "torch"
        memory.values = nn.Embedding(5, 3)
        memory.values.weight.host_values = host
        memory.values.weight.pk_value_param = True
        model.memories.append(memory)
    opt = build_optimizer(model, 0.01, 0.02, 0.0, value_state=mode)
    lazy = [o for o in opt.opts if isinstance(o, LazyRowAdam)]
    assert len(lazy) == 2
    assert isinstance(lazy[0], HostLazyRowAdam) and lazy[1].__class__ is LazyRowAdam
    for memory, optimizer in zip(model.memories, lazy):
        p = memory.values.weight
        assert optimizer.param_groups[0]["params"][0] is p
        row_store(p).touched[1] = True
        row_store(p).acc[1].fill_(0.25)
    opt.step()
    split = value_table_memory_by_device(model, mode)
    total = value_table_memory(model, mode)
    assert all(split["host"][key] + split["device"][key] == amount for key, amount in total.items())
    for location, memory, optimizer in zip(("host", "device"), model.memories, lazy):
        p = memory.values.weight
        tensors = [p, row_store(p).acc, row_store(p).touched, *optimizer.state[p].values()]
        actual = sum(t.numel() * t.element_size() for t in tensors)
        assert split[location]["total_bytes"] == actual
