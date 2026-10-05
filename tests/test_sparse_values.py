"""Row-sparse table gradients + lazy Adam (Hampter step 1)."""
import copy

import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.optim import build_optimizer, clip_grads
from smlm.sparse_values import LazyRowAdam, OptimizerSet, row_sparse_tables, row_store
from smlm.train import MODELS

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny(value_grad, seed=0, device="cpu", dtype=torch.float64):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[0, 2, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16,
                      mem_share_values=True, mem_value_grad=value_grad)
    torch.manual_seed(seed)
    return Transformer(cfg).to(dtype).to(device)


def table_of(m):
    return m.memory_layers()[0].values.weight


def backward_twice(m, device):
    """Two micro-batches (gradient accumulation) through three memory layers sharing one table."""
    torch.manual_seed(1)
    read = set()
    for _ in range(2):
        for x in m.memory_layers():
            x.record = True
        idx = torch.randint(0, 128, (2, 12), device=device)
        _, loss = m(idx[:, :-1], idx[:, 1:])
        (loss / 2).backward()
        for x in m.memory_layers():
            read |= set(x.last_indices.reshape(-1).tolist())
    return read


@pytest.mark.parametrize("device", DEVICES)
def test_row_sparse_gradients_equal_dense(device):
    dense, sparse = tiny("dense", device=device), tiny("row_sparse", device=device)
    sparse.load_state_dict(dense.state_dict())
    read_d = backward_twice(dense, device)
    read_s = backward_twice(sparse, device)
    assert read_d == read_s
    st = row_store(table_of(sparse))
    assert table_of(sparse).grad is None                               # the table never gets a dense .grad
    torch.testing.assert_close(st.acc, table_of(dense).grad)          # same row gradients
    touched = set(torch.nonzero(st.touched).squeeze(1).tolist())
    assert touched == read_s
    # every other parameter (keys, query, ... learn through the per-sample weights) has the same gradient
    for (n, pd), ps in zip(dense.named_parameters(), sparse.parameters()):
        if pd is table_of(dense):
            continue
        torch.testing.assert_close(ps.grad, pd.grad, msg=n)


def test_clip_norm_matches_dense():
    dense, sparse = tiny("dense"), tiny("row_sparse")
    sparse.load_state_dict(dense.state_dict())
    backward_twice(dense, "cpu")
    backward_twice(sparse, "cpu")
    nd, vd = clip_grads(dense, 1e-3)
    ns, vs = clip_grads(sparse, 1e-3)
    torch.testing.assert_close(ns, nd)
    torch.testing.assert_close(vs, vd)
    assert float(row_store(table_of(sparse)).grad_scale) == pytest.approx(min(1.0, 1e-3 / (float(vs) + 1e-6)))


@pytest.mark.parametrize("device", DEVICES)
def test_unread_rows_unchanged_values_and_state(device):
    """One step of the lazy optimizer: rows that were not read keep their values and Adam state exactly."""
    m = tiny("row_sparse", device=device)
    opt = build_optimizer(m, 1e-2, 2.4e-2, 0.1)
    assert isinstance(opt, OptimizerSet)
    lazy = next(o for o in opt.opts if isinstance(o, LazyRowAdam))
    table = table_of(m)
    assert all(table is not p for g in opt.opts[0].param_groups for p in g["params"])   # not in AdamW
    for step in range(3):
        before = table.detach().clone()
        st_before = copy.deepcopy({k: v.clone() if torch.is_tensor(v) else v
                                   for k, v in lazy.state[table].items()}) if lazy.state[table] else None
        read = backward_twice(m, device)
        rows = torch.tensor(sorted(read), device=device)
        unread = torch.ones(table.shape[0], dtype=torch.bool, device=device)
        unread[rows] = False
        clip_grads(m, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        after = table.detach()
        assert torch.equal(after[unread], before[unread])
        assert not torch.equal(after[rows], before[rows])
        st = lazy.state[table]
        if st_before is not None:
            assert torch.equal(st["exp_avg"][unread], st_before["exp_avg"][unread])
            assert torch.equal(st["exp_avg_sq"][unread], st_before["exp_avg_sq"][unread])
        else:
            assert torch.count_nonzero(st["exp_avg"][unread]) == 0
        assert st["step"] == step + 1
        assert not row_store(table).touched.any()                         # accumulator cleared
        assert torch.count_nonzero(row_store(table).acc) == 0


def test_lazy_adam_first_step_equals_adam_on_read_rows():
    """At step 1 (zero state) lazy Adam on the read rows equals torch Adam (no weight decay)."""
    torch.manual_seed(0)
    table = torch.nn.Parameter(torch.randn(50, 8, dtype=torch.float64))
    ref = torch.nn.Parameter(table.detach().clone())
    grad = torch.zeros_like(table)
    rows = torch.tensor([3, 7, 7, 20, 41])
    grad.index_add_(0, rows, torch.randn(5, 8, dtype=torch.float64))
    st = row_store(table)
    st.acc.copy_(grad)
    st.touched[rows] = True
    LazyRowAdam([table], lr=0.01, betas=(0.9, 0.95), eps=1e-8).step()
    ref.grad = grad
    torch.optim.Adam([ref], lr=0.01, betas=(0.9, 0.95), eps=1e-8).step()
    touched = torch.unique(rows)
    torch.testing.assert_close(table.detach()[touched], ref.detach()[touched])
    mask = torch.ones(50, dtype=torch.bool)
    mask[touched] = False
    assert torch.equal(table.detach()[mask], ref.detach()[mask])         # untouched: unchanged in both


def test_presets_and_eval_path():
    assert MODELS["B-1M-sparse"] == dict(MODELS["B-1M"], mem_value_grad="row_sparse")
    m = tiny("row_sparse").eval()
    x = torch.randint(0, 128, (1, 20))
    with torch.no_grad():
        out = m(x)                                                          # no RowStore needed for inference
    d = tiny("dense").eval()
    d.load_state_dict(m.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(out, d(x))
    assert row_sparse_tables(m) == [table_of(m)] and row_sparse_tables(d) == []


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_frozen_table_and_table_only_training(device, impl):
    """A frozen table stays exactly as it is (and gets no row accumulator) while the rest trains; with
    everything else frozen, the table alone trains (the lookup output still needs a gradient)."""
    if impl == "triton" and device == "cpu":
        pytest.skip("Triton kernels on the GPU (CPU: TRITON_INTERPRET=1 in test_kernels.py)")
    dtype = torch.float32 if impl == "triton" else torch.float64
    for frozen in ("table", "rest"):
        m = tiny("row_sparse", device=device, dtype=dtype)
        for x in m.memory_layers():
            x.impl = impl
        table = table_of(m)
        for p in m.parameters():
            p.requires_grad_((p is not table) if frozen == "table" else (p is table))
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = build_optimizer(m, 1e-2, 1e-2, 0.0)
        backward_twice(m, device)
        clip_grads(m, 1.0)
        opt.step()
        changed = {n for n, p in m.named_parameters() if not torch.equal(p.detach(), before[n])}
        if frozen == "table":
            assert "layers.0.ffn.values.weight" not in changed and not hasattr(table, "row_store")
            assert any("query_proj" in n for n in changed)
        else:
            assert changed == {"layers.0.ffn.values.weight"}


def test_lazy_adam_refuses_state_dict_load():
    """Resuming isn't supported: loading optimizer state must fail loudly, not restart the bias correction."""
    m = tiny("row_sparse")
    opt = LazyRowAdam(row_sparse_tables(m), lr=1e-2)
    with pytest.raises(NotImplementedError):
        opt.load_state_dict(opt.state_dict())
