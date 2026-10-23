"""HOT (smlm/host_values.py): the most-read rows of a host table on the GPU give the same reads, gradients and
updates as all rows in RAM; flush() / state_dict() write them back.

  python -m pytest -q tests/test_hot_rows.py
"""
import pytest
import torch
from torch import nn

from smlm.host_values import HostEmbedding, enable_hot_rows
from smlm.row_optim import RowWiseOptimizer
from smlm.sparse_values import clip_row_sparse, row_store

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Holder(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.values = emb


def make(initial, device, hot_frac=None, reads=None):
    emb = HostEmbedding(nn.Embedding(*initial.shape, _weight=initial.clone()))
    emb.weight.pk_value_param = True
    model = Holder(emb).to(device)
    if hot_frac:
        enable_hot_rows(model, reads, hot_frac, torch.device(device))
    return model, RowWiseOptimizer([emb.weight], lr=0.05, kind="rowrms")


@pytest.mark.parametrize("device", DEVICES)
def test_hot_rows_train_like_rows_in_ram(device):
    gen = torch.Generator().manual_seed(0)
    initial = torch.randn(200, 16, generator=gen)
    reads = torch.randint(0, 1000, (200,), generator=gen)
    (plain, opt_a), (hot, opt_b) = make(initial, device), make(initial, device, 0.3, reads)
    assert hot.values.hot.rows.numel() == 60
    assert torch.equal(hot.values.hot.rows.cpu(), torch.argsort(reads, descending=True, stable=True)[:60].sort().values)
    for step in range(4):
        idx = torch.randint(0, 200, (12, 5), generator=gen).to(device)
        w = torch.rand(12, 5, generator=gen).to(device).requires_grad_(True)
        outs = []
        for model in (plain, hot):
            out = model.values.bag(idx, w)
            out.square().sum().backward()
            outs.append(out.detach())
        torch.testing.assert_close(outs[1], outs[0], rtol=1e-6, atol=1e-6)
        norms = [clip_row_sparse([m.values.weight], 1.0) for m in (plain, hot)]
        torch.testing.assert_close(norms[1], norms[0], rtol=1e-5, atol=1e-6)
        opt_a.step()
        opt_b.step()
    hot_table = hot.state_dict()["values.weight"]              # state_dict flushes the GPU rows into the table
    torch.testing.assert_close(hot_table, plain.values.weight.detach(), rtol=1e-5, atol=1e-6)
    sa, sb = opt_a.state_dict()["state"][0], opt_b.state_dict()["state"][0]
    assert sa.keys() == sb.keys() == {"step", "v", "n"}
    torch.testing.assert_close(sb["v"], sa["v"], rtol=1e-5, atol=1e-9)
    assert torch.equal(sb["n"], sa["n"])
    assert not row_store(hot.values.weight).touched.any() and not hot.values.hot.touched.any()


def test_hot_rows_need_a_row_wise_optimizer():
    from smlm.host_optim import HostLazyRowAdam
    model, _ = make(torch.randn(50, 8), "cpu", 0.2, torch.arange(50))
    opt = HostLazyRowAdam([model.values.weight], lr=0.01)
    out = model.values.bag(torch.tensor([[1, 49]]), torch.ones(1, 2, requires_grad=True))
    out.sum().backward()
    with pytest.raises(ValueError, match="row-wise"):
        opt.step()


def test_reset_clears_hot_rows_too():
    model, _ = make(torch.randn(50, 8), "cpu", 0.2, torch.arange(50))
    out = model.values.bag(torch.tensor([[1, 49, 45]]), torch.ones(1, 3, requires_grad=True))
    out.sum().backward()
    st = row_store(model.values.weight)
    assert st.hot.touched.sum() == 2 and st.touched.sum() == 1          # rows 49, 45 hot; row 1 in RAM
    st.reset()
    assert not st.hot.touched.any() and torch.count_nonzero(st.hot.acc) == 0 and not st.touched.any()
