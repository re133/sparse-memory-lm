"""Step 14 (EINP): TableWriter does plain gradient descent on exactly the table rows read, and restore() undoes it.

CPU, tiny model: the rows it changes and their new values match a dense autograd gradient of the same loss on the
whole table; every other parameter and row stays bit-identical; restore() gives back the original table.
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F

from smlm.model import ModelConfig, Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from einp import TableWriter, write_person  # noqa: E402


def tiny(seed=0):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[0, 2, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16,
                      mem_share_values=True, mem_value_grad="row_sparse")
    torch.manual_seed(seed)
    m = Transformer(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def art():
    ids = [5, 17, 3, 99, 42, 7, 64, 11, 23, 80, 2]
    return {"ids": ids, "values": {"year": {"pos": 4, "rank": 0}, "city": {"pos": 7, "rank": 1},
                                   "job": {"pos": 9, "rank": 2}}}


def test_step_is_gradient_descent_on_read_rows_only():
    m, ref = tiny(), tiny()
    table0 = m.memory_layers()[0].values.weight.detach().clone()
    other0 = {k: v.clone() for k, v in m.state_dict().items() if "values" not in k}
    w = TableWriter(m)
    write_person(m, w, art(), steps=1, lr=0.5, device="cpu")
    # reference: dense gradient of the same loss with respect to the whole table
    t = ref.memory_layers()[0].values.weight
    t.requires_grad_(True)
    for mem in ref.memory_layers():
        mem.value_grad = "dense"
    a = art()
    pos = [v["pos"] for v in a["values"].values()]
    logits = ref(torch.tensor([a["ids"]]))
    F.cross_entropy(logits[0, [p - 1 for p in pos]].float(), torch.tensor([a["ids"][p] for p in pos])).backward()
    table1 = m.memory_layers()[0].values.weight.detach()
    assert torch.allclose(table1, table0 - 0.5 * t.grad, atol=1e-6)
    changed = (table1 != table0).any(1)
    assert torch.equal(changed, t.grad.abs().sum(1) > 0) and changed.sum() > 0
    assert all(torch.equal(v, other0[k]) for k, v in m.state_dict().items() if "values" not in k)
    w.restore()
    assert torch.equal(m.memory_layers()[0].values.weight.detach(), table0)


def test_several_steps_and_people_restore_exactly():
    m = tiny(1)
    table0 = m.memory_layers()[0].values.weight.detach().clone()
    w = TableWriter(m)
    b = copy.deepcopy(art())
    b["ids"] = [x + 1 for x in b["ids"]]
    for x in (art(), b):
        write_person(m, w, x, steps=3, lr=2.0, device="cpu")
    assert not torch.equal(m.memory_layers()[0].values.weight.detach(), table0)
    w.restore()
    assert torch.equal(m.memory_layers()[0].values.weight.detach(), table0)
    assert not any("read_values" in vars(mem) for mem in m.memory_layers())


def test_layer_subset_writes_only_rows_that_layer_reads():
    m = tiny(2)
    table0 = m.memory_layers()[0].values.weight.detach().clone()
    for mem in m.memory_layers():
        mem.record = True
    a = art()
    with torch.no_grad():
        m(torch.tensor([a["ids"]]))
    read_last = set(m.memory_layers()[2].last_indices.reshape(-1).tolist())
    for mem in m.memory_layers():
        mem.record = False
    w = TableWriter(m, layers=[2])
    write_person(m, w, a, steps=1, lr=1.0, device="cpu")
    changed = set(torch.nonzero((m.memory_layers()[0].values.weight.detach() != table0).any(1)).view(-1).tolist())
    assert changed and changed <= read_last
    w.restore()
    assert torch.equal(m.memory_layers()[0].values.weight.detach(), table0)
