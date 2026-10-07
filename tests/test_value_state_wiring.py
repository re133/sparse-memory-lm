"""Compact value optimizer routing and persistent memory estimates.

  python -m pytest -q tests/test_value_state_wiring.py
"""
import pytest
import torch
from torch import nn

from smlm.engram import EngramMemory
from smlm.optim import build_optimizer, value_table_memory
from smlm.pkm import ProductKeyMemory
from smlm.sparse_values import LazyRowAdam, OptimizerSet, row_sparse_tables, row_store


def memories(pkm_grad="row_sparse", eng_grad="row_sparse", impl="torch"):
    model = nn.Module()
    shared = nn.Embedding(16, 8)
    model.pkm = nn.ModuleList([ProductKeyMemory(8, 8, n_keys=4, heads=1, knn=2, k_dim=4,
                                               shared_values=shared, value_grad=pkm_grad, impl=impl)
                               for _ in range(2)])
    model.engram = EngramMemory(8, n_lookups=3, head_dim=5, rows_per_head=7,
                               value_grad=eng_grad, impl=impl)
    return model


@pytest.mark.parametrize("value_state", ["fp32", "bf16", "int8"])
@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_both_memories_route_state_and_learning_rate(value_state, impl):
    model = memories(impl=impl)
    opt = build_optimizer(model, 1e-3, 2e-3, 0.1, eng_value_lr=3e-3, value_state=value_state)
    assert isinstance(opt, OptimizerSet)
    lazy = {o.param_groups[0]["name"]: o for o in opt.opts if isinstance(o, LazyRowAdam)}
    assert set(lazy) == {"memory_values", "engram_values"}
    for name, table, lr in [("memory_values", model.pkm[0].values.weight, 2e-3),
                            ("engram_values", model.engram.values.weight, 3e-3)]:
        adam = lazy[name]
        assert adam.state_dtype == value_state and adam.impl == impl
        assert adam.param_groups[0]["lr"] == lr
        assert len(adam.param_groups[0]["params"]) == 1
        assert adam.param_groups[0]["params"][0] is table
        assert all(p is not table for g in opt.opts[0].param_groups for p in g["params"])


@pytest.mark.parametrize("value_state,bytes_per_moment_pair", [("fp32", 8), ("bf16", 4), ("int8", 2)])
def test_memory_estimate_matches_allocated_tensors(value_state, bytes_per_moment_pair):
    model = memories()
    estimate = value_table_memory(model, value_state)
    opt = build_optimizer(model, 1e-3, 2e-3, 0.1, value_state=value_state)
    tables = row_sparse_tables(model)
    assert len(tables) == 2                                      # shared PKM table counts once
    for table in tables:
        store = row_store(table)
        store.acc[0].fill_(0.1)
        store.touched[0] = True
    opt.step()
    n = sum(p.numel() for p in tables)
    rows = sum(p.shape[0] for p in tables)
    assert estimate == dict(table_bytes=4 * n, gradient_bytes=4 * n,
                            moment_bytes=bytes_per_moment_pair * n,
                            scale_bytes=8 * rows if value_state == "int8" else 0,
                            touched_bytes=rows, step_bytes=4 * len(tables),
                            total_bytes=(8 + bytes_per_moment_pair) * n + rows + 4 * len(tables)
                            + (8 * rows if value_state == "int8" else 0))
    tensors = list(tables)
    for table in tables:
        tensors.extend((row_store(table).acc, row_store(table).touched))
    for adam in opt.opts:
        if isinstance(adam, LazyRowAdam):
            for state in adam.state.values():
                tensors.extend(v for v in state.values() if torch.is_tensor(v))
    assert estimate["total_bytes"] == sum(t.numel() * t.element_size() for t in tensors)


@pytest.mark.parametrize("dense_kind", ["pkm", "engram"])
@pytest.mark.parametrize("value_state", ["bf16", "int8"])
def test_compact_states_reject_dense_trainable_tables(dense_kind, value_state):
    model = memories(pkm_grad="dense" if dense_kind == "pkm" else "row_sparse",
                     eng_grad="dense" if dense_kind == "engram" else "row_sparse")
    with pytest.raises(ValueError, match="row_sparse"):
        build_optimizer(model, 1e-3, 2e-3, 0.1, value_state=value_state)
    with pytest.raises(ValueError, match="row_sparse"):
        value_table_memory(model, value_state)


def test_dense_default_and_frozen_table_estimates():
    model = memories(pkm_grad="dense", eng_grad="dense")
    opt = build_optimizer(model, 1e-3, 2e-3, 0.1)
    assert isinstance(opt, torch.optim.AdamW)
    tables = [model.pkm[0].values.weight, model.engram.values.weight]
    n = sum(p.numel() for p in tables)
    assert value_table_memory(model)["total_bytes"] == 16 * n + 4 * len(tables)
    for table in tables:
        table.requires_grad_(False)
    for value_state in ("fp32", "bf16", "int8"):
        build_optimizer(model, 1e-3, 2e-3, 0.1, value_state=value_state)
        assert value_table_memory(model, value_state)["total_bytes"] == 4 * n


def test_fp64_default_estimate_and_invalid_mode():
    model = memories().double()
    n = sum(p.numel() for p in row_sparse_tables(model))
    rows = sum(p.shape[0] for p in row_sparse_tables(model))
    assert value_table_memory(model)["total_bytes"] == 32 * n + rows + 8
    for value_state in ("bf16", "int8"):
        with pytest.raises(ValueError, match="fp32"):
            value_table_memory(model, value_state)
    with pytest.raises(ValueError, match="value_state"):
        build_optimizer(model, 1e-3, 2e-3, 0.1, value_state="fp16")
    with pytest.raises(ValueError, match="value_state"):
        value_table_memory(model, "fp16")
