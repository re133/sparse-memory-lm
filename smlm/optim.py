"""AdamW with three parameter groups and a warmup + cosine schedule shared by all groups.

Memory values follow the papers: Lample et al. 2019 learn them with a higher Adam learning rate of
1e-3 ("since the memory values are learned with sparse updates"); the Meta reference sets
value_fixed_lr=0.001 and scales it with the same LambdaLR multiplier as the rest. We do the same,
without weight decay on the values (XLM's separate value optimizer had none; decay would also shrink
rarely-read rows every step) and with a separate gradient-norm clip (Meta reference, train.py).
Parameters tagged `no_weight_decay` (v2a: product-key sub-keys) go to the no-decay group.
"""
import math

import torch

from .sparse_values import LazyRowAdam, OptimizerSet, clip_row_sparse, row_sparse_tables


def build_optimizer(model, lr, value_lr, weight_decay, betas=(0.9, 0.95), eps=1e-8, eng_value_lr=None,
                    value_state="fp32"):
    """AdamW for everything; value tables with row-sparse gradients get a LazyRowAdam instead
    (same lr / betas / eps, no weight decay) and the two are returned as one OptimizerSet.
    value_state controls moment storage for both PKM and Engram row-sparse tables.
    """
    if value_state not in ("fp32", "bf16", "int8"):
        raise ValueError(f"unknown value_state: {value_state}")
    sparse_tables = row_sparse_tables(model)
    sparse_ids = {id(t) for t in sparse_tables}
    decay, no_decay, values = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in sparse_ids:
            continue
        if getattr(p, "pk_value_param", False):
            values.append(p)
        elif p.dim() >= 2 and not getattr(p, "no_weight_decay", False):
            decay.append(p)
        else:
            no_decay.append(p)
    if values and value_state != "fp32":
        raise ValueError("value_state requires row_sparse gradients for every trainable value table")
    groups = [
        {"params": decay, "weight_decay": weight_decay, "base_lr": lr, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "base_lr": lr, "name": "no_decay"},
    ]
    if values:
        groups.append({"params": values, "weight_decay": 0.0, "base_lr": value_lr, "name": "memory_values"})
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, betas=betas, eps=eps, fused=True)
    if sparse_tables:
        # one lazy Adam for the product-key table(s), one for the Engram tables (own learning rate eng_value_lr,
        # default value_lr); clipping stays one global norm over all of them (optim.clip_grads)
        owners = {id(m.values.weight): m for m in model.modules() if getattr(m, "value_grad", None) == "row_sparse"}
        lazy = []
        for engram in (False, True):
            tabs = [t for t in sparse_tables if getattr(owners[id(t)], "is_engram", False) == engram]
            if not tabs:
                continue
            impl = "triton" if any(getattr(owners[id(t)], "impl", "torch") == "triton" for t in tabs) else "torch"
            tlr = (eng_value_lr or value_lr) if engram else value_lr
            lazy.append(LazyRowAdam(tabs, lr=tlr, betas=betas, eps=eps, impl=impl, state_dtype=value_state,
                                    name="engram_values" if engram else "memory_values"))
        return OptimizerSet(opt, *lazy)
    return opt


def value_table_memory(model, value_state="fp32"):
    """Expected persistent table tensors after optimizer initialization, excluding temporary workspace.

    Count shared parameters once and frozen tables as weights only. Dense gradients count at their peak
    allocation (zero_grad(set_to_none=True) releases them); row accumulators persist between steps.
    """
    if value_state not in ("fp32", "bf16", "int8"):
        raise ValueError(f"unknown value_state: {value_state}")
    sparse_ids = {id(t) for t in row_sparse_tables(model)}
    out = dict(table_bytes=0, gradient_bytes=0, moment_bytes=0, scale_bytes=0, touched_bytes=0, step_bytes=0)
    for p in model.parameters():
        if not getattr(p, "pk_value_param", False):
            continue
        n, size = p.numel(), p.element_size()
        out["table_bytes"] += n * size
        if not p.requires_grad:
            continue
        sparse = id(p) in sparse_ids
        if not sparse and value_state != "fp32":
            raise ValueError("value_state requires row_sparse gradients for every trainable value table")
        if sparse and value_state != "fp32" and p.dtype != torch.float32:
            raise ValueError("compact value states require fp32 tables")
        out["gradient_bytes"] += n * (max(4, size) if sparse else size)
        if sparse and value_state == "bf16":
            out["moment_bytes"] += n * 4
        elif sparse and value_state == "int8":
            out["moment_bytes"] += n * 2
            out["scale_bytes"] += p.shape[0] * 8
        else:
            out["moment_bytes"] += n * 2 * size
        out["touched_bytes"] += p.shape[0] if sparse else 0
        out["step_bytes"] += 4
    out["total_bytes"] = sum(out.values())
    return out


def lr_multiplier(step, total_steps, warmup_steps, min_ratio):
    """Linear warmup, then cosine decay to min_ratio (as lingua's lr_cosine with theta=1)."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    s = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * s))


def set_lr(opt, mult):
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * mult


def clip_grads(model, max_norm):
    """Clip memory values and the rest separately; return the (pre-clip) norm of the rest."""
    values = [p for p in model.parameters() if getattr(p, "pk_value_param", False) and p.grad is not None]
    rest = [p for p in model.parameters() if not getattr(p, "pk_value_param", False) and p.grad is not None]
    vnorm = torch.nn.utils.clip_grad_norm_(values, max_norm, foreach=True) if values else None
    sparse_tables = row_sparse_tables(model)
    if sparse_tables:
        vnorm = clip_row_sparse(sparse_tables, max_norm)
    norm = torch.nn.utils.clip_grad_norm_(rest, max_norm, foreach=True)
    return norm, vnorm
