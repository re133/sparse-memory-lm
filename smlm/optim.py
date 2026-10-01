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


def build_optimizer(model, lr, value_lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
    decay, no_decay, values = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if getattr(p, "pk_value_param", False):
            values.append(p)
        elif p.dim() >= 2 and not getattr(p, "no_weight_decay", False):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "base_lr": lr, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "base_lr": lr, "name": "no_decay"},
    ]
    if values:
        groups.append({"params": values, "weight_decay": 0.0, "base_lr": value_lr, "name": "memory_values"})
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, betas=betas, eps=eps, fused=True)
    return opt


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
    norm = torch.nn.utils.clip_grad_norm_(rest, max_norm, foreach=True)
    return norm, vnorm
