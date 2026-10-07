"""Optional Muon for hidden projections, with AdamW and lazy Adam for the remaining parameters.

Usage: python -m smlm.train --model A --optimizer muon --tokens 1e6 --out_dir runs/muon-check
Math: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
"""
import math

import torch
from torch import nn

from .engram import EngramMemory
from .model import Attention, SwiGLU
from .pkm import ProductKeyMemory
from .sparse_values import LazyRowAdam, OptimizerSet


def zeropower_via_newtonschulz5(G, steps=5):
    """Quintic approximate polar factor; tiny/zero singular values need not reach the usual band."""
    if G.ndim != 2 or not G.is_floating_point() or G.numel() == 0:
        raise ValueError("Newton-Schulz needs a nonempty real floating-point matrix")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Newton-Schulz steps must be a positive integer")
    # Explicit precision also holds when the caller is inside an autocast context.
    with torch.autocast(device_type=G.device.type, enabled=False):
        X = G.to(torch.bfloat16 if G.is_cuda else torch.float32)
        tall = G.shape[0] > G.shape[1]
        if tall:
            X = X.T
        X = X / (X.norm() + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X.T if tall else X


class Muon(torch.optim.Optimizer):
    """Nesterov SGD momentum followed by Newton-Schulz, with Moonshot's AdamW LR scaling.

    We keep the accumulated (not EMA) momentum convention of the Moonshot reference.
    Packed QKV and SwiGLU projections are each orthogonalised as one matrix.
    """

    def __init__(self, params, lr=6e-4, weight_decay=0.1, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in [defaults, *self.param_groups]:
            if not math.isfinite(group["lr"]) or group["lr"] < 0:
                raise ValueError("Muon learning rate must be finite and nonnegative")
            if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
                raise ValueError("Muon weight decay must be finite and nonnegative")
            if not 0 <= group["momentum"] < 1:
                raise ValueError("Muon momentum must be in [0, 1)")
            if not isinstance(group["ns_steps"], int) or group["ns_steps"] < 1:
                raise ValueError("Muon ns_steps must be a positive integer")
        for group in self.param_groups:
            group.setdefault("base_lr", group["lr"])
            for p in group["params"]:
                if p.ndim != 2 or not p.is_floating_point() or p.numel() == 0:
                    raise ValueError("Muon only accepts nonempty real floating-point matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("Muon requires dense gradients; use LazyRowAdam for sparse value tables")
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = zeropower_via_newtonschulz5(g.add(buf, alpha=group["momentum"]), group["ns_steps"])
                # The shape correction scales the update, never the weight decay.
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * 0.2 * math.sqrt(max(p.shape)))
        return loss


def muon_parameters(model):
    """Select projections by their owning module, so new gates/embeddings cannot silently enter Muon."""
    excluded = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    if getattr(model, "lm_head", None) is not None:
        excluded.add(id(model.lm_head.weight))
    selected = set()
    for m in model.modules():
        if isinstance(m, Attention):
            names = ("wqkv", "wo")
        elif isinstance(m, SwiGLU):
            names = ("w13", "w2")
        elif isinstance(m, ProductKeyMemory):
            # swilu_proj multiplies the readout as a gate; it stays on AdamW.
            names = ("query_proj", "value_proj")
        elif isinstance(m, EngramMemory):
            names = ("w_k", "w_v")
        else:
            continue
        for name in names:
            proj = getattr(m, name, None)
            if isinstance(proj, nn.Linear):
                p = proj.weight
                if p.requires_grad and p.ndim == 2 and id(p) not in excluded and not getattr(p, "pk_value_param", False):
                    selected.add(id(p))
    return [p for p in model.parameters() if id(p) in selected]


class MuonOptimizerSet(OptimizerSet):
    """Dense Muon/AdamW checkpointing; reject unsupported lazy-Adam checkpoints before changing state."""

    def _check_checkpointable(self):
        if any(isinstance(o, LazyRowAdam) for o in self.opts):
            raise NotImplementedError("Muon with LazyRowAdam cannot save/load optimizer state: "
                                      "LazyRowAdam does not support resuming; model weights can still be saved")

    def state_dict(self):
        self._check_checkpointable()
        return {"format": "smlm.muon.v1", "optimizers": [
            {"type": type(o).__name__, "state_dict": o.state_dict()} for o in self.opts]}

    def load_state_dict(self, state_dict):
        self._check_checkpointable()
        if state_dict.get("format") != "smlm.muon.v1":
            raise ValueError("Expected a MuonOptimizerSet state dict (smlm.muon.v1)")
        saved = state_dict.get("optimizers", [])
        if len(saved) != len(self.opts) or any(s.get("type") != type(o).__name__ for o, s in zip(self.opts, saved)):
            raise ValueError("MuonOptimizerSet checkpoint optimizer types do not match")
        for o, s in zip(self.opts, saved):
            groups = s["state_dict"]["param_groups"]
            if len(groups) != len(o.param_groups) or any(
                    len(a["params"]) != len(b["params"]) for a, b in zip(groups, o.param_groups)):
                raise ValueError("MuonOptimizerSet checkpoint parameter groups do not match")
        for o, s in zip(self.opts, saved):
            o.load_state_dict(s["state_dict"])


def build_muon_optimizer(model, lr, value_lr, weight_decay, betas, eps, eng_value_lr):
    from .optim import build_optimizer

    # Reuse today's routing of all exceptions and lazy tables, before any optimizer state exists.
    legacy = build_optimizer(model, lr, value_lr, weight_decay, betas, eps, eng_value_lr)
    opts = legacy.opts if isinstance(legacy, OptimizerSet) else [legacy]
    selected = {id(p) for p in muon_parameters(model)}
    groups = []
    for group in opts[0].param_groups:
        params = [p for p in group["params"] if id(p) in selected]
        if params:
            groups.append({"params": params, "lr": group["lr"], "base_lr": group["base_lr"],
                           "weight_decay": group["weight_decay"], "name": "muon_" + group["name"]})
            group["params"] = [p for p in group["params"] if id(p) not in selected]
    muon = Muon(groups, lr=lr, weight_decay=weight_decay) if groups else None
    opt = MuonOptimizerSet(muon, *opts)
    assigned = [id(p) for group in opt.param_groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(assigned) == len(set(assigned)), "A parameter was assigned to multiple optimizers"
    assert set(assigned) == expected, "Optimizer partition must cover every trainable parameter exactly once"
    return opt


def optimizer_description(opt):
    """Describe actual optimizer children and groups, including separate PKM/Engram table learning rates."""
    opts = opt.opts if isinstance(opt, OptimizerSet) else [opt]
    parts = []
    for child in opts:
        if isinstance(child, Muon):
            algorithm = ("Muon Nesterov (accumulated momentum), quintic Newton-Schulz "
                         "coefficients=(3.4445,-4.7750,2.0315), norm_eps=1e-7, GPU bf16 / CPU fp32, "
                         "update_scale=0.2*sqrt(max(rows,cols)), decoupled weight decay")
        elif isinstance(child, LazyRowAdam):
            algorithm = f"LazyRowAdam({child.impl}), only read rows updated"
        else:
            algorithm = "AdamW(fused)"
        groups = []
        for group in child.param_groups:
            detail = f"{group['name']}: base_lr={group['base_lr']:g}, wd={group['weight_decay']:g}"
            if isinstance(child, Muon):
                detail += f", momentum={group['momentum']:g}, steps={group['ns_steps']}"
            if "betas" in group:
                detail += f", betas={group['betas']}, eps={group['eps']:g}"
            groups.append(detail)
        parts.append(algorithm + " [" + "; ".join(groups) + "]")
    return "; ".join(parts) + "; memory values clipped separately"
