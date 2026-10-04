"""Step 3: product-key memory as an add-on to a frozen pretrained model (Qwen3.5-0.8B via HF transformers).

After each chosen decoder layer i the hidden state gets an extra residual block

    h <- h + g_i * M_i(norm(h))            g_i: learnable scalar gate, initialised to 0

so with all g_i = 0 the model computes exactly what the original model computes (tests/test_qwen_memory.py checks
bit-identical logits). The original weights stay frozen; trainable are only the add-on blocks and the gates.

kind="memory": M_i is a ProductKeyMemory as in B: one value table shared by all add-on blocks; keys, query
               projection + BatchNorm and the swilu output path (Memory+) per block; row-sparse table gradients
               with lazy Adam (smlm/sparse_values.py).
kind="dense":  control run: M_i is a SwiGLU block with the same multiply-accumulates per token as the memory block
               (hidden 1408 for d = 1024 and the default memory shape), same positions, same training.
norm is a parameter-free RMSNorm (no extra trainable vector).

Add-ons are attached with forward hooks, so the original module tree (and HF generation / caching) is unchanged.
"""
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .model import SwiGLU
from .pkm import ProductKeyMemory


@dataclass
class AddOnConfig:
    kind: str = "memory"                       # "memory" | "dense"
    layers: list = field(default_factory=lambda: [5, 11, 17])   # 0-indexed decoder layers (after block 6/12/18)
    n_keys: int = 1024                         # table = n_keys^2 rows
    heads: int = 4
    knn: int = 32
    k_dim: int = 256
    v_dim: int = -1                            # -1: hidden size of the base model
    swilu: bool = True
    impl: str = "triton"                       # "torch" for CPU tests
    dense_hidden: int = -1                     # -1: same MACs per token as the memory block (rounded to 64)
    norm_eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


def memory_macs(d, c: AddOnConfig):
    v = c.v_dim if c.v_dim > 0 else d
    macs = d * c.heads * c.k_dim + c.heads * 2 * c.n_keys * (c.k_dim // 2) + c.heads * c.knn * v
    if c.swilu:
        macs += 2 * d * v
    return macs


class AddOn(nn.Module):
    def __init__(self, d, cfg: AddOnConfig, shared_values=None):
        super().__init__()
        self.eps = cfg.norm_eps
        self.gate = nn.Parameter(torch.zeros(()))
        self.gate.no_weight_decay = True
        if cfg.kind == "memory":
            self.body = ProductKeyMemory(d, d, n_keys=cfg.n_keys, heads=cfg.heads, knn=cfg.knn, k_dim=cfg.k_dim,
                                         v_dim=cfg.v_dim, query_norm="batchnorm", swilu=cfg.swilu,
                                         shared_values=shared_values, value_grad="row_sparse", impl=cfg.impl)
        elif cfg.kind == "dense":
            hidden = cfg.dense_hidden if cfg.dense_hidden > 0 else max(64, round(memory_macs(d, cfg) / (3 * d) / 64) * 64)
            self.body = SwiGLU(d, hidden)
        else:
            raise ValueError(cfg.kind)

    def forward(self, h):
        x = F.rms_norm(h, (h.shape[-1],), eps=self.eps)
        y = self.body(x)
        return h + self.gate.to(h.dtype) * y.to(h.dtype)


def decoder_layers(model):
    inner = model.model if hasattr(model, "model") else model
    inner = getattr(inner, "language_model", inner)
    return inner.layers


def attach(model, cfg: AddOnConfig):
    """Freeze `model`, create the add-on blocks (model.addons) and hook them behind cfg.layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.get_text_config().hidden_size
    shared = None
    if cfg.kind == "memory":
        v = cfg.v_dim if cfg.v_dim > 0 else d
        shared = nn.Embedding(cfg.n_keys ** 2, v)
    layers = decoder_layers(model)
    dev = next(model.parameters()).device
    addons = nn.ModuleList([AddOn(d, cfg, shared) for _ in cfg.layers]).to(dev)
    model.addons = addons
    model.addon_cfg = cfg
    handles = []
    for i, a in zip(cfg.layers, addons):
        def hook(mod, args, out, a=a):
            if isinstance(out, tuple):
                return (a(out[0]),) + tuple(out[1:])
            return a(out)
        handles.append(layers[i].register_forward_hook(hook))
    model._addon_handles = handles
    return addons


def trainable_parameters(model):
    return [p for p in model.addons.parameters() if p.requires_grad]


def addon_state_dict(model):
    return {"addon_cfg": model.addon_cfg.to_dict(), "state_dict": model.addons.state_dict()}
