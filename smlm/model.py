"""Llama-style decoder (RMSNorm, RoPE, SwiGLU, no biases) with optional product-key memory layers."""
import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .pkm import ProductKeyMemory


@dataclass
class ModelConfig:
    vocab_size: int = 50304          # GPT-2 BPE (50257) padded to a multiple of 64
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    ffn_hidden: int = 1024
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    # product-key memory: 0-indexed layers whose FFN is replaced by a memory layer
    mem_layers: list = field(default_factory=list)
    mem_n_keys: int = 512            # memory size = mem_n_keys ** 2
    mem_heads: int = 4
    mem_knn: int = 32
    mem_k_dim: int = 256
    mem_v_dim: int = -1              # -1: d_model
    mem_query_norm: str = "batchnorm"
    mem_swilu: bool = True
    mem_value_impl: str = "embedding_bag"

    def to_dict(self):
        return asdict(self)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)                     # (T, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x, pos0=0):
        # x: (B, H, T, D); rotate pairs (x[..., :D/2], x[..., D/2:])
        T = x.shape[-2]
        cos = self.cos[pos0:pos0 + T].to(x.dtype)
        sin = self.sin[pos0:pos0 + T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x, kv_cache=None, pos0=0):
        B, T, C = x.shape
        q, k, v = self.wqkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, pos0), self.rope(k, pos0)
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        # causal mask only needed when queries cover the whole key range (training / prefill)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and k.shape[2] == T))
        return self.wo(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w13 = nn.Linear(d, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_id: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.is_memory = layer_id in cfg.mem_layers
        if self.is_memory:
            self.ffn = ProductKeyMemory(
                cfg.d_model, cfg.d_model, n_keys=cfg.mem_n_keys, heads=cfg.mem_heads,
                knn=cfg.mem_knn, k_dim=cfg.mem_k_dim, v_dim=cfg.mem_v_dim,
                query_norm=cfg.mem_query_norm, swilu=cfg.mem_swilu, value_impl=cfg.mem_value_impl)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x, kv_cache=None, pos0=0):
        x = x + self.attn(self.attn_norm(x), kv_cache, pos0)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # tied input / output embeddings
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        std = self.cfg.init_std
        nn.init.normal_(self.tok_emb.weight, std=std)
        for blk in self.layers:
            nn.init.normal_(blk.attn.wqkv.weight, std=std)
            nn.init.normal_(blk.attn.wo.weight, std=std / math.sqrt(2 * self.cfg.n_layers))
            if not blk.is_memory:
                nn.init.normal_(blk.ffn.w13.weight, std=std)
                nn.init.normal_(blk.ffn.w2.weight, std=std / math.sqrt(2 * self.cfg.n_layers))
            # memory layers keep the initialisation of the Meta reference (ProductKeyMemory.reset_parameters)

    def memory_layers(self):
        return [b.ffn for b in self.layers if b.is_memory]

    def forward(self, idx, targets=None, kv_caches=None, pos0=0):
        x = self.tok_emb(idx)
        for i, blk in enumerate(self.layers):
            x = blk(x, None if kv_caches is None else kv_caches[i], pos0)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---- bookkeeping -------------------------------------------------------------------------
    def param_counts(self):
        """Parameter counts. 'embedding' = tied token embedding / LM head matrix;
        'memory_values' = product-key value table; 'dense_body' = everything else."""
        emb = self.tok_emb.weight.numel()
        values = sum(m.values.weight.numel() for m in self.memory_layers())
        total = sum(p.numel() for p in self.parameters())
        buffers = sum(b.numel() for n, b in self.named_buffers() if "running" in n)
        return {
            "total": total,
            "embedding": emb,
            "non_embedding": total - emb,
            "memory_values": values,
            "dense_body": total - emb - values,
            # active per token (non-embedding): dense body + the heads*knn value rows actually read
            "active_non_embedding_per_token": total - emb - values + sum(
                m.heads * m.knn * m.v_dim for m in self.memory_layers()),
            "bn_running_stats": buffers,
        }

    def macs_per_token(self, context_len=None):
        """Analytic forward multiply-accumulates per token (training FLOPs ~ 6x this for matmul parts).
        Attention score/value MACs use the average causal context (T/2)."""
        c = self.cfg
        T = context_len or c.max_seq_len
        d = c.d_model
        out = {"attn_proj": 0, "attn_scores": 0, "ffn": 0, "memory": 0, "lm_head": d * c.vocab_size}
        for blk in self.layers:
            out["attn_proj"] += 4 * d * d
            out["attn_scores"] += 2 * d * (T / 2)
            if blk.is_memory:
                out["memory"] += blk.ffn.macs_per_token()
            else:
                out["ffn"] += 3 * d * c.ffn_hidden
        out["total"] = sum(out.values())
        return out
