"""Llama-style decoder (RMSNorm, RoPE, SwiGLU, no biases) with optional product-key memory layers."""
import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .engram import EngramMemory, NgramHash
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
    # v2 "sharpness" switches (defaults = stage-1 / v1 behaviour)
    mem_keys_weight_decay: bool = True   # v2a: False -> no weight decay on the sub-keys
    mem_score_scale: str = "none"        # v2b: "learned" -> per-head learnable scale on top-k scores
    mem_score_scale_init: float = 1.0
    # stage 1b: one value table shared by all memory layers (keys / query / BN / swilu stay per layer)
    mem_share_values: bool = False
    # Hampter: "row_sparse" = gradients only for read rows + lazy Adam on the table (smlm/sparse_values.py)
    mem_value_grad: str = "dense"
    # "torch": PyTorch reference; "triton": kernels of smlm/kernels.py (same results up to rounding / ties)
    mem_impl: str = "torch"
    # Engram-style n-gram memory (smlm/engram.py): 0-indexed layers that get a module added before attention
    eng_layers: list = field(default_factory=list)
    eng_orders: list = field(default_factory=lambda: [2, 3])
    eng_heads: int = 8               # hash heads per n-gram order
    eng_head_dim: int = 24           # width of one row; one module reads len(orders) * heads rows per token
    eng_rows: int = 524287           # rows per head table (a prime, 2^19 - 1)
    eng_conv_kernel: int = 4
    eng_hash_seed: int = 0           # fixed hash functions, independent of the init seed
    eng_impl: str = "triton"         # row lookup / gradient / lazy Adam: "torch" reference or the Triton kernels
    eng_value_grad: str = "row_sparse"   # "dense": plain embedding gradient + Adam on all rows (the paper's setup)

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
        S = k.shape[2]
        if T > 1 and S > T:
            # several new tokens after a filled cache: token i may see the cache and new tokens 0..i
            mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(S - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # training / prefill (queries cover all keys) or a single new token (sees everything)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
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
    def __init__(self, cfg: ModelConfig, layer_id: int, shared_values=None):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.is_memory = layer_id in cfg.mem_layers
        if self.is_memory:
            self.ffn = ProductKeyMemory(
                cfg.d_model, cfg.d_model, n_keys=cfg.mem_n_keys, heads=cfg.mem_heads,
                knn=cfg.mem_knn, k_dim=cfg.mem_k_dim, v_dim=cfg.mem_v_dim,
                query_norm=cfg.mem_query_norm, swilu=cfg.mem_swilu, value_impl=cfg.mem_value_impl,
                keys_weight_decay=cfg.mem_keys_weight_decay, score_scale=cfg.mem_score_scale,
                score_scale_init=cfg.mem_score_scale_init, shared_values=shared_values,
                value_grad=cfg.mem_value_grad, impl=cfg.mem_impl)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.engram = None
        if layer_id in cfg.eng_layers:
            self.engram = EngramMemory(cfg.d_model, len(cfg.eng_orders) * cfg.eng_heads, cfg.eng_head_dim,
                                       cfg.eng_rows, conv_kernel=cfg.eng_conv_kernel, dilation=max(cfg.eng_orders),
                                       eps=cfg.norm_eps, impl=cfg.eng_impl, value_grad=cfg.eng_value_grad)

    def forward(self, x, kv_cache=None, pos0=0, eng_rows=None):
        if self.engram is not None:
            x = x + self.engram(x, eng_rows, kv_cache)
        x = x + self.attn(self.attn_norm(x), kv_cache, pos0)
        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        shared = None
        if cfg.mem_share_values and cfg.mem_layers:
            v_dim = cfg.mem_v_dim if cfg.mem_v_dim > 0 else cfg.d_model
            shared = nn.Embedding(cfg.mem_n_keys ** 2, v_dim)          # initialised by the memory layers
        self.layers = nn.ModuleList([Block(cfg, i, shared) for i in range(cfg.n_layers)])
        self.ngram = (NgramHash(cfg.vocab_size, cfg.eng_orders, cfg.eng_heads, cfg.eng_rows, cfg.eng_hash_seed)
                      if cfg.eng_layers else None)
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

    def engram_layers(self):
        return [b.engram for b in self.layers if b.engram is not None]

    def set_memory_decode_graphs(self, enabled=True):
        """Single-token no-grad forwards of the memory layers replay a captured CUDA/HIP graph
        (ProductKeyMemory._graph_forward): same kernels, no per-op launch overhead."""
        for m in self.memory_layers():
            m.decode_graph = enabled
            m._graph = None

    @torch.no_grad()
    def set_memory_inference_table(self, kind="fp32"):
        """Inference copy of the value table(s) for mem_impl="triton" (kernel 4): "fp32" (the weights
        themselves), "bf16" or "q4" (4 bit, one fp16 scale per row, see smlm/kernels.py). Only used in no-grad
        forwards under bf16 autocast; call again (or with "fp32") after the weights change."""
        from .kernels import quantize_q4
        seen = set()
        for m in self.memory_layers():
            m._graph = None                                 # captured decode graphs point at the old table
            v = m.values
            if id(v) in seen:
                continue
            seen.add(id(v))
            if kind == "fp32":
                v.infer_table = None
            elif kind == "bf16":
                v.infer_table = (v.weight.detach().to(torch.bfloat16), None)
            elif kind == "q4":
                v.infer_table = quantize_q4(v.weight.detach())
            else:
                raise ValueError(kind)

    def forward(self, idx, targets=None, kv_caches=None, pos0=0):
        x = self.tok_emb(idx)
        rows = None
        if self.ngram is not None:
            # n-gram rows need the last tokens before this chunk: pad at the start, else from the cache
            c = self.ngram.canon[idx]
            hist = kv_caches[0].get("eng_hist") if kv_caches is not None else None
            if hist is None:
                hist = torch.full((c.shape[0], self.ngram.history), self.ngram.pad, dtype=c.dtype, device=c.device)
            c = torch.cat([hist, c], 1)
            if kv_caches is not None:
                kv_caches[0]["eng_hist"] = c[:, -self.ngram.history:]
            rows = self.ngram(c)
        for i, blk in enumerate(self.layers):
            x = blk(x, None if kv_caches is None else kv_caches[i], pos0, rows)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---- bookkeeping -------------------------------------------------------------------------
    def param_counts(self):
        """Parameter counts. 'embedding' = tied token embedding / LM head matrix;
        'memory_values' = product-key value table and Engram tables; 'dense_body' = everything else."""
        emb = self.tok_emb.weight.numel()
        tables = {id(m.values.weight): m.values.weight.numel() for m in self.memory_layers() + self.engram_layers()}
        values = sum(tables.values())                                     # a shared table counts once
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
                m.heads * m.knn * m.v_dim for m in self.memory_layers()) + sum(
                m.n_lookups * m.head_dim for m in self.engram_layers()),
            "bn_running_stats": buffers,
        }

    def macs_per_token(self, context_len=None):
        """Analytic forward multiply-accumulates per token (training FLOPs ~ 6x this for matmul parts).
        Attention score/value MACs use the average causal context (T/2)."""
        c = self.cfg
        T = context_len or c.max_seq_len
        d = c.d_model
        out = {"attn_proj": 0, "attn_scores": 0, "ffn": 0, "memory": 0, "engram": 0, "lm_head": d * c.vocab_size}
        for blk in self.layers:
            if blk.engram is not None:
                out["engram"] += blk.engram.macs_per_token()
            out["attn_proj"] += 4 * d * d
            out["attn_scores"] += 2 * d * (T / 2)
            if blk.is_memory:
                out["memory"] += blk.ffn.macs_per_token()
            else:
                out["ffn"] += 3 * d * c.ffn_hidden
        out["total"] = sum(out.values())
        return out
