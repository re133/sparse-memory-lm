"""Engram-style n-gram memory, small version for the comparison with the product-key memory.

After Cheng et al. 2026, "Conditional Memory via Scalable Lookup" (Engram), section 2:
  * token compression: every GPT-2 token is mapped to the id of its NFKC-normalised, lower-cased string
  * for every position t and n-gram order n (2 and 3), K hash heads each pick one row of their own table (prime
    size) from the last n compressed tokens with a multiplicative-XOR hash; the rows are concatenated (eq. 1-2)
  * context-aware gate: k = W_K e, v = W_V e, alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(d)), v~ = alpha v (eq. 3-4)
  * Y = SiLU(Conv1D(RMSNorm(V~))) + V~, depthwise causal conv, kernel 4, dilation = max order, zero-initialised (eq. 5)
  * the module is added to the residual stream before attention (H <- H + Y); the FFN stays
Rows are read and trained like the product-key table: row-sparse gradients, lazy Adam (smlm/sparse_values.py).
The paper uses plain Adam for the tables; lazy Adam is used here so both memories are trained the same way.
"""
import math
import unicodedata

import torch
import torch.nn.functional as F
from torch import nn

from .sparse_values import row_sparse_embedding_bag


def canonical_ids(vocab_size):
    """GPT-2 token id -> id of its NFKC-normalised, lower-cased string. Tokens that aren't valid UTF-8 on their own
    (byte pieces) keep their bytes as key, ids past the tokenizer (vocabulary padding) keep their own id.
    Returns (tensor of vocab_size ids, number of distinct ids)."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    keys, ids = {}, []
    for i in range(vocab_size):
        if i < enc.n_vocab:
            raw = enc.decode_single_token_bytes(i)
            try:
                key = ("s", unicodedata.normalize("NFKC", raw.decode("utf-8")).lower())
            except UnicodeDecodeError:
                key = ("b", raw)
        else:
            key = ("pad", i)
        ids.append(keys.setdefault(key, len(keys)))
    return torch.tensor(ids, dtype=torch.int64), len(keys)


class NgramHash(nn.Module):
    """Row indices of all hash heads for every position. One set of indices for all Engram modules of the model;
    each module has its own tables."""

    def __init__(self, vocab_size, orders=(2, 3), heads=8, rows=524287, seed=0):
        super().__init__()
        canon, n_canon = canonical_ids(vocab_size)
        self.register_buffer("canon", canon, persistent=False)
        self.pad = n_canon                                  # compressed id for positions before the sequence
        self.orders, self.heads, self.rows = list(orders), heads, rows
        self.history = max(orders) - 1
        g = torch.Generator().manual_seed(seed)
        # odd multipliers below 2^31: id (< 2^16) x multiplier stays below 2^47, so nothing overflows int64
        mult = torch.randint(0, 2 ** 30, (len(self.orders), heads, max(orders)), generator=g) * 2 + 1
        self.register_buffer("mult", mult, persistent=False)

    @property
    def n_lookups(self):
        return len(self.orders) * self.heads

    def forward(self, c):
        """c: (B, history + T) compressed ids, the first `history` ones from before the chunk (or pad) ->
        (B, T, n_lookups) row indices, head h of order o in rows [(o * heads + h) * rows, ... + rows)."""
        T = c.shape[1] - self.history
        out = []
        for oi, n in enumerate(self.orders):
            h = 0
            for i in range(n):                              # token t - i
                tok = c[:, self.history - i:self.history - i + T]
                h = h ^ (tok[:, :, None] * self.mult[oi, :, i])
            h = h ^ (h >> 17)
            out.append(h % self.rows + (oi * self.heads + torch.arange(self.heads, device=c.device)) * self.rows)
        return torch.cat(out, -1)


class EngramMemory(nn.Module):
    """One Engram module: looks up n_lookups rows of head_dim, gates them with the hidden state, short conv."""

    def __init__(self, d_model, n_lookups, head_dim, rows_per_head, conv_kernel=4, dilation=3, eps=1e-5,
                 impl="triton", value_grad="row_sparse"):
        super().__init__()
        self.d_model, self.n_lookups, self.head_dim = d_model, n_lookups, head_dim
        self.d_mem = n_lookups * head_dim
        self.values = nn.Embedding(n_lookups * rows_per_head, head_dim)
        self.values.weight.pk_value_param = True            # own learning rate, no weight decay (smlm/optim.py)
        # "row_sparse": trained like the product-key table (row-sparse gradients, lazy Adam); "dense": an ordinary
        # embedding gradient and Adam on every row, as in the paper
        assert value_grad in ("row_sparse", "dense")
        self.value_grad = value_grad
        self.is_engram = True                               # own lazy Adam / learning rate (smlm/optim.py)
        assert impl in ("torch", "triton")
        self.impl = impl
        self.w_k = nn.Linear(self.d_mem, d_model, bias=False)
        self.w_v = nn.Linear(self.d_mem, d_model, bias=False)
        self.h_norm = nn.RMSNorm(d_model, eps=eps)
        self.k_norm = nn.RMSNorm(d_model, eps=eps)
        self.v_norm = nn.RMSNorm(d_model, eps=eps)
        self.conv = nn.Conv1d(d_model, d_model, conv_kernel, dilation=dilation, groups=d_model, bias=False)
        self.past = (conv_kernel - 1) * dilation            # earlier positions the conv sees
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std=0.02):
        nn.init.normal_(self.values.weight, std=self.d_mem ** -0.5)
        nn.init.normal_(self.w_k.weight, std=std)
        nn.init.normal_(self.w_v.weight, std=std)
        nn.init.zeros_(self.conv.weight)                    # Y = V~ at the start (paper: identity of the conv branch)

    def forward(self, h, rows, cache=None):
        """h: (B, T, d) residual stream, rows: (B, T, n_lookups) from NgramHash -> Y (B, T, d) to add to h."""
        B, T, _ = h.shape
        flat = rows.reshape(-1, 1)
        if self.value_grad == "dense":
            e = F.embedding(flat.squeeze(1), self.values.weight)
        else:
            e = row_sparse_embedding_bag(flat, torch.ones(flat.shape, device=h.device), self.values.weight, self.impl)
        e = e.view(B, T, self.d_mem)
        k, v = self.w_k(e), self.w_v(e)
        alpha = torch.sigmoid((self.h_norm(h) * self.k_norm(k)).sum(-1, keepdim=True) / math.sqrt(self.d_model))
        vt = alpha * v
        u = self.v_norm(vt)
        past = cache.get("eng_u") if cache is not None else None
        if past is None:
            past = torch.zeros(B, self.past, self.d_model, device=u.device, dtype=u.dtype)
        full = torch.cat([past.to(u.dtype), u], 1)
        if cache is not None:
            cache["eng_u"] = full[:, -self.past:]
        y = self.conv(full.transpose(1, 2)).transpose(1, 2)    # no padding: output position j sees full[j .. j+past]
        return F.silu(y) + vt

    def macs_per_token(self):
        return 2 * self.d_mem * self.d_model + 2 * self.d_model + self.conv.kernel_size[0] * self.d_model
