"""Product-key memory layer (Lample et al. 2019; Berges et al. 2024 "Memory Layers at Scale").

Follows the Meta reference implementation (facebookresearch/memory, lingua/product_key/memory.py):
  * per-head query projection, separate sub-key sets per head (H x 2 x n_keys x k_dim/2)
  * exact top-k over the n_keys^2 product keys via two half searches + knn x knn cartesian step
  * softmax over the k selected scores, heads summed into one weighted bag of shared values
  * optional "swilu" output path (Memory+): out = W2 (mem(x) * silu(W1 x))
  * initialisation: keys U(+-1/sqrt(k_dim)), values N(0, v_dim^-0.5), query xavier_uniform
Additionally (Lample et al. 2019, sec. 4.5, and He 2024): optional BatchNorm on the query, which the
papers report as the main lever against unused keys for memories >= 147k slots.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class ProductKeyMemory(nn.Module):
    def __init__(self, d_in, d_out, n_keys=512, heads=4, knn=32, k_dim=256, v_dim=-1,
                 query_norm="batchnorm", swilu=True, value_impl="embedding_bag"):
        super().__init__()
        assert k_dim % 2 == 0 and knn <= n_keys
        self.d_in, self.d_out = d_in, d_out
        self.n_keys, self.heads, self.knn, self.k_dim = n_keys, heads, knn, k_dim
        self.size = n_keys ** 2
        self.v_dim = v_dim if v_dim > 0 else d_out
        self.swilu = swilu
        self.value_impl = value_impl

        self.keys = nn.Parameter(torch.empty(heads, 2, n_keys, k_dim // 2))
        self.query_proj = nn.Linear(d_in, heads * k_dim, bias=True)
        if query_norm == "batchnorm":
            self.query_norm = nn.BatchNorm1d(heads * k_dim)
        elif query_norm == "none":
            self.query_norm = None
        else:
            raise ValueError(query_norm)
        self.values = nn.Embedding(self.size, self.v_dim)
        if swilu:
            self.swilu_proj = nn.Linear(d_in, self.v_dim)
            self.value_proj = nn.Linear(self.v_dim, d_out)
        else:
            assert self.v_dim == d_out
        # filled on every forward while record=True (usage statistics / index samples)
        self.record = False
        self.last_indices = None
        self.last_scores = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, -bound, bound)
        nn.init.normal_(self.values.weight, mean=0.0, std=self.v_dim ** -0.5)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)
        if self.swilu:
            nn.init.normal_(self.swilu_proj.weight, mean=0.0, std=self.d_out ** -0.5)
            nn.init.normal_(self.value_proj.weight, mean=0.0, std=self.d_out ** -0.5)
            nn.init.zeros_(self.swilu_proj.bias)
            nn.init.zeros_(self.value_proj.bias)
        # tag for the optimizer: own learning rate, no weight decay, separate grad clipping
        self.values.weight.pk_value_param = True

    def get_indices(self, query):
        """query: (N, heads, k_dim) -> scores, indices: (N, heads, knn), exact top-k over all n_keys^2 keys."""
        N = query.shape[0]
        half = self.k_dim // 2
        knn, n = self.knn, self.n_keys
        q1, q2 = query[..., :half], query[..., half:]
        k1, k2 = self.keys[:, 0], self.keys[:, 1]                      # (heads, n_keys, half)
        s1 = torch.einsum("nhd,hkd->nhk", q1, k1)                       # (N, heads, n_keys)
        s2 = torch.einsum("nhd,hkd->nhk", q2, k2)
        s1, i1 = s1.topk(knn, dim=-1)                                   # (N, heads, knn)
        s2, i2 = s2.topk(knn, dim=-1)
        all_s = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).view(N, self.heads, knn * knn)
        all_i = (i1.unsqueeze(-1) * n + i2.unsqueeze(-2)).view(N, self.heads, knn * knn)
        scores, best = all_s.topk(knn, dim=-1)
        indices = all_i.gather(-1, best)
        return scores, indices

    def read_values(self, indices, weights):
        """indices, weights: (N, heads*knn) -> (N, v_dim) = sum_j weights[:, j] * values[indices[:, j]]"""
        w = self.values.weight
        if self.value_impl == "embedding_bag":
            return F.embedding_bag(indices, w, per_sample_weights=weights.to(w.dtype), mode="sum")
        # reference path: materialises (N, heads*knn, v_dim)
        return torch.einsum("nj,njd->nd", weights.to(w.dtype), F.embedding(indices, w))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.d_in)
        N = x.shape[0]
        q = self.query_proj(x)
        if self.query_norm is not None:
            q = self.query_norm(q.to(self.query_norm.weight.dtype))   # BN in fp32 under autocast
        q = q.view(N, self.heads, self.k_dim)
        scores, indices = self.get_indices(q)
        if self.record:
            self.last_indices = indices.detach()
            self.last_scores = scores.detach()
        weights = F.softmax(scores.float(), dim=-1)                     # softmax over each head's knn
        out = self.read_values(indices.view(N, -1), weights.view(N, -1))
        if self.swilu:
            out = self.value_proj(out * F.silu(self.swilu_proj(x)).to(out.dtype))
        return out.view(*shape[:-1], self.d_out)

    def macs_per_token(self):
        """Multiply-accumulates per token (analytic): query, sub-key scoring, value bag, swilu."""
        m = self.d_in * self.heads * self.k_dim                         # query projection
        m += self.heads * 2 * self.n_keys * (self.k_dim // 2)          # scores against both sub-key sets
        m += self.heads * self.knn * self.v_dim                         # weighted sum of read values
        if self.swilu:
            m += self.d_in * self.v_dim + self.v_dim * self.d_out + self.v_dim
        return m
