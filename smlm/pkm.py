"""Product-key memory layer (Lample et al. 2019; Berges et al. 2024 "Memory Layers at Scale").

Follows the Meta reference implementation (facebookresearch/memory, lingua/product_key/memory.py):
  * per-head query projection, separate sub-key sets per head (H x 2 x n_keys x k_dim/2)
  * exact top-k over the n_keys^2 product keys via two half searches + knn x knn cartesian step
  * softmax over the k selected scores, heads summed into one weighted bag of shared values
  * optional "swilu" output path (Memory+): out = W2 (mem(x) * silu(W1 x))
  * initialisation: keys U(+-1/sqrt(k_dim)), values N(0, v_dim^-0.5), query xavier_uniform
Additionally (Lample et al. 2019, sec. 4.5, and He 2024): optional BatchNorm on the query, which the
papers report as the main lever against unused keys for memories >= 147k slots.

v2 "sharpness" switches (stage-1 finding: the softmax over the top-k stayed almost flat, ~30.5 of 32
entries effectively mixed per head, with the score scale barely growing from its initialisation):
  * keys_weight_decay=False (v2a): the sub-keys are tagged `no_weight_decay`, so the optimizer does not
    pull their norm (and with it the score scale) towards zero.
  * score_scale="learned" (v2b): a learnable per-head scale s_h = exp(log_s_h) multiplies the k selected
    scores before the softmax: w = softmax(s_h * scores). s_h > 0 never changes which entries are
    selected, only how sharply they are weighted. Initialised to score_scale_init (1.0 = v1 behaviour).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .sparse_values import row_sparse_embedding_bag


class ProductKeyMemory(nn.Module):
    def __init__(self, d_in, d_out, n_keys=512, heads=4, knn=32, k_dim=256, v_dim=-1,
                 query_norm="batchnorm", swilu=True, value_impl="embedding_bag",
                 keys_weight_decay=True, score_scale="none", score_scale_init=1.0, shared_values=None,
                 value_grad="dense", impl="torch"):
        super().__init__()
        assert k_dim % 2 == 0 and knn <= n_keys
        self.d_in, self.d_out = d_in, d_out
        self.n_keys, self.heads, self.knn, self.k_dim = n_keys, heads, knn, k_dim
        self.size = n_keys ** 2
        self.v_dim = v_dim if v_dim > 0 else d_out
        self.swilu = swilu
        self.value_impl = value_impl
        assert value_grad in ("dense", "row_sparse")
        self.value_grad = value_grad                    # "row_sparse": see smlm/sparse_values.py
        assert impl in ("torch", "triton")
        self.impl = impl                                # "triton": kernels in smlm/kernels.py
        self.score_scale_init = score_scale_init

        self.keys = nn.Parameter(torch.empty(heads, 2, n_keys, k_dim // 2))
        if not keys_weight_decay:
            self.keys.no_weight_decay = True                              # read by optim.build_optimizer
        if score_scale == "learned":
            assert score_scale_init > 0
            self.log_score_scale = nn.Parameter(torch.empty(heads))       # per-head softmax sharpness
        elif score_scale == "none":
            self.log_score_scale = None
        else:
            raise ValueError(score_scale)
        self.query_proj = nn.Linear(d_in, heads * k_dim, bias=True)
        if query_norm == "batchnorm":
            self.query_norm = nn.BatchNorm1d(heads * k_dim)
        elif query_norm == "none":
            self.query_norm = None
        else:
            raise ValueError(query_norm)
        if shared_values is not None:
            # one value table shared by several memory layers (Meta "Memory+"); keys / query stay per layer.
            # Registering the same module in every layer is what the reference does; parameters() and the
            # optimizer see it once, state_dict lists it under each layer (one storage when saved).
            assert shared_values.weight.shape == (self.size, self.v_dim)
            self.values = shared_values
        else:
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
        self.last_weights = None
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
        if self.log_score_scale is not None:
            nn.init.constant_(self.log_score_scale, math.log(self.score_scale_init))
        # tag for the optimizer: own learning rate, no weight decay, separate grad clipping
        self.values.weight.pk_value_param = True

    def score_scale(self):
        """Per-head multiplier applied to the top-k scores before the softmax (ones if disabled)."""
        if self.log_score_scale is None:
            return torch.ones(self.heads, device=self.keys.device)
        return self.log_score_scale.exp()

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
        if self.value_grad == "row_sparse":
            return row_sparse_embedding_bag(indices, weights, w, self.impl)
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
        logits = scores.float()
        if self.log_score_scale is not None:
            logits = logits * self.log_score_scale.float().exp().view(1, self.heads, 1)
        weights = F.softmax(logits, dim=-1)                             # softmax over each head's knn
        if self.record:
            self.last_indices = indices.detach()
            self.last_scores = scores.detach()
            self.last_weights = weights.detach()
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
