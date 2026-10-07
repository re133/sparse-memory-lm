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


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


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
        if impl == "triton" and not (_pow2(n_keys) and n_keys <= 65536 and _pow2(knn)):
            # the selection kernel (kernels.pk_select) packs key indices into 16 bits and works on whole
            # power-of-two blocks
            raise ValueError(f"mem_impl='triton' needs n_keys and knn to be powers of two and n_keys <= 65536, "
                             f"got n_keys={n_keys}, knn={knn}. Use mem_impl='torch' for other sizes.")
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

    def subkey_scores(self, query):
        """query: (N, heads, k_dim) -> scores against both sub-key sets, each (N, heads, n_keys)."""
        half = self.k_dim // 2
        k1, k2 = self.keys[:, 0], self.keys[:, 1]                      # (heads, n_keys, half)
        s1 = torch.einsum("nhd,hkd->nhk", query[..., :half], k1)
        s2 = torch.einsum("nhd,hkd->nhk", query[..., half:], k2)
        return s1, s2

    def get_indices(self, query):
        """query: (N, heads, k_dim) -> scores, indices: (N, heads, knn), exact top-k over all n_keys^2 keys."""
        N = query.shape[0]
        knn, n = self.knn, self.n_keys
        s1, s2 = self.subkey_scores(query)                              # (N, heads, n_keys)
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
        if getattr(w, "host_values", False):
            return self.values.bag(indices, weights, self.impl)
        if self.value_grad == "row_sparse":
            return row_sparse_embedding_bag(indices, weights, w, self.impl)
        if self.value_impl == "embedding_bag":
            return F.embedding_bag(indices, w, per_sample_weights=weights.to(w.dtype), mode="sum")
        # reference path: materialises (N, heads*knn, v_dim)
        return torch.einsum("nj,njd->nd", weights.to(w.dtype), F.embedding(indices, w))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.d_in)
        if (getattr(self, "decode_graph", False) and x.shape[0] == 1 and not self.record
                and not getattr(self.values.weight, "host_values", False)
                and not torch.is_grad_enabled() and x.is_cuda):
            return self._graph_forward(x).view(*shape[:-1], self.d_out)
        return self._forward(x).view(*shape[:-1], self.d_out)

    def _graph_forward(self, x):
        """Decoding (one token, no grad): replay a CUDA/HIP graph of _forward captured for this shape. Same
        kernels as the eager path, so the result is bit-identical; it only removes the per-op launch overhead
        (~20 PyTorch ops and Triton launches cost ~0.39 ms of CPU time per call, the GPU work is ~0.05 ms)."""
        ac = torch.is_autocast_enabled("cuda")
        key = (tuple(x.shape), x.dtype, ac, torch.get_autocast_dtype("cuda") if ac else None,
               id(getattr(self.values, "infer_table", None)))
        g = getattr(self, "_graph", None)
        if g is None or g[0] != key:
            static_x = x.clone()
            ctx = (torch.autocast("cuda", dtype=key[3], cache_enabled=False) if ac
                   else torch.autocast("cuda", enabled=False))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), ctx:
                for _ in range(2):                                      # warm-up (Triton compilation etc.)
                    self._forward(static_x)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), ctx:
                static_out = self._forward(static_x)
            g = self._graph = (key, graph, static_x, static_out)
        _, graph, static_x, static_out = g
        static_x.copy_(x)
        graph.replay()
        return static_out.clone()

    def _forward(self, x):
        N = x.shape[0]
        q = self.query_proj(x)
        if self.query_norm is not None:
            q = self.query_norm(q.to(self.query_norm.weight.dtype))   # BN in fp32 under autocast
        q = q.view(N, self.heads, self.k_dim)
        s1 = s2 = None
        autocast_bf16 = (x.is_cuda and torch.is_autocast_enabled("cuda")
                         and torch.get_autocast_dtype("cuda") == torch.bfloat16)
        if self.impl == "triton" and self.log_score_scale is None:
            # cast once instead of per half (autocast would cast each slice the same way): same bits
            s1, s2 = self.subkey_scores(q.to(torch.bfloat16) if autocast_bf16 else q)
        if s1 is not None and s1.dtype == torch.bfloat16:
            # kernel 2: both half top-k, cartesian top-k and softmax in one Triton kernel (smlm/kernels.py)
            from .kernels import PKSelect
            scores, indices, weights = PKSelect.apply(s1, s2, self.knn)
        else:
            # reference (also the fallback for fp32 scores without autocast and for the v2b score scale)
            scores, indices = self.get_indices(q)
            logits = scores.float()
            if self.log_score_scale is not None:
                logits = logits * self.log_score_scale.float().exp().view(1, self.heads, 1)
            weights = F.softmax(logits, dim=-1)                         # softmax over each head's knn
        if self.record:
            self.last_indices = indices.detach()
            self.last_scores = scores.detach()
            self.last_weights = weights.detach()
        if (self.impl == "triton" and not torch.is_grad_enabled() and self.swilu and autocast_bf16
                and self.values.weight.is_cuda):
            # kernel 4 (inference): value bag on the inference table (fp32 / bf16 / 4 bit, see
            # Transformer.set_memory_inference_table) with the swilu product and the bf16 cast fused in
            from .kernels import bag_infer
            pre = self.swilu_proj(x)                                    # bf16 under autocast
            t = getattr(self.values, "infer_table", None) or (self.values.weight, None)
            if hasattr(t, "bag"):                                       # table outside the GPU (smlm/offload.py)
                out = t.bag(indices.view(N, -1), weights.view(N, -1), pre)
            else:
                out = bag_infer(indices.view(N, -1), weights.view(N, -1), t[0], t[1], pre=pre, out_bf16=True)
            return self.value_proj(out)
        out = self.read_values(indices.view(N, -1), weights.view(N, -1))
        if self.swilu:
            out = self.value_proj(out * F.silu(self.swilu_proj(x)).to(out.dtype))
        return out

    def macs_per_token(self):
        """Multiply-accumulates per token (analytic): query, sub-key scoring, value bag, swilu."""
        m = self.d_in * self.heads * self.k_dim                         # query projection
        m += self.heads * 2 * self.n_keys * (self.k_dim // 2)          # scores against both sub-key sets
        m += self.heads * self.knn * self.v_dim                         # weighted sum of read values
        if self.swilu:
            m += self.d_in * self.v_dim + self.v_dim * self.d_out + self.v_dim
        return m
