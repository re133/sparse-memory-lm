"""Row-sparse gradients and a lazy Adam for the product-key value table (Hampter plan, step 1).

Dense path (stage 1 / 1b): every backward of the value lookup materialises a dense (rows x dim) gradient
(1.6 GB for 1M x 384 fp32), zero-fills it and adds it into .grad; AdamW then updates every row, so momentum
keeps moving rows that were not read.

Row-sparse path:
  * RowStore: one persistent fp32 accumulator (rows x dim) and a bool "touched" mask per table.
  * _RowSparseBag: forward = embedding_bag; backward writes w_k * grad_out[bag(k)] only into the rows that were
    read and returns the gradient for the per-sample weights (so keys / query keep learning). The table
    parameter itself never gets a .grad. Accumulation uses index_put_(accumulate=True), which sorts the
    indices and reduces per row; index_add_ uses atomics and is ~5x slower on this GPU because popular rows
    (e.g. sentence ends) are hit by many tokens of a micro-batch (measured 5.6 ms vs 28.8 ms per call).
  * LazyRowAdam: Adam (no weight decay) applied only to touched rows; untouched rows keep their values and
    their optimizer state exactly. Global step count for bias correction (as torch SparseAdam / TF LazyAdam).
    After the step the touched rows of the accumulator are zeroed and the mask is cleared.
"""
import math

import torch
import torch.nn.functional as F

CHUNK = 1 << 18          # lookups per accumulation chunk (262144 x 384 fp32 = 400 MB temporary)


class RowStore:
    def __init__(self, table):
        # at least fp32 (fp64 tables, e.g. in tests, keep fp64)
        dtype = torch.promote_types(table.dtype, torch.float32)
        self.acc = torch.zeros_like(table, dtype=dtype, memory_format=torch.contiguous_format)
        self.touched = torch.zeros(table.shape[0], dtype=torch.bool, device=table.device)
        self.grad_scale = 1.0           # set by clipping, applied in the optimizer step

    def reset(self, rows=None):
        if rows is None:
            self.acc.zero_()
            self.touched.zero_()
        else:
            self.acc.index_fill_(0, rows, 0.0)
            self.touched.index_fill_(0, rows, False)
        self.grad_scale = 1.0


def row_store(table):
    """The RowStore attached to a (possibly shared) table parameter; created on first use."""
    st = getattr(table, "row_store", None)
    if st is None or st.acc.device != table.device or st.acc.shape != table.shape:
        st = RowStore(table)
        table.row_store = st
    return st


class _RowSparseBag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, table, store, impl="torch"):
        n, j = indices.shape
        flat = indices.reshape(-1).contiguous()
        offsets = torch.arange(0, n * j, j, device=indices.device)
        w = weights.reshape(-1).to(table.dtype).contiguous()
        out, offset2bag, _, _ = torch.ops.aten._embedding_bag(table, flat, offsets, False, 0, False, w, False, -1)
        if offset2bag.numel() != flat.numel():          # CPU fp32 fast path leaves it empty (fixed-size bags)
            offset2bag = torch.arange(n, device=indices.device).repeat_interleave(j)
        ctx.save_for_backward(w, flat, offsets, offset2bag)
        ctx.table, ctx.store, ctx.wshape, ctx.wdtype = table, store, weights.shape, weights.dtype
        ctx.impl, ctx.ishape = impl, indices.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, flat, offsets, offset2bag = ctx.saved_tensors
        table, store = ctx.table, ctx.store
        g = grad_out.contiguous().to(store.acc.dtype)
        if ctx.impl == "triton":                       # kernel 1 (smlm/kernels.py)
            from .kernels import bag_backward_rows
            gw = bag_backward_rows(g, flat.view(ctx.ishape), w, table, store.acc, store.touched)
            ctx.table = ctx.store = None
            return gw.view(ctx.wshape).to(ctx.wdtype), None, None, None, None
        gw = torch.ops.aten._embedding_bag_per_sample_weights_backward(g.to(table.dtype), table, flat, offsets,
                                                                       offset2bag, 0, -1)
        for a in range(0, flat.numel(), CHUNK):
            contrib = g.index_select(0, offset2bag[a:a + CHUNK]).mul_(w[a:a + CHUNK, None].to(g.dtype))
            store.acc.index_put_((flat[a:a + CHUNK],), contrib, accumulate=True)
        store.touched[flat] = True
        # the graph node outlives backward while the last loss tensor is alive; drop the RowStore reference
        # so `del table.row_store` after training really frees the accumulator
        ctx.table = ctx.store = None
        return gw.view(ctx.wshape).to(ctx.wdtype), None, None, None, None


def row_sparse_embedding_bag(indices, weights, table, impl="torch"):
    """sum_j weights[:, j] * table[indices[:, j]] with row-sparse gradient accumulation into table.row_store.
    impl="triton": the backward runs kernel 1 of smlm/kernels.py instead of the aten/index_put_ reference."""
    if not torch.is_grad_enabled():
        return F.embedding_bag(indices, table, per_sample_weights=weights.to(table.dtype), mode="sum")
    return _RowSparseBag.apply(weights, indices, table.detach(), row_store(table), impl)


def row_sparse_tables(model):
    """Distinct value-table parameters that use row-sparse gradients."""
    seen, out = set(), []
    for m in model.modules():
        if getattr(m, "value_grad", "dense") == "row_sparse" and id(m.values.weight) not in seen:
            seen.add(id(m.values.weight))
            out.append(m.values.weight)
    return out


def clip_row_sparse(tables, max_norm):
    """Global norm over the accumulated row gradients of all row-sparse tables; the clip factor is stored and
    applied in LazyRowAdam.step (avoids rewriting the whole accumulator). Returns the pre-clip norm."""
    if not tables:
        return None
    norms = [row_store(t).acc.norm() for t in tables]
    total = torch.linalg.vector_norm(torch.stack(norms))
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for t in tables:
        row_store(t).grad_scale = coef
    return total


class LazyRowAdam(torch.optim.Optimizer):
    """Adam (no weight decay) on the rows read since the last step; rows that were not read keep their values
    and their Adam state bit-for-bit (global step count for bias correction, as torch SparseAdam / TF LazyAdam).

    Implementation: in practice almost every row is read in a 32k-token step, so instead of gathering and
    scattering ~1M rows (slow), one fused Adam step runs over the whole table with the accumulated row
    gradients, and the few unread rows (values, exp_avg, exp_avg_sq) are saved before and written back after.
    The result equals a row-wise lazy update exactly (see tests/test_sparse_values.py)."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, name="memory_values"):
        params = list(params)
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0))
        for g in self.param_groups:
            g["name"] = name
            g["base_lr"] = lr
        fused = all(p.is_cuda for p in params)
        self._adam = {p: torch.optim.Adam([p], lr=lr, betas=betas, eps=eps, fused=fused or None) for p in params}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = row_store(p)
                if not bool(st.touched.any()):
                    continue
                adam = self._adam[p]
                for g in adam.param_groups:
                    g["lr"], g["betas"], g["eps"] = group["lr"], group["betas"], group["eps"]
                state = adam.state[p]
                unread = (~st.touched).nonzero().squeeze(1)
                keep = [p.index_select(0, unread)]
                if state:
                    keep += [state["exp_avg"].index_select(0, unread), state["exp_avg_sq"].index_select(0, unread)]
                scale = st.grad_scale
                if not (isinstance(scale, float) and scale == 1.0):
                    st.acc.mul_(scale)
                p.grad = st.acc
                adam.step()
                p.grad = None
                p.index_put_((unread,), keep[0])
                if len(keep) == 3:
                    state["exp_avg"].index_put_((unread,), keep[1])
                    state["exp_avg_sq"].index_put_((unread,), keep[2])
                else:                                   # first step: state was created as zeros, unread rows got
                    state["exp_avg"].index_fill_(0, unread, 0.0)        # zero gradients; keep them exactly zero
                    state["exp_avg_sq"].index_fill_(0, unread, 0.0)
                self.state[p] = state                   # expose exp_avg / exp_avg_sq / step for inspection
                st.reset()

    def zero_grad(self, set_to_none=True):
        pass                    # the accumulator is cleared in step()


class OptimizerSet:
    """Several optimizers driven as one (lr schedule via param_groups, step, zero_grad)."""

    def __init__(self, *opts):
        self.opts = [o for o in opts if o is not None]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)
