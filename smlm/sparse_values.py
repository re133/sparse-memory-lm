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
        w = weights.reshape(-1).to(table.dtype).contiguous()
        if impl == "triton":                            # kernel 2 bag (smlm/kernels.py)
            from .kernels import bag_forward
            out = bag_forward(flat.view(n, j), w.view(n, j), table).to(table.dtype)
            ctx.save_for_backward(w, flat)
        else:
            offsets = torch.arange(0, n * j, j, device=indices.device)
            out, offset2bag, _, _ = torch.ops.aten._embedding_bag(table, flat, offsets, False, 0, False, w,
                                                                   False, -1)
            if offset2bag.numel() != flat.numel():      # CPU fp32 fast path leaves it empty (fixed-size bags)
                offset2bag = torch.arange(n, device=indices.device).repeat_interleave(j)
            ctx.save_for_backward(w, flat, offsets, offset2bag)
        ctx.table, ctx.store, ctx.wshape, ctx.wdtype = table, store, weights.shape, weights.dtype
        ctx.impl, ctx.ishape = impl, indices.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        table, store = ctx.table, ctx.store
        g = grad_out.contiguous().to(store.acc.dtype)
        if ctx.impl == "triton":                       # kernel 1 (smlm/kernels.py)
            w, flat = ctx.saved_tensors
            from .kernels import bag_backward_rows
            gw = bag_backward_rows(g, flat.view(ctx.ishape), w, table, store.acc, store.touched)
            ctx.table = ctx.store = None
            return gw.view(ctx.wshape).to(ctx.wdtype), None, None, None, None
        w, flat, offsets, offset2bag = ctx.saved_tensors
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
        if impl == "triton":
            from .kernels import bag_forward
            return bag_forward(indices, weights, table).to(table.dtype)
        return F.embedding_bag(indices, table, per_sample_weights=weights.to(table.dtype), mode="sum")
    if not table.requires_grad:
        # frozen table: gradients only for the weights, no row accumulator
        return F.embedding_bag(indices, table, per_sample_weights=weights.to(table.dtype), mode="sum")
    # the table goes in as an input (not detached) so that the output needs a gradient even when only the table
    # is trained; its backward returns None for it, the gradient goes into table.row_store instead
    return _RowSparseBag.apply(weights, indices, table, row_store(table), impl)


def row_sparse_tables(model):
    """Distinct value-table parameters that use row-sparse gradients."""
    seen, out = set(), []
    for m in model.modules():
        if (getattr(m, "value_grad", "dense") == "row_sparse" and m.values.weight.requires_grad
                and id(m.values.weight) not in seen):
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

    The fp32 torch path: in practice almost every row is read in a 32k-token step, so instead of gathering and
    scattering ~1M rows (slow), one fused Adam step runs over the whole table with the accumulated row
    gradients, and the few unread rows (values, exp_avg, exp_avg_sq) are saved before and written back after.
    The result equals a row-wise lazy update exactly (see tests/test_sparse_values.py).

    state_dtype="bf16" / "int8" keeps compact moments and computes updates in fp32. The torch compact path
    gathers only touched rows in bounded chunks; Triton updates them in place. Int8 uses signed square-root
    first moments and unsigned fourth-root second moments, each with an fp32 scale per row.
    Stores round to nearest even; a positive second moment never gets a zero code.
    Values and RowStore stay fp32; compact modes require contiguous fp32 tables.
    """

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, name="memory_values", impl="torch",
                 state_dtype="fp32"):
        params = list(params)
        if state_dtype not in ("fp32", "bf16", "int8"):
            raise ValueError(f"unknown state_dtype: {state_dtype}")
        if state_dtype != "fp32":
            if any(p.dtype != torch.float32 or p.ndim != 2 or not p.is_contiguous() or p.shape[1] == 0
                   for p in params):
                raise ValueError("compact LazyRowAdam states require contiguous fp32 tables with nonzero width")
            if not 0 <= lr or not 0 <= eps or not all(0 <= b < 1 for b in betas):
                raise ValueError("invalid LazyRowAdam learning rate, epsilon or betas")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0))
        for g in self.param_groups:
            g["name"] = name
            g["base_lr"] = lr
        assert impl in ("torch", "triton")
        self.impl = impl                    # "triton": kernel 3 (smlm/kernels.py), in place, no copies
        self.state_dtype = state_dtype
        fused = all(p.is_cuda for p in params)
        self._adam = ({p: torch.optim.Adam([p], lr=lr, betas=betas, eps=eps, fused=fused or None) for p in params}
                      if state_dtype == "fp32" else {})

    def _lowmem_state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = torch.zeros((), dtype=torch.float32)
            quantized = self.state_dtype == "int8"
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.int8 if quantized else torch.bfloat16)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.uint8 if quantized else torch.bfloat16)
            if quantized:
                state["exp_avg_scale"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
                state["exp_avg_sq_scale"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
        return state

    @torch.no_grad()
    def _step_lowmem(self, group, p, st):
        if self.impl == "torch" and not bool(st.touched.any()):
            return
        state = self._lowmem_state(p)
        state["step"] += 1
        step = int(state["step"])
        b1, b2 = group["betas"]
        if self.impl == "triton":
            from .kernels import lazy_adam_step_lowmem
            lazy_adam_step_lowmem(p, state["exp_avg"], state["exp_avg_sq"], st.acc, st.touched, st.grad_scale,
                                  step, group["lr"], b1, b2, group["eps"], self.state_dtype,
                                  state.get("exp_avg_scale"), state.get("exp_avg_sq_scale"))
            st.grad_scale = 1.0
            return
        rows = st.touched.nonzero().squeeze(1)
        step_size = group["lr"] / (1 - b1 ** step)
        bc2_sqrt = math.sqrt(1 - b2 ** step)
        # Bound fp32 scratch even when almost every row of a large table was read.
        chunk_rows = max(1, (1 << 20) // p.shape[1])
        for a in range(0, rows.numel(), chunk_rows):
            r = rows[a:a + chunk_rows]
            g = st.acc.index_select(0, r) * st.grad_scale
            m = state["exp_avg"].index_select(0, r).float()
            v = state["exp_avg_sq"].index_select(0, r).float()
            if self.state_dtype == "int8":
                m = m / 127
                m = m.sign() * m.square() * state["exp_avg_scale"].index_select(0, r)[:, None]
                v = (v / 255 * state["exp_avg_sq_scale"].index_select(0, r)[:, None]).square().square()
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            values = p.index_select(0, r) - step_size * m / (v.sqrt() / bc2_sqrt + group["eps"])
            p.index_copy_(0, r, values)
            if self.state_dtype == "int8":
                root_v = v.sqrt().sqrt()
                ms, vs = m.abs().amax(dim=1), root_v.amax(dim=1)
                # A zero row has zero codes and zero scale; avoid dividing by zero while encoding it.
                mq = (m.sign() * (m.abs() / torch.where(ms > 0, ms, 1.0)[:, None]).sqrt()
                      * 127).round().clamp(-127, 127)
                vq = (root_v / torch.where(vs > 0, vs, 1.0)[:, None] * 255).round().clamp(0, 255)
                # A lost denominator with surviving momentum can turn a zero gradient into a huge update.
                vq = torch.where(v > 0, vq.clamp_min(1), vq)
                state["exp_avg"].index_copy_(0, r, mq.to(torch.int8))
                state["exp_avg_sq"].index_copy_(0, r, vq.to(torch.uint8))
                state["exp_avg_scale"].index_copy_(0, r, ms)
                state["exp_avg_sq_scale"].index_copy_(0, r, vs)
            else:
                state["exp_avg"].index_copy_(0, r, m.to(torch.bfloat16))
                state["exp_avg_sq"].index_copy_(0, r, v.to(torch.bfloat16))
        st.reset(rows)

    @torch.no_grad()
    def _step_triton(self, group, p, st):
        from .kernels import lazy_adam_step
        state = self.state[p]
        if not state:
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["step"] += 1
        b1, b2 = group["betas"]
        lazy_adam_step(p, state["exp_avg"], state["exp_avg_sq"], st.acc, st.touched, st.grad_scale,
                       int(state["step"]), group["lr"], b1, b2, group["eps"])
        st.grad_scale = 1.0                 # the kernel has zeroed the read accumulator rows and the mask

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = row_store(p)
                if self.state_dtype != "fp32":
                    self._step_lowmem(group, p, st)
                    continue
                if self.impl == "triton":
                    self._step_triton(group, p, st)
                    continue
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
        # the accumulator is cleared in step(); rows accumulated by a backward without a following step stay
        # in table.row_store (row_store(table).reset() drops them)
        pass

    def load_state_dict(self, state_dict):
        # the torch path keeps its moments in an inner Adam per table that the inherited load would not fill,
        # so a resumed run would silently restart the bias correction. Training never resumes here.
        raise NotImplementedError("LazyRowAdam can't be restored from a state dict (no resuming)")


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
