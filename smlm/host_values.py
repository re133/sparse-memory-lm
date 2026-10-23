"""Train PKM tables in host RAM, staging only the rows read by each memory layer.

  model = enable_host_values(Transformer(cfg)).cuda()
  python -m smlm.train --model B-16M-sparse --value_device host --value_state int8 ...
  python -m smlm.train ... --value_device host --value_optim rowrms --hot_rows mem_access_train.npy --hot_frac 0.1

HOT (enable_hot_rows): the most-read rows, ranked by the read counts of an earlier run (profile-guided, like
AutoFDO/Propeller for code), stay on the GPU with their gradient accumulator and row-wise optimizer state; only the
other rows travel between RAM and GPU. Same math, other summation order. The host copy of a hot row is stale while
training and is written back by flush(), which state_dict() calls (checkpoints, model.pt).

GACC (enable_step_rows): the gradients of the rows read in a step are summed on the GPU (StepRows) instead of in a
table-sized accumulator in RAM; the row-wise optimizer gets each row's sum once per step. Saves the host accumulator
(B-16M: 26 GB), the scatter-add per layer and micro-batch, and the clipping pass over it. Same math.
"""
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from .host_optim import _c_ok, rows_library
from .sparse_values import CHUNK, row_store

PHASES = ("unique", "indices_d2h", "gather", "rows_h2d", "gradients_d2h", "scatter")


class HostEmbedding(nn.Embedding):
    """A normal checkpoint parameter that stays fp32 on CPU when its parent moves to CUDA.

    Only staging memory is pinned, never the whole table. Transfers finish before reusing that buffer;
    optimizer steps cannot race pending gradient copies or reads for the next step.
    """

    def __init__(self, source, profile=False):
        super().__init__(source.num_embeddings, source.embedding_dim, _weight=source.weight)
        self.weight = source.weight
        self.weight.host_values = True
        self.profile = profile
        self.timings = dict.fromkeys(PHASES, 0.0)
        self._staging = None
        self.hot = None
        self.step_rows = None
        self.register_state_dict_pre_hook(lambda module, *_: module.flush())

    def flush(self):
        """Write the GPU copies of the hot rows back into the host table."""
        if self.hot is not None:
            self.weight.detach().index_copy_(0, self.hot.rows.cpu(), self.hot.values.detach().cpu())

    def _apply(self, fn, recurse=True):
        # Do not even call fn on the table: a temporary full-table CUDA copy would already OOM.
        return self

    @contextmanager
    def phase(self, name, device=None):
        if not self.profile:
            yield
            return
        cuda = device is not None and device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(device)
            self.timings[name] += time.perf_counter() - start

    def staging(self, rows):
        if self._staging is None or self._staging.shape[0] < rows:
            # Bucket capacities: the pinned allocator caches freed blocks, so a new allocation for
            # each slightly larger unique-row count could retain many almost identical large buffers.
            capacity = min(self.num_embeddings, 1 << (max(1, rows) - 1).bit_length())
            self._staging = None
            self._staging = torch.empty((capacity, self.embedding_dim), dtype=torch.float32,
                                        device="cpu", pin_memory=True)
        return self._staging[:rows]

    def gather(self, indices):
        """(host rows, remapped indices, values of the distinct rows on the device, hot) where hot = None or
        (positions of the hot rows among the distinct ones, their slots); host rows are only the other ones."""
        device = indices.device
        with self.phase("unique", device):
            unique, remapped = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        hot = None
        if self.hot is not None:
            slot = self.hot.slot[unique]
            is_hot = slot >= 0
            at_hot, at_cold = is_hot.nonzero().squeeze(1), (~is_hot).nonzero().squeeze(1)
            hot = (at_hot, slot[at_hot].long(), at_cold)
            unique = unique[at_cold]
        with self.phase("indices_d2h", device):
            rows = unique.cpu()
        with self.phase("gather"):
            lib = rows_library()
            if device.type == "cuda":
                cpu = self.staging(rows.numel())
                if lib is not None and _c_ok(self.weight, rows):
                    lib.host_rows_gather(cpu.data_ptr(), self.weight.data_ptr(), rows.data_ptr(), rows.numel(),
                                         self.embedding_dim, torch.get_num_threads())
                else:
                    torch.index_select(self.weight.detach(), 0, rows, out=cpu)
            else:
                cpu = self.weight.detach().index_select(0, rows)
        with self.phase("rows_h2d", device):
            # Blocking completion makes the one pinned buffer safe to reuse in the next layer.
            values = cpu.to(device, non_blocking=False)
        if hot is not None:
            at_hot, slots, at_cold = hot
            both = values.new_empty((at_hot.numel() + at_cold.numel(), self.embedding_dim))
            both.index_copy_(0, at_cold, values)
            both.index_copy_(0, at_hot, self.hot.values.index_select(0, slots))
            values = both
        if self.step_rows is not None:
            hot = (hot, unique)                   # the step accumulator needs the rows on the device
        return rows, remapped.view_as(indices), values, hot

    def accumulate(self, rows, gradient, hot=None):
        device_rows = None
        if self.step_rows is not None:
            hot, device_rows = hot
        if hot is not None:
            # rows are distinct within one call, so index_add_ adds each hot row once per layer and micro-batch
            at_hot, slots, at_cold = hot
            self.hot.acc.index_add_(0, slots, gradient.index_select(0, at_hot))
            self.hot.touched[slots] = True
            gradient = gradient.index_select(0, at_cold)
        if device_rows is not None:
            with self.phase("scatter"):
                self.step_rows.add(device_rows, gradient)
            return
        with self.phase("gradients_d2h", gradient.device):
            if gradient.is_cuda:
                cpu = self.staging(rows.numel())
                cpu.copy_(gradient, non_blocking=False)
            else:
                cpu = gradient
        with self.phase("scatter"):
            store = row_store(self.weight)
            # rows are unique within this layer; index_add also sums across layers and micro-batches.
            lib = rows_library()
            if lib is not None and store.acc.dtype == torch.float32 and _c_ok(store.acc, store.touched, rows, cpu):
                lib.host_rows_add(store.acc.data_ptr(), store.touched.data_ptr(), rows.data_ptr(), cpu.data_ptr(),
                                  rows.numel(), store.acc.shape[1], torch.get_num_threads())
            else:
                store.acc.index_add_(0, rows, cpu)
                store.touched[rows] = True

    def bag(self, indices, weights, impl="torch"):
        if self.weight.device.type != "cpu" or self.weight.dtype != torch.float32:
            raise ValueError("host values must remain fp32 on CPU")
        rows, remapped, values, hot = self.gather(indices)
        if not torch.is_grad_enabled():
            if impl == "triton":
                from .kernels import bag_forward
                return bag_forward(remapped, weights, values).to(values.dtype)
            return F.embedding_bag(remapped, values, per_sample_weights=weights.to(values.dtype), mode="sum")
        return _HostBag.apply(weights, remapped, values, self.weight, rows, self, impl, hot)


class _HostBag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, values, table, rows, owner, impl, hot=None):
        n, j = indices.shape
        flat = indices.reshape(-1).contiguous()
        w = weights.reshape(-1).to(values.dtype).contiguous()
        offsets = torch.arange(0, n * j, j, device=indices.device)
        if impl == "triton":
            from .kernels import bag_forward
            out = bag_forward(indices, w.view_as(indices), values).to(values.dtype)
            offset2bag = offsets.new_empty(0)
        else:
            out, offset2bag, _, _ = torch.ops.aten._embedding_bag(values, flat, offsets, False, 0, False,
                                                                w, False, -1)
            if offset2bag.numel() != flat.numel():
                offset2bag = torch.arange(n, device=indices.device).repeat_interleave(j)
        ctx.save_for_backward(w, flat, values, rows, offsets, offset2bag)
        ctx.owner, ctx.impl, ctx.hot = owner, impl, hot
        ctx.train_table = table.requires_grad
        ctx.wshape, ctx.wdtype, ctx.ishape = weights.shape, weights.dtype, indices.shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, flat, values, rows, offsets, offset2bag = ctx.saved_tensors
        g = grad_out.contiguous().float()
        acc = torch.zeros_like(values) if ctx.train_table else None
        if ctx.impl == "triton" and ctx.train_table:
            from .kernels import bag_backward_rows
            touched = torch.zeros(values.shape[0], dtype=torch.bool, device=values.device)
            gw = bag_backward_rows(g, flat.view(ctx.ishape), w, values, acc, touched)
        else:
            if offset2bag.numel() != flat.numel():
                offset2bag = torch.arange(ctx.ishape[0], device=flat.device).repeat_interleave(ctx.ishape[1])
            gw = torch.ops.aten._embedding_bag_per_sample_weights_backward(g, values, flat, offsets,
                                                                          offset2bag, 0, -1)
            if ctx.train_table:
                for a in range(0, flat.numel(), CHUNK):
                    contrib = g.index_select(0, offset2bag[a:a + CHUNK]).mul_(w[a:a + CHUNK, None])
                    acc.index_put_((flat[a:a + CHUNK],), contrib, accumulate=True)
        if ctx.train_table:
            ctx.owner.accumulate(rows, acc, ctx.hot)
        return gw.view(ctx.wshape).to(ctx.wdtype), None, None, None, None, None, None, None


def enable_host_values(model, profile=False):
    """Opt in before moving a CPU fp32 model to the GPU; state_dict/config stay loadable by normal models.

    PKM row-sparse training only. Full-model dtype conversions keep the host table fp32 by design.
    The synchronous dynamic lookup cannot be captured in a CUDA graph.
    """
    from .pkm import ProductKeyMemory

    memories = [m for m in model.modules() if isinstance(m, ProductKeyMemory)]
    if not memories or any(getattr(m, "is_engram", False) for m in model.modules()):
        raise ValueError("value_device=host requires a PKM model without Engram tables")
    for m in memories:
        if m.value_grad != "row_sparse":
            raise ValueError("value_device=host requires row_sparse PKM gradients")
        if m.values.weight.device.type != "cpu" or m.values.weight.dtype != torch.float32:
            raise ValueError("enable host values on a CPU fp32 model before moving it to the GPU")
        if not m.values.weight.is_contiguous() or m.values.weight.shape[1] == 0:
            raise ValueError("host values require contiguous tables with nonzero width")
    shared = {}
    for m in memories:
        key = id(m.values.weight)
        if key not in shared:
            shared[key] = m.values if isinstance(m.values, HostEmbedding) else HostEmbedding(m.values, profile)
        m.values = shared[key]
        m.values.profile = profile
        m.decode_graph = False
        m._graph = None
    return model


class HotRows:
    """Rows of a host table that live on the GPU (HOT): values, gradient accumulator, touched mask, and the map
    row -> slot (-1 for rows that stay in RAM). The row-wise optimizer keeps their state in slot order too."""

    def __init__(self, table, rows, device):
        rows = torch.as_tensor(rows, dtype=torch.int64).sort().values
        self.rows = rows.to(device)
        self.slot = torch.full((table.shape[0],), -1, dtype=torch.int32, device=device)
        self.slot[self.rows] = torch.arange(rows.numel(), dtype=torch.int32, device=device)
        self.values = table.detach().index_select(0, rows).to(device)
        self.acc = torch.zeros_like(self.values)
        self.touched = torch.zeros(rows.numel(), dtype=torch.bool, device=device)


class StepRows:
    """GACC: gradient sums of the host-table rows read in the current step, on the GPU. A row gets the next free
    slot when it is first read in the step; every later read (other layers, micro-batches) adds into that slot.
    Once all slots are taken, the gradients of rows without a slot go to RAM (overflow) and are summed per row
    before the optimizer step, so a step that reads more rows than the capacity is slower, not wrong."""

    def __init__(self, num_rows, dim, capacity, device):
        self.slot = torch.full((num_rows,), -1, dtype=torch.int32, device=device)
        self.rows = torch.empty(capacity, dtype=torch.int64, device=device)
        self.acc = torch.zeros(capacity, dim, dtype=torch.float32, device=device)
        self.n = 0
        self.peak = 0                      # most distinct rows read in one step so far, overflow included
        self.spilled_steps = 0             # steps with an overflow
        self._spill = []                   # (rows, gradients) on the CPU, a row may appear in several parts
        self._summed = None

    def add(self, rows, gradient):
        """rows: distinct row ids on the device, gradient: one row each."""
        slots = self.slot[rows]
        new = (slots < 0).nonzero().squeeze(1)
        free = self.rows.numel() - self.n
        if new.numel() > free:
            out = new[free:]
            self._spill.append((rows.index_select(0, out).cpu(), gradient.index_select(0, out).cpu()))
            self._summed = None
            new = new[:free]
            keep = torch.ones_like(slots, dtype=torch.bool)
            keep[out] = False
        else:
            keep = None
        k = new.numel()
        if k:
            ids = torch.arange(self.n, self.n + k, dtype=torch.int32, device=rows.device)
            fresh = rows.index_select(0, new)
            self.slot[fresh] = ids
            self.rows[self.n:self.n + k] = fresh
            slots[new] = ids
            self.n += k
            self.peak = max(self.peak, self.n)
        if keep is not None:
            slots, gradient = slots[keep], gradient[keep]
        self.acc.index_add_(0, slots.long(), gradient)

    def spilled(self):
        """(rows, gradient sums) of the overflow on the CPU, each row once; None without an overflow."""
        if not self._spill:
            return None
        if self._summed is None:
            rows = torch.cat([r for r, _ in self._spill])
            unique, inverse = torch.unique(rows, sorted=True, return_inverse=True)
            sums = torch.zeros(unique.numel(), self.acc.shape[1], dtype=torch.float32)
            sums.index_add_(0, inverse, torch.cat([g for _, g in self._spill]))
            self._summed = (unique, sums)
            self.peak = max(self.peak, self.n + unique.numel())
        return self._summed

    def reset(self):
        if self._spill:
            self.spilled()                 # counts the step's overflow rows in peak
            self.spilled_steps += 1
            self._spill, self._summed = [], None
        self.slot[self.rows[:self.n]] = -1
        self.acc[:self.n].zero_()
        self.n = 0


class StepStore:
    """row_store() of a host table with GACC: no table-sized accumulator, gradients sit in `steprows` (and in
    `hot` for hot rows); grad_scale is set by clipping as for a RowStore."""

    def __init__(self, steprows, hot=None):
        self.steprows, self.hot, self.grad_scale = steprows, hot, 1.0

    def norm(self):
        parts = [self.steprows.acc[:self.steprows.n].norm()]
        if self.hot is not None:
            parts.append(self.hot.acc.norm())
        spilled = self.steprows.spilled()
        if spilled is not None:
            parts.append(spilled[1].norm().to(parts[0].device))
        return torch.linalg.vector_norm(torch.stack(parts)).cpu()

    def reset(self, rows=None):
        self.steprows.reset()
        if self.hot is not None:
            self.hot.acc.zero_()
            self.hot.touched.zero_()
        self.grad_scale = 1.0


def enable_step_rows(model, capacity, device):
    """GACC for every host table: sum each step's row gradients on `device` in up to `capacity` rows (rows read in
    one step, hot rows not counted). Needs a row-wise table optimizer. Call after enable_hot_rows, if both."""
    for module in model.modules():
        if isinstance(module, HostEmbedding) and module.step_rows is None:
            module.step_rows = StepRows(module.num_embeddings, module.embedding_dim, capacity, device)
            module.weight.row_store = StepStore(module.step_rows, module.hot)
    return model


def enable_hot_rows(model, reads, fraction, device):
    """Keep the `fraction` most-read rows of each host table on `device`, ranked by `reads` (read counts per row
    from an earlier run, e.g. its mem_access_train.npy). Needs a row-wise table optimizer (smlm/row_optim.py):
    Adam's two moments per value would not fit next to the rows. Call after moving the model to the device and
    before the first step."""
    reads = torch.as_tensor(reads)
    for module in model.modules():
        if isinstance(module, HostEmbedding) and module.hot is None:
            if reads.shape != (module.num_embeddings,):
                raise ValueError(f"read counts for {reads.shape[0]} rows, the table has {module.num_embeddings}")
            n = int(round(fraction * module.num_embeddings))
            if module.step_rows is not None:
                raise ValueError("enable hot rows before the step accumulator")
            module.hot = HotRows(module.weight, torch.argsort(reads, descending=True, stable=True)[:n], device)
            # a row store created later picks it up (HostRowStore); don't create the table-sized one here
            module.weight.hot_rows = module.hot
            if getattr(module.weight, "row_store", None) is not None:
                module.weight.row_store.hot = module.hot
    return model


def host_timings(model, reset=False):
    """Cumulative staging wall seconds (synchronized when profiling), counting shared tables once."""
    totals = dict.fromkeys(PHASES, 0.0)
    for module in model.modules():
        if isinstance(module, HostEmbedding):
            for phase, elapsed in module.timings.items():
                totals[phase] += elapsed
            if reset:
                module.timings = dict.fromkeys(PHASES, 0.0)
    return totals
