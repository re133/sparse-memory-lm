"""Train PKM tables in host RAM, staging only the rows read by each memory layer.

  model = enable_host_values(Transformer(cfg)).cuda()
  python -m smlm.train --model B-16M-sparse --value_device host --value_state int8 ...
"""
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

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
        device = indices.device
        with self.phase("unique", device):
            unique, remapped = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        with self.phase("indices_d2h", device):
            rows = unique.cpu()
        with self.phase("gather"):
            if device.type == "cuda":
                cpu = self.staging(rows.numel())
                torch.index_select(self.weight.detach(), 0, rows, out=cpu)
            else:
                cpu = self.weight.detach().index_select(0, rows)
        with self.phase("rows_h2d", device):
            # Blocking completion makes the one pinned buffer safe to reuse in the next layer.
            values = cpu.to(device, non_blocking=False)
        return rows, remapped.view_as(indices), values

    def accumulate(self, rows, gradient):
        with self.phase("gradients_d2h", gradient.device):
            if gradient.is_cuda:
                cpu = self.staging(rows.numel())
                cpu.copy_(gradient, non_blocking=False)
            else:
                cpu = gradient
        with self.phase("scatter"):
            store = row_store(self.weight)
            # rows are unique within this layer; index_add also sums across layers and micro-batches.
            store.acc.index_add_(0, rows, cpu)
            store.touched[rows] = True

    def bag(self, indices, weights, impl="torch"):
        if self.weight.device.type != "cpu" or self.weight.dtype != torch.float32:
            raise ValueError("host values must remain fp32 on CPU")
        rows, remapped, values = self.gather(indices)
        if not torch.is_grad_enabled():
            if impl == "triton":
                from .kernels import bag_forward
                return bag_forward(remapped, weights, values).to(values.dtype)
            return F.embedding_bag(remapped, values, per_sample_weights=weights.to(values.dtype), mode="sum")
        return _HostBag.apply(weights, remapped, values, self.weight, rows, self, impl)


class _HostBag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, indices, values, table, rows, owner, impl):
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
        ctx.owner, ctx.impl = owner, impl
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
            ctx.owner.accumulate(rows, acc)
        return gw.view(ctx.wshape).to(ctx.wdtype), None, None, None, None, None, None


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
