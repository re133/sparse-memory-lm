"""CPU row storage and touched-row Adam for host-resident value tables.

  python -m smlm.train --model B-1M-sparse --value_device host --value_state int8
"""
import math

import torch

from .sparse_values import LazyRowAdam, RowStore, row_store

CHUNK_ELEMENTS = 1 << 20


class HostRowStore(RowStore):
    """Only visited rows are reset or read for clipping; the full accumulator stays in host RAM."""

    def __init__(self, table):
        if table.device.type != "cpu":
            raise ValueError("host value tables must stay on the CPU")
        super().__init__(table)

    def reset(self, rows=None):
        if rows is None:
            rows = self.touched.nonzero().squeeze(1)
        super().reset(rows)

    def norm(self):
        rows = self.touched.nonzero().squeeze(1)
        if not rows.numel():
            return self.acc.new_zeros(())
        chunk_rows = max(1, CHUNK_ELEMENTS // self.acc.shape[1])
        # Keep each gather bounded; summing chunk norms avoids a full-table fp32 temporary.
        norms = [self.acc.index_select(0, rows[a:a + chunk_rows]).norm()
                 for a in range(0, rows.numel(), chunk_rows)]
        return torch.linalg.vector_norm(torch.stack(norms))


class HostLazyRowAdam(LazyRowAdam):
    """LazyRowAdam semantics without the fp32 reference's full-table update and unread-row copies."""

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, name="memory_values", impl="torch",
                 state_dtype="fp32"):
        params = list(params)
        if any(p.device.type != "cpu" or p.dtype != torch.float32 or p.ndim != 2 or
               not p.is_contiguous() or p.shape[1] == 0 for p in params):
            raise ValueError("HostLazyRowAdam requires contiguous CPU fp32 tables with nonzero width")
        super().__init__(params, lr, betas=betas, eps=eps, name=name, impl="torch", state_dtype=state_dtype)
        # The inherited wrapper is only needed by the full-table reference implementation.
        self._adam = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = row_store(p)
                if self.state_dtype != "fp32":
                    self._step_lowmem(group, p, st)
                    continue
                rows = st.touched.nonzero().squeeze(1)
                if not rows.numel():
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = int(state["step"])
                b1, b2 = group["betas"]
                step_size = group["lr"] / (1 - b1 ** step)
                bc2_sqrt = math.sqrt(1 - b2 ** step)
                chunk_rows = max(1, CHUNK_ELEMENTS // p.shape[1])
                for a in range(0, rows.numel(), chunk_rows):
                    r = rows[a:a + chunk_rows]
                    g = st.acc.index_select(0, r) * st.grad_scale
                    m = state["exp_avg"].index_select(0, r)
                    v = state["exp_avg_sq"].index_select(0, r)
                    # Match the scalar torch Adam operation order, including lerp's fp32 rounding.
                    m.lerp_(g, 1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    denom = (v.sqrt() / bc2_sqrt).add_(group["eps"])
                    values = p.index_select(0, r).addcdiv_(m, denom, value=-step_size)
                    p.index_copy_(0, r, values)
                    state["exp_avg"].index_copy_(0, r, m)
                    state["exp_avg_sq"].index_copy_(0, r, v)
                st.reset(rows)


def value_table_memory_by_device(model, value_state="fp32"):
    """Persistent table allocation split by the explicit host flag, excluding staging/workspace."""
    from .optim import value_table_memory
    return {"host": value_table_memory(model, value_state, host=True),
            "device": value_table_memory(model, value_state, host=False)}
