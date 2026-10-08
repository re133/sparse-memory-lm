"""CPU row storage and touched-row Adam for host-resident value tables.

  python -m smlm.train --model B-1M-sparse --value_device host --value_state int8
"""
import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from .sparse_values import LazyRowAdam, RowStore, row_store

CHUNK_ELEMENTS = 1 << 20
_LIB = []


def rows_library():
    """C row loops (smlm/host_rows.c), compiled on first use into .cache/host_rows. None without gcc, or with
    SMLM_HOST_C=0; then everything runs through the torch path."""
    if os.environ.get("SMLM_HOST_C", "1") == "0":
        return None
    if _LIB:
        return _LIB[0]
    source = Path(__file__).with_name("host_rows.c")
    flags = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off", "-pthread", "-shared", "-fPIC"]
    digest = hashlib.sha256(source.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    cache = source.parents[1] / ".cache" / "host_rows"
    library = cache / f"host_rows_{digest}.so"
    lib = None
    try:
        if not library.exists():
            cache.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="host_rows-", suffix=".so", dir=cache)
            os.close(fd)
            try:
                result = subprocess.run(["gcc", *flags, str(source), "-o", temporary, "-lm"],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                os.replace(temporary, library)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        lib = ctypes.CDLL(str(library))
        p, i64, f32, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float, ctypes.c_int
        lib.host_rows_adam.argtypes = [p, p, p, p, p, p, i64, i64, f32, f32, f32, f32, f32, f32, f32, i32]
        lib.host_rows_add.argtypes = [p, p, p, p, i64, i64, i32]
        lib.host_rows_gather.argtypes = [p, p, p, i64, i64, i32]
        for fn in (lib.host_rows_adam, lib.host_rows_add, lib.host_rows_gather):
            fn.restype = None
    except (OSError, RuntimeError) as exc:
        import warnings
        warnings.warn(f"host row loops fall back to torch: {exc}")
        lib = None
    _LIB.append(lib)
    return lib


def _c_ok(*tensors):
    return all(t.device.type == "cpu" and t.is_contiguous() and t.dtype in (torch.float32, torch.int64, torch.bool)
               for t in tensors)


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
                lib = rows_library()
                if lib is not None and _c_ok(p, st.acc, state["exp_avg"], state["exp_avg_sq"], st.touched):
                    # one pass per touched row; also zeroes the accumulator rows and clears touched
                    lib.host_rows_adam(p.data_ptr(), st.acc.data_ptr(), state["exp_avg"].data_ptr(),
                                       state["exp_avg_sq"].data_ptr(), st.touched.data_ptr(), rows.data_ptr(),
                                       rows.numel(), p.shape[1], st.grad_scale, 1 - b1, b2, 1 - b2, bc2_sqrt,
                                       group["eps"], -step_size, torch.get_num_threads())
                    st.grad_scale = 1.0
                    continue
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
