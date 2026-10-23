"""Row-wise optimizers for row-sparse value tables: one state number per row instead of Adam's two moments per value
(step 18, HEIM 2).

  python -m smlm.train --model B-1M-sparse --value_optim rowrms ...
  python -m smlm.train --model B-16M-sparse --value_device host --value_optim rowrms ...

Like LazyRowAdam, only the rows read since the last step change; rows that were not read keep their values and their
state exactly. g = a row's accumulated gradient times the clip factor, q = the mean of g^2 over the row.
  rowrms      v_r = b2 v_r + (1 - b2) q,  n_r += 1,  p_r -= lr g / (sqrt(v_r / (1 - b2^n_r)) + eps)
              Adam without momentum and with one second moment per row. The bias correction counts the row's own
              updates (n_r), not the global step: with a global count, a row first read at step 5000 would take a
              first step of lr / sqrt(1 - b2) = 4.5 lr instead of lr.
  rowadagrad  v_r += q,  p_r -= lr g / (sqrt(v_r) + eps)            (exact row-wise Adagrad as in FBGEMM / TorchRec)
State per row: fp32 v plus an int32 n (rowrms), 8 or 4 bytes per row instead of Adam's 2 x 4 x width (B-16M: 134 MB
instead of 51.5 GB). The update reads and writes a row's values and its gradient, half the bytes of Adam's step.
"""
import torch

from .sparse_values import row_store

KINDS = ("rowrms", "rowadagrad")
CHUNK_ROWS = 1 << 16                       # rows per chunk on the GPU path (65,536 x 384 fp32 = 100 MB temporaries)


class RowWiseOptimizer(torch.optim.Optimizer):
    """rowrms / rowadagrad (see the module docstring) on row-sparse tables, on the GPU or in host RAM (C loop)."""

    def __init__(self, params, lr, kind="rowrms", beta2=0.95, eps=1e-8, name="memory_values"):
        params = list(params)
        if kind not in KINDS:
            raise ValueError(f"unknown row-wise optimizer: {kind}")
        if any(p.dtype != torch.float32 or p.ndim != 2 or not p.is_contiguous() or p.shape[1] == 0 for p in params):
            raise ValueError("row-wise optimizers require contiguous fp32 tables with nonzero width")
        if not 0 <= lr or not 0 <= eps or not 0 <= beta2 < 1:
            raise ValueError("invalid learning rate, epsilon or beta2")
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps, weight_decay=0.0))
        for g in self.param_groups:
            g["name"] = name
            g["base_lr"] = lr
        self.kind = kind

    def _state(self, p):
        state = self.state[p]
        if not state:
            state["step"] = torch.zeros((), dtype=torch.float32)
            state["v"] = torch.zeros(p.shape[0], dtype=torch.float32, device=p.device)
            if self.kind == "rowrms":
                state["n"] = torch.zeros(p.shape[0], dtype=torch.int32, device=p.device)
        return state

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                st = row_store(p)
                state = self._state(p)
                state["step"] += 1
                if getattr(p, "host_values", False) or p.device.type == "cpu":
                    hot = getattr(st, "hot", None)
                    if hot is not None:
                        # the hot rows' state lives next to them on the GPU, in slot order (smlm/host_values.py)
                        if "hot_v" not in state:
                            state["hot_v"] = state["v"].index_select(0, hot.rows.cpu()).to(hot.values.device)
                            if "n" in state:
                                state["hot_n"] = state["n"].index_select(0, hot.rows.cpu()).to(hot.values.device)
                        self._masked(group, hot.values, hot.acc, hot.touched, state["hot_v"], state.get("hot_n"),
                                     st.grad_scale)
                    self._step_rows(group, p, st, state)
                else:
                    self._masked(group, p, st.acc, st.touched, state["v"], state.get("n"), st.grad_scale)
                    st.grad_scale = 1.0

    def _scale(self, group, v, n):
        # lr / (sqrt(bias-corrected v) + eps); n is the row's update count after this update (rowrms)
        if self.kind == "rowrms":
            v = v / (1 - group["beta2"] ** n.double()).float()
        return group["lr"] / (v.sqrt() + group["eps"])

    def _step_gacc(self, group, p, st, state):
        # GACC (smlm/host_values.py StepRows): each row read in the step has its summed gradient in a slot on the
        # GPU. Copy the slots to RAM in chunks, two pinned buffers in turn, so that the copy of the next chunk runs
        # while the C loop updates the rows of this one.
        sr = st.steprows
        n, scale = sr.n, float(st.grad_scale)
        if n:
            from .host_optim import rows_library
            lib = rows_library()
            rows = sr.rows[:n].cpu()
            cuda = sr.acc.is_cuda
            chunk = CHUNK_ROWS * 2
            if cuda and (getattr(self, "_pinned", None) is None or self._pinned[0].shape[1] != p.shape[1]):
                self._pinned = [torch.empty((chunk, p.shape[1]), dtype=torch.float32, pin_memory=True)
                                for _ in range(2)]

            def fetch(i):
                a, b = i * chunk, min(n, (i + 1) * chunk)
                if not cuda:
                    return sr.acc[a:b], None
                buf = self._pinned[i % 2][:b - a]
                buf.copy_(sr.acc[a:b], non_blocking=True)
                done = torch.cuda.Event()
                done.record()
                return buf, done

            count = state.get("n")
            pending = fetch(0)
            for i in range(-(-n // chunk)):
                g, done = pending
                if (i + 1) * chunk < n:
                    pending = fetch(i + 1)          # its buffer held chunk i - 1, which is finished
                if done is not None:
                    done.synchronize()
                r = rows[i * chunk:i * chunk + g.shape[0]]
                if lib is not None:
                    lib.host_rows_rowwise(p.data_ptr(), None, g.data_ptr(), state["v"].data_ptr(),
                                          0 if count is None else count.data_ptr(), None, r.data_ptr(), r.numel(),
                                          p.shape[1], scale, group["beta2"], group["eps"], group["lr"],
                                          KINDS.index(self.kind), torch.get_num_threads())
                else:
                    g = g * scale
                    v, k = self._update_state(group, state, r, g.square().mean(1))
                    p.index_copy_(0, r, p.index_select(0, r) - self._scale(group, v, k)[:, None] * g)
        sr.reset()
        st.grad_scale = 1.0

    def _step_rows(self, group, p, st, state):
        if getattr(st, "steprows", None) is not None:
            return self._step_gacc(group, p, st, state)
        rows = st.touched.nonzero().squeeze(1)
        if not rows.numel():
            st.grad_scale = 1.0
            return
        from .host_optim import _c_ok, rows_library
        lib = rows_library()
        n = state.get("n")
        scale = float(st.grad_scale)
        if lib is not None and _c_ok(p, st.acc, state["v"], st.touched, rows) and (n is None or n.is_contiguous()):
            lib.host_rows_rowwise(p.data_ptr(), st.acc.data_ptr(), None, state["v"].data_ptr(),
                                  0 if n is None else n.data_ptr(), st.touched.data_ptr(), rows.data_ptr(),
                                  rows.numel(), p.shape[1], scale, group["beta2"], group["eps"], group["lr"],
                                  KINDS.index(self.kind), torch.get_num_threads())
            st.grad_scale = 1.0
            return
        chunk = max(1, (1 << 20) // p.shape[1])
        for a in range(0, rows.numel(), chunk):
            r = rows[a:a + chunk]
            g = st.acc.index_select(0, r) * scale
            v, k = self._update_state(group, state, r, g.square().mean(1))
            p.index_copy_(0, r, p.index_select(0, r) - self._scale(group, v, k)[:, None] * g)
        st.reset(rows)

    def _update_state(self, group, state, r, q):
        b2 = group["beta2"]
        if self.kind == "rowrms":
            v = b2 * state["v"].index_select(0, r) + (1 - b2) * q
            k = state["n"].index_select(0, r) + 1
            state["n"].index_copy_(0, r, k)
        else:
            v, k = state["v"].index_select(0, r) + q, None
        state["v"].index_copy_(0, r, v)
        return v, k

    def _masked(self, group, p, acc, touched, v_all, n_all, scale):
        # GPU: in a 32k-token step B-1M reads almost every row, so run over the whole table in chunks with the
        # touched mask instead of gathering the rows; untouched rows get a zero step (their gradient is zero too)
        b2 = group["beta2"]
        for a in range(0, p.shape[0], CHUNK_ROWS):
            t = touched[a:a + CHUNK_ROWS]
            g = acc[a:a + CHUNK_ROWS]
            if not (isinstance(scale, float) and scale == 1.0):
                g = g * scale
            q = g.square().mean(1)
            v_old = v_all[a:a + CHUNK_ROWS]
            if self.kind == "rowrms":
                v = torch.where(t, b2 * v_old + (1 - b2) * q, v_old)
                n = n_all[a:a + CHUNK_ROWS]
                n += t.to(torch.int32)
                step = torch.where(t, self._scale(group, v, n.clamp_min(1)), 0.0)
            else:
                v = torch.where(t, v_old + q, v_old)
                step = torch.where(t, self._scale(group, v, None), 0.0)
            v_old.copy_(v)
            p[a:a + CHUNK_ROWS].addcmul_(step[:, None], g, value=-1.0)
        acc.zero_()
        touched.zero_()

    def zero_grad(self, set_to_none=True):
        pass                                   # the accumulator is cleared in step(), as in LazyRowAdam

    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

    def flush(self):
        """Write the hot rows' state (GPU, slot order) back into the per-row state of the whole table."""
        for p in self._params():
            state, hot = self.state[p], getattr(row_store(p), "hot", None)
            if hot is not None and "hot_v" in state:
                rows = hot.rows.cpu()
                state["v"].index_copy_(0, rows, state["hot_v"].cpu())
                if "hot_n" in state:
                    state["n"].index_copy_(0, rows, state["hot_n"].cpu())

    def state_dict(self):
        """The state of every row in table order (hot rows written back first), so it loads with or without HOT."""
        self.flush()
        params = self._params()
        return {"format": "smlm.row_wise.v1", "kind": self.kind, "shapes": [list(p.shape) for p in params],
                "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in self.param_groups],
                "state": [{k: v for k, v in self.state[p].items() if not k.startswith("hot_")} for p in params]}

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "smlm.row_wise.v1" or state_dict.get("kind") != self.kind:
            raise ValueError(f"not a {self.kind} state dict (smlm.row_wise.v1)")
        params = self._params()
        if state_dict["shapes"] != [list(p.shape) for p in params]:
            raise ValueError("row-wise optimizer state was saved for other tables (shapes differ)")
        for g, saved in zip(self.param_groups, state_dict["param_groups"]):
            g.update(saved)
        for p, saved in zip(params, state_dict["state"]):
            if saved:
                self.state[p] = {k: (v.detach().clone() if k == "step" else v.to(device=p.device, copy=True))
                                 for k, v in saved.items()}


def state_bytes(kind, rows):
    """Persistent optimizer state of one table."""
    return rows * (8 if kind == "rowrms" else 4) + 4

