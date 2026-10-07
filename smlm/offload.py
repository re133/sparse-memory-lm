"""Step 2: the value table of a trained memory model outside the GPU memory (inference only).

Variants (see REPORT.md, step 2):
  a) table in GPU memory: bf16 or 4 bit (kernels.bag_infer reads it directly; decode graphs possible)
  b) HostTable: table in RAM (bf16 or fp32); per memory-layer call only the rows read by the batch go to the GPU
  c) MmapQ4Table: 4-bit table as a file on the NVMe, opened with mmap; RAM cache of frequently read rows
     (static: the rows read most during training) plus an optional FIFO part that keeps recently missed rows

b) and c) fetch the distinct rows of a call, copy them into a compact table on the GPU and run the same bag kernel
as a) on it with re-mapped indices. Every output is therefore computed from the same numbers in the same order as
with the table in GPU memory (a-bf16 = b-bf16, a-q4 = c), only the place the rows come from differs. The fetch needs
the indices on the CPU, i.e. one GPU->CPU synchronisation per memory layer and call; decode graphs are not possible.

Use load_model(rest_path, table) -> Transformer whose memory layers read from `table`.
"""
import mmap
import os
import time

import numpy as np
import torch
from torch import nn

from .kernels import bag_infer
from .model import ModelConfig, RotaryEmbedding, Transformer

PAGE = mmap.PAGESIZE


class OffloadTable:
    """Common part: unique rows -> fetch (subclass) -> pinned staging -> compact GPU table -> bag kernel."""

    def __init__(self, rows, dim, q4, dtype, device="cuda"):
        self.rows, self.dim, self.q4, self.dtype, self.device = rows, dim, q4, dtype, device
        self.width = dim // 2 if q4 else dim
        self._cap = 0
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"calls": 0, "lookups": 0, "unique_rows": 0, "hits": 0, "misses": 0, "pages_requested": 0,
                      "fetch_s": 0.0, "total_s": 0.0}

    def _buffers(self, n):
        if n > self._cap:
            cap = max(n, 2 * self._cap, 4096)
            self.stage = torch.empty(cap, self.width, dtype=self.dtype, pin_memory=True)
            self.dev = torch.empty(cap, self.width, dtype=self.dtype, device=self.device)
            if self.q4:
                self.stage_s = torch.empty(cap, dtype=torch.float16, pin_memory=True)
                self.dev_s = torch.empty(cap, dtype=torch.float16, device=self.device)
            self._cap = cap

    def fetch(self, rows_np, out, out_scales):
        """Write table rows `rows_np` (sorted, unique, int64) into out[:n] (and out_scales[:n] for 4 bit)."""
        raise NotImplementedError

    def bag(self, indices, weights, pre=None):
        t0 = time.perf_counter()
        uniq, inv = torch.unique(indices.reshape(-1), return_inverse=True)
        u = uniq.cpu().numpy()                                   # synchronisation point
        n = len(u)
        self._buffers(n)
        t1 = time.perf_counter()
        self.fetch(u, self.stage[:n], self.stage_s[:n] if self.q4 else None)
        self.stats["fetch_s"] += time.perf_counter() - t1
        self.dev[:n].copy_(self.stage[:n], non_blocking=True)
        if self.q4:
            self.dev_s[:n].copy_(self.stage_s[:n], non_blocking=True)
        out = bag_infer(inv.view(indices.shape), weights, self.dev[:n], self.dev_s[:n] if self.q4 else None,
                        pre=pre, out_bf16=True)
        st = self.stats
        st["calls"] += 1
        st["lookups"] += indices.numel()
        st["unique_rows"] += n
        st["total_s"] += time.perf_counter() - t0
        return out


class HostTable(OffloadTable):
    """b) The whole table in RAM (pageable; only the staging buffer is pinned)."""

    def __init__(self, table, scales=None, device="cuda"):
        q4 = scales is not None
        super().__init__(table.shape[0], table.shape[1] * (2 if q4 else 1), q4, table.dtype, device)
        self.table, self.scales = table, scales

    def fetch(self, rows_np, out, out_scales):
        idx = torch.from_numpy(rows_np)
        torch.index_select(self.table, 0, idx, out=out)
        if self.q4:
            torch.index_select(self.scales, 0, idx, out=out_scales)
        self.stats["hits"] += len(rows_np)


class MmapQ4Table(OffloadTable):
    """c) 4-bit table file on the NVMe (mmap, readahead off), fp16 scales in RAM, RAM row cache.

    cache_rows: static part, filled once with `hot_rows[:cache_rows]` (most read during training).
    fifo_rows: dynamic part; rows that missed are copied in, the oldest ones are evicted (FIFO).
    Missing rows: all their pages are requested at once with madvise(MADV_WILLNEED) (asynchronous readahead of
    exactly those pages, so the NVMe sees many parallel requests), then copied."""

    def __init__(self, path, scales_path, rows, dim, hot_rows=None, cache_rows=0, fifo_rows=0, device="cuda"):
        super().__init__(rows, dim, True, torch.uint8, device)
        self.path = path
        self.row_bytes = dim // 2
        self.scales = torch.from_numpy(np.fromfile(scales_path, dtype=np.float16))
        assert self.scales.numel() == rows
        self.row2slot = np.full(rows, -1, dtype=np.int32)
        self.n_static = int(cache_rows)
        self.n_fifo = int(fifo_rows)
        self.cache = np.empty((self.n_static + self.n_fifo, self.row_bytes), dtype=np.uint8)
        self.slot2row = np.full(self.n_static + self.n_fifo, -1, dtype=np.int64)
        self.fifo_ptr = 0
        self._open()
        if self.n_static:
            hot = np.sort(hot_rows[:self.n_static])
            for a in range(0, len(hot), 1 << 20):              # sequential-ish read of the hot rows
                part = hot[a:a + (1 << 20)]
                self.cache[a:a + len(part)] = self.mm[part]
                self.row2slot[part] = np.arange(a, a + len(part), dtype=np.int32)
                self.slot2row[a:a + len(part)] = part

    def _open(self):
        self.fd = os.open(self.path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        assert size == self.rows * self.row_bytes, (size, self.rows, self.row_bytes)
        self.map = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        self.map.madvise(mmap.MADV_RANDOM)                       # no readahead around page faults
        self.mm = np.frombuffer(self.map, dtype=np.uint8).reshape(self.rows, self.row_bytes)

    def close(self):
        del self.mm
        self.map.close()
        os.close(self.fd)

    def drop_os_cache(self):
        """Cold start without root: unmap, drop the file's pages from the page cache, map again."""
        self.close()
        fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        self._open()

    def fetch(self, rows_np, out, out_scales):
        slots = self.row2slot[rows_np]
        hit = slots >= 0
        o = out.numpy()
        o[hit] = self.cache[slots[hit]]
        miss_rows = rows_np[~hit]
        if len(miss_rows):
            first = miss_rows * self.row_bytes // PAGE
            last = (miss_rows * self.row_bytes + self.row_bytes - 1) // PAGE
            pages = np.unique(np.concatenate([first, last]))
            # coalesce consecutive pages into ranges, one madvise per range
            brk = np.flatnonzero(np.diff(pages) != 1) + 1
            starts = pages[np.r_[0, brk]]
            ends = pages[np.r_[brk - 1, len(pages) - 1]] + 1
            for s, e in zip(starts.tolist(), ends.tolist()):
                self.map.madvise(mmap.MADV_WILLNEED, s * PAGE, (e - s) * PAGE)
            data = self.mm[miss_rows]                            # waits for the pages
            o[~hit] = data
            self.stats["pages_requested"] += len(pages)
            if self.n_fifo:
                k = min(len(miss_rows), self.n_fifo)
                new_rows, new_data = miss_rows[-k:], data[-k:]
                pos = self.n_static + (self.fifo_ptr + np.arange(k)) % self.n_fifo
                old = self.slot2row[pos]
                self.row2slot[old[old >= 0]] = -1
                self.row2slot[new_rows] = pos.astype(np.int32)
                self.slot2row[pos] = new_rows
                self.cache[pos] = new_data
                self.fifo_ptr = (self.fifo_ptr + k) % self.n_fifo
        torch.index_select(self.scales, 0, torch.from_numpy(rows_np), out=out_scales)
        self.stats["hits"] += int(hit.sum())
        self.stats["misses"] += int(len(miss_rows))


class DirectQ4Table(OffloadTable):
    """4-bit table read with O_DIRECT, with the same static/FIFO row cache as MmapQ4Table.

    Use DirectQ4Table(path, scales_path, rows, dim, backend="pread", threads=16).
    The optional "uring" backend uses queue_depth instead of threads. Both request O_DIRECT reads.
    """

    def __init__(self, path, scales_path, rows, dim, hot_rows=None, cache_rows=0, fifo_rows=0, device="cuda",
                 *, backend="pread", threads=16, queue_depth=128):
        super().__init__(rows, dim, True, torch.uint8, device)
        self.path = path
        self.row_bytes = dim // 2
        self.backend, self.threads, self.queue_depth = backend, threads, queue_depth
        self.scales = torch.from_numpy(np.fromfile(scales_path, dtype=np.float16))
        assert self.scales.numel() == rows
        self.row2slot = np.full(rows, -1, dtype=np.int32)
        self.n_static = int(cache_rows)
        self.n_fifo = int(fifo_rows)
        self.cache = np.empty((self.n_static + self.n_fifo, self.row_bytes), dtype=np.uint8)
        self.slot2row = np.full(self.n_static + self.n_fifo, -1, dtype=np.int64)
        self.fifo_ptr = 0
        self.reader = None
        self._open()
        try:
            if self.n_static:
                hot = np.sort(hot_rows[:self.n_static])
                # Bound temporary page buffers when the hot rows are scattered over a large table.
                for a in range(0, len(hot), 8192):
                    part = hot[a:a + 8192]
                    self.cache[a:a + len(part)] = self.reader.fetch(part)
                    self.row2slot[part] = np.arange(a, a + len(part), dtype=np.int32)
                    self.slot2row[a:a + len(part)] = part
        except BaseException:
            self.close()
            raise

    def _open(self):
        from .table_io import PreadReader, UringReader

        if self.backend == "pread":
            reader = PreadReader(self.path, self.row_bytes, threads=self.threads)
        elif self.backend == "uring":
            reader = UringReader(self.path, self.row_bytes, queue_depth=self.queue_depth)
        else:
            raise ValueError(f"unknown direct I/O backend: {self.backend!r}")
        if reader.size != self.rows * self.row_bytes:
            reader.close()
            raise ValueError(f"table size {reader.size} does not match {self.rows} rows of {self.row_bytes} bytes")
        self.reader = reader

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def drop_os_cache(self):
        """Drop clean file pages without changing the static/FIFO cache or its statistics."""
        self.close()
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        self._open()

    def fetch(self, rows_np, out, out_scales):
        if self.reader is None:
            raise RuntimeError("DirectQ4Table is closed")
        slots = self.row2slot[rows_np]
        hit = slots >= 0
        o = out.numpy()
        o[hit] = self.cache[slots[hit]]
        miss_rows = rows_np[~hit]
        if len(miss_rows):
            data = self.reader.fetch(miss_rows)
            o[~hit] = data
            self.stats["pages_requested"] += self.reader.last_pages
            if self.n_fifo:
                k = min(len(miss_rows), self.n_fifo)
                new_rows, new_data = miss_rows[-k:], data[-k:]
                pos = self.n_static + (self.fifo_ptr + np.arange(k)) % self.n_fifo
                old = self.slot2row[pos]
                self.row2slot[old[old >= 0]] = -1
                self.row2slot[new_rows] = pos.astype(np.int32)
                self.slot2row[pos] = new_rows
                self.cache[pos] = new_data
                self.fifo_ptr = (self.fifo_ptr + k) % self.n_fifo
        torch.index_select(self.scales, 0, torch.from_numpy(rows_np), out=out_scales)
        self.stats["hits"] += int(hit.sum())
        self.stats["misses"] += int(len(miss_rows))


def load_model(rest_path, table=None, device="cuda"):
    """Model from rest.pt (scripts/convert_table.py) without allocating the fp32 table.
    table: OffloadTable (variants b, c) or a tuple (gpu_table, gpu_scales_or_None) (variant a)."""
    ck = torch.load(rest_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**ck["model_config"])
    with torch.device("meta"):
        model = Transformer(cfg)
    mems = model.memory_layers()
    shared = mems[0].values
    rows, dim = ck["table_shape"]
    shared.weight = nn.Parameter(torch.empty(0, dim, device="meta"), requires_grad=False)
    model = model.to_empty(device=device)
    for mod in model.modules():                                  # non-persistent RoPE buffers
        if hasattr(mod, "rope") and isinstance(mod.rope, RotaryEmbedding):
            mod.rope = RotaryEmbedding(mod.head_dim, cfg.max_seq_len, cfg.rope_theta).to(device)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected, unexpected
    assert set(missing) == set(ck["table_keys"]), missing
    shared.infer_table = table
    shared.table_shape = (rows, dim)
    model.eval()
    return model
