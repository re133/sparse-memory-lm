"""CPU row readers for a flat byte table, with no torch dependency.

Use PreadReader(path, row_bytes, threads=16).fetch(rows), or UringReader(..., queue_depth=128).
"""
import ctypes
import errno
import hashlib
import mmap
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import numpy as np

PAGE = 4096


@dataclass
class PagePlan:
    pages: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    buffer_starts: np.ndarray


def page_plan(rows_np, row_bytes):
    """Sorted unique pages and coalesced ranges; buffer_starts indexes compact page storage."""
    if not 0 < row_bytes <= PAGE:
        raise ValueError("row_bytes must be between 1 and 4096")
    rows = np.asarray(rows_np, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0):
        raise ValueError("rows must be a one-dimensional array of nonnegative indices")
    if not len(rows):
        empty = np.empty(0, dtype=np.int64)
        return PagePlan(empty, empty, empty, empty)
    first = rows * row_bytes // PAGE
    last = (rows * row_bytes + row_bytes - 1) // PAGE
    pages = np.unique(np.concatenate([first, last]))
    brk = np.flatnonzero(np.diff(pages) != 1) + 1
    positions = np.r_[0, brk]
    return PagePlan(pages, pages[positions], pages[np.r_[brk - 1, len(pages) - 1]] + 1, positions)


class _Reader:
    def __init__(self, path, row_bytes, direct=False):
        if not 0 < row_bytes <= PAGE:
            raise ValueError("row_bytes must be between 1 and 4096")
        self.path, self.row_bytes = os.fspath(path), int(row_bytes)
        self.fd = None
        self.last_pages = self.last_ranges = 0
        flags = os.O_RDONLY | os.O_CLOEXEC
        if direct:
            if not hasattr(os, "O_DIRECT"):
                raise OSError(errno.ENOTSUP, "O_DIRECT is not available")
            flags |= os.O_DIRECT
        self.fd = os.open(self.path, flags)
        try:
            self.size = os.fstat(self.fd).st_size
            if self.size == 0 or self.size % self.row_bytes:
                raise ValueError("table size must be a positive multiple of row_bytes")
            self.rows = self.size // self.row_bytes
        except BaseException:
            self.close()
            raise

    def _rows(self, rows_np):
        if self.fd is None:
            raise ValueError("reader is closed")
        rows = np.asarray(rows_np)
        if rows.ndim != 1 or (rows.size and not np.issubdtype(rows.dtype, np.integer)):
            raise ValueError("rows must be a one-dimensional integer array")
        if rows.size and (np.any(rows < 0) or np.any(rows >= self.rows)):
            raise IndexError("table row out of bounds")
        return rows.astype(np.int64, copy=False)

    def _plan(self, rows):
        plan = page_plan(rows, self.row_bytes)
        self.last_pages, self.last_ranges = len(plan.pages), len(plan.starts)
        return plan

    def reserve(self, n_pages):
        """Allocate scratch space before timing (mmap needs no scratch space)."""

    def drop_os_cache(self):
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_DONTNEED)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class MmapReader(_Reader):
    """The existing MADV_RANDOM + coalesced MADV_WILLNEED + row-copy path."""

    def __init__(self, path, row_bytes):
        self.map = self.mm = None
        super().__init__(path, row_bytes)
        try:
            self._map()
        except BaseException:
            self.close()
            raise

    def _map(self):
        self.map = mmap.mmap(self.fd, self.size, prot=mmap.PROT_READ)
        self.map.madvise(mmap.MADV_RANDOM)
        self.mm = np.frombuffer(self.map, dtype=np.uint8).reshape(self.rows, self.row_bytes)

    def _unmap(self):
        self.mm = None
        if self.map is not None:
            self.map.close()
            self.map = None

    def drop_os_cache(self):
        self._unmap()
        super().drop_os_cache()
        self._map()

    def fetch(self, rows_np):
        rows = self._rows(rows_np)
        plan = self._plan(rows)
        for start, end in zip(plan.starts.tolist(), plan.ends.tolist()):
            self.map.madvise(mmap.MADV_WILLNEED, start * PAGE, (end - start) * PAGE)
        return self.mm[rows]

    def close(self):
        self._unmap()
        super().close()


class _DirectReader(_Reader):
    def __init__(self, path, row_bytes):
        self._buffer = None
        self._capacity = 0
        super().__init__(path, row_bytes, direct=True)

    def reserve(self, n_pages):
        if n_pages <= self._capacity:
            return
        capacity = max(int(n_pages), 2 * self._capacity)
        buffer = mmap.mmap(-1, capacity * PAGE)
        try:
            # First-touch cost belongs to setup, not to random disk latency.
            np.frombuffer(buffer, dtype=np.uint8)[::PAGE] = 0
        except BaseException:
            buffer.close()
            raise
        if self._buffer is not None:
            self._buffer.close()
        self._capacity, self._buffer = capacity, buffer

    def _copy_rows(self, rows, plan):
        if not len(rows):
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        byte_offsets = rows * self.row_bytes
        offsets = np.searchsorted(plan.pages, byte_offsets // PAGE) * PAGE + byte_offsets % PAGE
        # Adjacent file pages remain adjacent in compact storage, including straddling rows.
        # A strided view avoids an n_rows * row_bytes array of integer gather indices.
        windows = np.ndarray((len(plan.pages) * PAGE - self.row_bytes + 1, self.row_bytes),
                             dtype=np.uint8, buffer=self._buffer, strides=(1, 1))
        return windows[offsets]

    def fetch(self, rows_np):
        rows = self._rows(rows_np)
        plan = self._plan(rows)
        if len(rows):
            self.reserve(len(plan.pages))
            self._read(plan)
        return self._copy_rows(rows, plan)

    def close(self):
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
            self._capacity = 0
        super().close()


class PreadReader(_DirectReader):
    """O_DIRECT preadv into aligned reusable buffers, with bounded worker dispatch."""

    def __init__(self, path, row_bytes, threads=16):
        if int(threads) < 1:
            raise ValueError("threads must be positive")
        self.threads = int(threads)
        self.pool = None
        super().__init__(path, row_bytes)
        try:
            if self.threads > 1:
                self.pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="table-io")
                # Executors create threads lazily; keep that one-time cost out of the first disk batch.
                ready = threading.Barrier(self.threads + 1)
                futures = []
                try:
                    for _ in range(self.threads):
                        futures.append(self.pool.submit(ready.wait))
                    ready.wait()
                except BaseException:
                    ready.abort()
                    raise
                finally:
                    wait(futures)
                for future in futures:
                    future.result()
        except BaseException:
            self.close()
            raise

    def _worker(self, requests, worker, workers):
        with memoryview(self._buffer) as buffer:
            for offset, length, target in requests[worker::workers]:
                with buffer[target:target + length] as view:
                    while True:
                        try:
                            count = os.preadv(self.fd, [view], offset)
                            break
                        except InterruptedError:
                            continue
                # A final partial page is legal for an unpadded table. Any other short read is not.
                expected = min(length, self.size - offset)
                if count != expected:
                    raise OSError(errno.EIO, f"short direct read at {offset}: {count}, expected {expected}")

    def _read(self, plan):
        requests = [(int(start) * PAGE, int(end - start) * PAGE, int(target) * PAGE)
                    for start, end, target in zip(plan.starts, plan.ends, plan.buffer_starts)]
        workers = min(self.threads, len(requests))
        if workers == 1:
            self._worker(requests, 0, 1)
            return
        futures = []
        try:
            for worker in range(workers):
                futures.append(self.pool.submit(self._worker, requests, worker, workers))
        finally:
            # Other tasks must stop using the shared buffer even when one read or submit fails.
            wait(futures)
        for future in futures:
            future.result()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None
        super().close()


def _uring_library():
    source = Path(__file__).resolve().parents[1] / "scripts" / "table_io_uring.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = source.parent.parent / ".cache" / "table_io"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"table_io_uring_{digest}.so"
    if not library.exists():
        fd, temporary = tempfile.mkstemp(prefix="uring-", suffix=".so", dir=cache)
        os.close(fd)
        try:
            result = subprocess.run(["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
                                     "-fPIC", str(source), "-o", temporary, "-luring"],
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"cannot compile io_uring helper: {result.stderr.strip()}")
            os.replace(temporary, library)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(library))
    lib.table_io_open.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    lib.table_io_open.restype = ctypes.c_int
    lib.table_io_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    lib.table_io_read.restype = ctypes.c_int
    lib.table_io_close.argtypes = [ctypes.c_void_p]
    lib.table_io_close.restype = None
    return lib


class UringReader(_DirectReader):
    """O_DIRECT reads through a persistent liburing ring; no registered-buffer requirement."""

    def __init__(self, path, row_bytes, queue_depth=128):
        if not 0 < int(queue_depth) <= 32768:
            raise ValueError("queue_depth must be between 1 and 32768")
        self.queue_depth = int(queue_depth)
        self._lib, self._ring = None, ctypes.c_void_p()
        super().__init__(path, row_bytes)
        try:
            self._lib = _uring_library()
            result = self._lib.table_io_open(self.queue_depth, ctypes.byref(self._ring))
            if result < 0:
                raise OSError(-result, f"io_uring setup: {os.strerror(-result)}")
        except BaseException:
            self.close()
            raise

    def _read(self, plan):
        offsets = np.ascontiguousarray(plan.starts * PAGE, dtype=np.uint64)
        lengths = np.ascontiguousarray((plan.ends - plan.starts) * PAGE, dtype=np.uint64)
        targets = np.ascontiguousarray(plan.buffer_starts * PAGE, dtype=np.uint64)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
        result = self._lib.table_io_read(self._ring, self.fd, address, offsets.ctypes.data,
                                         lengths.ctypes.data, targets.ctypes.data, len(offsets), self.size)
        if result < 0:
            raise OSError(-result, f"io_uring read: {os.strerror(-result)}")

    def close(self):
        if self._ring.value is not None:
            self._lib.table_io_close(self._ring)
            self._ring = ctypes.c_void_p()
        super().close()
