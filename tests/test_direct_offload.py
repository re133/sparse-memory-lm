"""CPU checks for the direct-I/O table and its mmap-compatible row cache.

Run: python -m pytest -q tests/test_direct_offload.py
"""
import errno

import numpy as np
import pytest
import torch

from smlm.offload import DirectQ4Table, MmapQ4Table


@pytest.fixture
def q4_file(tmp_path):
    rng = np.random.default_rng(7)
    packed = rng.integers(0, 256, size=(1293, 192), dtype=np.uint8)
    scales = rng.uniform(0.1, 2.0, size=len(packed)).astype(np.float16)
    q, s = tmp_path / "values_q4.bin", tmp_path / "scales_q4.bin"
    packed.tofile(q)
    scales.tofile(s)
    return q, s, packed, scales


def _direct(q, s, packed, backend, **kwargs):
    try:
        return DirectQ4Table(q, s, len(packed), packed.shape[1] * 2, device="cpu", backend=backend,
                             threads=4, queue_depth=32, **kwargs)
    except OSError as exc:
        unsupported = {errno.ENOSYS, errno.EOPNOTSUPP}
        if backend == "uring":
            unsupported.add(errno.EPERM)
        if exc.errno in unsupported:
            pytest.skip(f"{backend} unavailable on this host/filesystem: {exc}")
        raise


def _same_cache(direct, mmap):
    assert direct.stats == mmap.stats
    assert direct.fifo_ptr == mmap.fifo_ptr
    np.testing.assert_array_equal(direct.row2slot, mmap.row2slot)
    np.testing.assert_array_equal(direct.slot2row, mmap.slot2row)
    filled = mmap.slot2row >= 0
    np.testing.assert_array_equal(direct.cache[filled], mmap.cache[filled])


def _fetch_same(direct, mmap, rows, packed, scales):
    outputs = []
    for table in (direct, mmap):
        out = torch.empty(len(rows), packed.shape[1], dtype=torch.uint8)
        out_s = torch.empty(len(rows), dtype=torch.float16)
        table.fetch(rows, out, out_s)
        np.testing.assert_array_equal(out.numpy(), packed[rows])
        np.testing.assert_array_equal(out_s.numpy(), scales[rows])
        outputs.append((out, out_s))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    _same_cache(direct, mmap)


@pytest.mark.parametrize("backend", ["pread", "uring"])
@pytest.mark.parametrize("cache,fifo", [(0, 0), (19, 0), (0, 17), (19, 17)])
def test_direct_fetch_matches_mmap(q4_file, backend, cache, fifo):
    q, s, packed, scales = q4_file
    hot = np.random.default_rng(11).permutation(len(packed))
    kwargs = {"hot_rows": hot, "cache_rows": cache, "fifo_rows": fifo}
    direct = _direct(q, s, packed, backend, **kwargs)
    mmap = MmapQ4Table(q, s, len(packed), packed.shape[1] * 2, device="cpu", **kwargs)
    rng = np.random.default_rng(13)
    try:
        _same_cache(direct, mmap)
        # Include straddling rows and the short final page of an unpadded file.
        _fetch_same(direct, mmap, np.array([0, 21, 42, len(packed) - 1]), packed, scales)
        _fetch_same(direct, mmap, np.empty(0, dtype=np.int64), packed, scales)
        if cache:
            _fetch_same(direct, mmap, np.sort(hot[:cache]), packed, scales)
        for iteration in range(20):
            rows = np.unique(rng.integers(0, len(packed), size=rng.integers(1, 150)))
            _fetch_same(direct, mmap, rows, packed, scales)
            if fifo:
                filled = mmap.slot2row[mmap.slot2row >= 0]
                _fetch_same(direct, mmap, np.sort(filled), packed, scales)
            if iteration == 9:
                direct.drop_os_cache()
                mmap.drop_os_cache()
                _same_cache(direct, mmap)
            if iteration == 14:
                direct.reset_stats()
                mmap.reset_stats()
                _same_cache(direct, mmap)
        assert direct.stats["misses"] > 0
        if cache or fifo:
            assert direct.stats["hits"] > 0
    finally:
        direct.close()
        mmap.close()


def test_direct_cache_hits_do_not_read(q4_file, monkeypatch):
    q, s, packed, scales = q4_file
    hot = np.array([0, 21, 42, len(packed) - 1], dtype=np.int64)
    direct = _direct(q, s, packed, "pread", hot_rows=hot, cache_rows=len(hot), fifo_rows=3)
    try:
        fifo_rows = np.array([1, 2, 3], dtype=np.int64)
        out = torch.empty(len(fifo_rows), packed.shape[1], dtype=torch.uint8)
        out_s = torch.empty(len(fifo_rows), dtype=torch.float16)
        direct.fetch(fifo_rows, out, out_s)

        def fail_read(rows):
            pytest.fail("an all-hit batch must not issue direct I/O")

        monkeypatch.setattr(direct.reader, "fetch", fail_read)
        before = dict(direct.stats)
        rows = np.sort(np.concatenate([hot, fifo_rows]))
        out = torch.empty(len(rows), packed.shape[1], dtype=torch.uint8)
        out_s = torch.empty(len(rows), dtype=torch.float16)
        direct.fetch(rows, out, out_s)
        np.testing.assert_array_equal(out.numpy(), packed[rows])
        np.testing.assert_array_equal(out_s.numpy(), scales[rows])
        assert direct.stats["hits"] == before["hits"] + len(rows)
        assert direct.stats["misses"] == before["misses"]
        assert direct.stats["pages_requested"] == before["pages_requested"]
    finally:
        direct.close()
    direct.close()
    with pytest.raises(RuntimeError, match="closed"):
        direct.fetch(np.empty(0, dtype=np.int64), torch.empty(0, 192, dtype=torch.uint8),
                     torch.empty(0, dtype=torch.float16))


def test_direct_rejects_unknown_backend(q4_file):
    q, s, packed, _ = q4_file
    with pytest.raises(ValueError, match="unknown direct I/O backend"):
        DirectQ4Table(q, s, len(packed), packed.shape[1] * 2, backend="unknown", device="cpu")


@pytest.mark.parametrize("failure", ["size", "static"])
def test_direct_init_failure_closes_reader(q4_file, monkeypatch, failure):
    from smlm import table_io

    q, s, packed, _ = q4_file
    readers = []

    class FailingReader:
        def __init__(self, path, row_bytes, threads):
            self.size = packed.nbytes + (1 if failure == "size" else 0)
            self.closed = False
            readers.append(self)

        def fetch(self, rows):
            raise OSError(errno.EIO, "injected read failure")

        def close(self):
            self.closed = True

    monkeypatch.setattr(table_io, "PreadReader", FailingReader)
    error = ValueError if failure == "size" else OSError
    with pytest.raises(error):
        DirectQ4Table(q, s, len(packed), packed.shape[1] * 2, hot_rows=np.array([0]), cache_rows=1,
                      device="cpu", backend="pread")
    assert len(readers) == 1
    assert readers[0].closed
