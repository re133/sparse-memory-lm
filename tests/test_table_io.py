"""CPU table I/O and benchmark checks.

Run python -m pytest -q tests/test_table_io.py.
"""
import ctypes
import errno
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from smlm.table_io import PAGE, MmapReader, PreadReader, UringReader, page_plan

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_table_io", ROOT / "scripts" / "bench_table_io.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
UNAVAILABLE = {errno.ENOSYS, errno.EOPNOTSUPP}


@pytest.fixture
def table(tmp_path):
    data = np.random.default_rng(451).integers(0, 256, size=(513, 192), dtype=np.uint8)
    path = tmp_path / "rows.bin"
    data.tofile(path)
    return path, data


def reader_or_skip(kind, path, width, **kwargs):
    try:
        return kind(path, width, **kwargs)
    except OSError as exc:
        if exc.errno in UNAVAILABLE or kind is UringReader and exc.errno == errno.EPERM:
            pytest.skip(f"{kind.__name__} unavailable: {exc}")
        raise


def test_page_plan_straddles_coalescing_and_empty():
    # The first straddling row reaches page 1; row 200 leaves a gap before page 9.
    plan = page_plan(np.array([0, 21, 42, 200], dtype=np.int64), 192)
    assert plan.pages.tolist() == [0, 1, 2, 9]
    assert plan.starts.tolist() == [0, 9]
    assert plan.ends.tolist() == [3, 10]
    assert plan.buffer_starts.tolist() == [0, 3]
    empty = page_plan([], 192)
    assert all(array.size == 0 for array in (empty.pages, empty.starts, empty.ends, empty.buffer_starts))
    with pytest.raises(ValueError):
        page_plan([0], PAGE + 1)


@pytest.mark.parametrize("kind,options", [(MmapReader, {}), (PreadReader, {"threads": 1}),
                                         (PreadReader, {"threads": 4}),
                                         (UringReader, {"queue_depth": 1}),
                                         (UringReader, {"queue_depth": 32})])
def test_readers_rows_straddles_partial_eof_and_ownership(table, kind, options):
    path, expected = table
    reader = reader_or_skip(kind, path, expected.shape[1], **options)
    try:
        batches = [np.array([0, 21, 42, 200, 512]),
                   np.sort(np.random.default_rng(12).choice(len(expected), 101, replace=False)),
                   np.arange(len(expected))]
        previous = None
        for rows in batches:
            try:
                actual = reader.fetch(rows)
            except OSError as exc:
                if exc.errno in UNAVAILABLE and kind is not MmapReader:
                    pytest.skip(f"direct reads unavailable on test filesystem: {exc}")
                raise
            np.testing.assert_array_equal(actual, expected[rows])
            pages = np.unique(np.r_[rows * 192 // PAGE, (rows * 192 + 191) // PAGE])
            assert reader.last_pages == len(pages)
            assert reader.last_ranges == 1 + np.count_nonzero(np.diff(pages) != 1)
            if previous is not None:
                np.testing.assert_array_equal(previous[0], previous[1])
            previous = actual, expected[rows].copy()
        assert reader.fetch([]).shape == (0, 192)
        assert reader.last_pages == reader.last_ranges == 0
    finally:
        reader.close()
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.fetch([0])


def test_pread_alignment_interruption_and_short_read_cleanup(table, monkeypatch):
    path, expected = table
    reader = reader_or_skip(PreadReader, path, 192, threads=1)
    original = os.preadv
    calls = []

    def interrupted_once(fd, buffers, offset):
        view, = buffers
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        assert address % PAGE == offset % PAGE == len(view) % PAGE == 0
        calls.append(offset)
        if len(calls) == 1:
            raise InterruptedError(errno.EINTR, "test interruption")
        return original(fd, buffers, offset)

    try:
        monkeypatch.setattr(os, "preadv", interrupted_once)
        try:
            actual = reader.fetch(np.array([0, 21, 512]))
        except OSError as exc:
            if exc.errno in UNAVAILABLE:
                pytest.skip(f"direct reads unavailable on test filesystem: {exc}")
            raise
        np.testing.assert_array_equal(actual, expected[[0, 21, 512]])
        assert calls[0] == calls[1]
        monkeypatch.setattr(os, "preadv", lambda *args: 0)
        with pytest.raises(OSError) as caught:
            reader.fetch(np.array([0]))
        assert caught.value.errno == errno.EIO
    finally:
        # All exported memory views must be released even on interrupted/short reads.
        reader.close()


@pytest.mark.parametrize("rows,error", [([-1], IndexError), ([513], IndexError),
                                       ([0.5], ValueError), ([[0]], ValueError)])
def test_invalid_rows_fail_before_io(table, rows, error, monkeypatch):
    path, _ = table
    with reader_or_skip(PreadReader, path, 192, threads=1) as reader:
        def unexpected_read(*args):
            pytest.fail("invalid rows reached preadv")

        monkeypatch.setattr(os, "preadv", unexpected_read)
        with pytest.raises(error):
            reader.fetch(rows)


def test_benchmark_output_cannot_alias_input(table, tmp_path):
    path, _ = table
    alias = tmp_path / "alias.bin"
    os.link(path, alias)
    with pytest.raises(SystemExit):
        BENCH.parse_args(["--table", str(path), "--out", str(alias)])


def test_cold_benchmark_json_and_no_torch():
    # tmpfs retains all pages, so mincore/DONTNEED needs a disk-backed worktree fixture.
    scratch = ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="table-io-test-", dir=scratch) as directory:
        directory = Path(directory)
        table = directory / "rows.bin"
        expected = np.random.default_rng(93).integers(0, 256, size=(1025, 192), dtype=np.uint8)
        expected.tofile(table)
        fd = os.open(table, os.O_RDONLY)
        try:
            # DONTNEED does not evict dirty fixture pages.
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            if BENCH.resident_pages(fd, table.stat().st_size):
                pytest.skip("test filesystem cannot evict this clean fixture")
        finally:
            os.close(fd)
        # tmpfs may ignore O_DIRECT. Exercise the unpadded EOF on this disk-backed file too.
        with reader_or_skip(PreadReader, table, 192, threads=4) as reader:
            rows = np.array([0, 21, 42, len(expected) - 1])
            np.testing.assert_array_equal(reader.fetch(rows), expected[rows])
        out = directory / "result.json"
        wrapper = ("import runpy, sys; "
                   "module=runpy.run_path('scripts/bench_table_io.py'); "
                   "assert 'torch' not in sys.modules; "
                   "code=module['main'](); "
                   "assert 'torch' not in sys.modules; sys.exit(code)")
        command = [sys.executable, "-c", wrapper, "--quick", "--table", str(table),
                   "--batch-sizes", "7", "70", "--batches", "2", "--threads", "1", "4",
                   "--queue-depths", "1", "32", "--out", str(out)]
        proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert proc.returncode in (0, 1), proc.stdout + proc.stderr
        result = json.loads(out.read_text())
        assert result["page_bytes"] == PAGE
        assert result["table_rows"] == len(expected)
        assert len(result["results"]) == 10
        assert result["kernel"] and "device" in result
        for config in result["results"]:
            if config["status"] == "unavailable":
                assert config["method"] in ("pread", "uring")
                assert (config["errno"] in UNAVAILABLE or
                        config["method"] == "uring" and config["errno"] == errno.EPERM)
                continue
            assert config["status"] == "ok", config
            assert config["correct"]
            assert config["batches"] == 2
            assert config["mb_per_s"] == pytest.approx(config["pages_per_s"] * PAGE / 1e6)
            assert config["useful_mb_per_s"] == pytest.approx(config["rows_per_s"] * 192 / 1e6)
            assert 0 < config["batch_latency_p50_ms"] <= config["batch_latency_p99_ms"]
            checksums = result["reference_sha256"][str(config["batch_rows"])]
            for sample, checksum in zip(config["samples"], checksums):
                assert sample["sha256"] == checksum
                assert sample["resident_pages_before"] == 0
                assert sample["rows"] == config["batch_rows"]
        assert result["resident_pages_at_exit"] == 0


# Kernel io_uring can be forbidden by a sandbox. This stub exercises completion accounting and
# draining failures, without pretending to validate the kernel ABI or asynchronous storage I/O.
URING_STUB = r"""
#include <errno.h>
#include <stdint.h>
#include <string.h>
struct io_uring_sqe { void *buffer; unsigned length; uint64_t offset, data; };
struct io_uring_cqe { int res; uint64_t data; };
struct io_uring {
    struct io_uring_sqe sq[64]; struct io_uring_cqe cq[64];
    unsigned depth, sh, st, ch, ct;
};
static int mode, submits, waits, completed, outstanding, exited_busy;
void stub_mode(int value) { mode=value; submits=waits=completed=outstanding=exited_busy=0; }
int stub_outstanding(void) { return outstanding; }
int stub_exited_busy(void) { return exited_busy; }
int io_uring_queue_init(unsigned depth, struct io_uring *r, unsigned flags) {
    (void)flags; memset(r, 0, sizeof(*r)); r->depth=depth; return 0;
}
void io_uring_queue_exit(struct io_uring *r) { exited_busy += r->ct != r->ch; }
struct io_uring_sqe *io_uring_get_sqe(struct io_uring *r) {
    if (r->st-r->sh == r->depth) return 0;
    return &r->sq[r->st++ % 64];
}
void io_uring_prep_read(struct io_uring_sqe *s, int fd, void *buffer, unsigned length, uint64_t offset) {
    (void)fd; s->buffer=buffer; s->length=length; s->offset=offset;
}
void io_uring_sqe_set_data64(struct io_uring_sqe *s, uint64_t data) { s->data=data; }
uint64_t io_uring_cqe_get_data64(struct io_uring_cqe *c) { return c->data; }
int io_uring_submit(struct io_uring *r) {
    submits++;
    if (mode==1 && submits==1) return -EINTR;
    if (mode==4 && submits==2) return -EIO;
    if (mode==8) return 0;
    unsigned count=r->st-r->sh;
    if ((mode==1 && submits==2) || (mode==4 && submits==1)) count=1;
    for (unsigned i=0; i<count; i++) {
        struct io_uring_sqe *s=&r->sq[r->sh++ % 64];
        struct io_uring_cqe *c=&r->cq[r->ct++ % 64];
        memset(s->buffer, (int)(s->offset/4096+17), s->length);
        c->data=s->data; c->res=(int)s->length;
        if (!completed && mode==2) c->res=-EINTR;
        if (!completed && mode==3) c->res--;
        if (!completed && mode==6) c->res=-EIO;
        completed++; outstanding++;
    }
    return (int)count;
}
int io_uring_peek_cqe(struct io_uring *r, struct io_uring_cqe **c) {
    if (r->ch==r->ct) return -EAGAIN;
    *c=&r->cq[r->ch % 64]; return 0;
}
int io_uring_wait_cqe(struct io_uring *r, struct io_uring_cqe **c) {
    waits++;
    if (mode==5 && waits==1) return -EIO;
    if (mode==1 && waits==1) return -EINTR;
    return io_uring_peek_cqe(r, c);
}
void io_uring_cqe_seen(struct io_uring *r, struct io_uring_cqe *c) {
    (void)c; r->ch++; outstanding--;
}
"""


@pytest.fixture(scope="module")
def uring_stub(tmp_path_factory):
    if not shutil.which("gcc"):
        pytest.skip("gcc is required to exercise the io_uring C helper")
    directory = tmp_path_factory.mktemp("uring-stub")
    (directory / "liburing.h").write_text(URING_STUB)
    source = directory / "stub.c"
    source.write_text(f'#include "{ROOT / "scripts" / "table_io_uring.c"}"\n')
    library = directory / "stub.so"
    subprocess.run(["gcc", "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-I", str(directory), str(source), "-o", str(library)],
                   capture_output=True, text=True, check=True, timeout=30)
    lib = ctypes.CDLL(str(library))
    lib.table_io_open.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    lib.table_io_read.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    lib.table_io_close.argtypes = [ctypes.c_void_p]
    lib.table_io_close.restype = None
    return lib


@pytest.mark.parametrize("mode,expected,broken", [(0, 0, False), (1, 0, False), (2, 0, False),
                                                 (3, -errno.EIO, False), (4, -errno.EIO, True),
                                                 (5, -errno.EIO, True), (6, -errno.EIO, False),
                                                 (8, -errno.EIO, True)])
def test_uring_completions_and_errors_drain_buffers(uring_stub, mode, expected, broken):
    lib = uring_stub
    lib.stub_mode(mode)
    handle = ctypes.c_void_p()
    assert lib.table_io_open(4, ctypes.byref(handle)) == 0
    offsets = np.arange(12, dtype=np.uint64) * PAGE
    lengths = np.full(12, PAGE, dtype=np.uint64)
    output = np.empty((12, PAGE), dtype=np.uint8)

    def read():
        return lib.table_io_read(handle, -1, output.ctypes.data, offsets.ctypes.data, lengths.ctypes.data,
                                 offsets.ctypes.data, len(offsets), output.nbytes)

    try:
        assert read() == expected
        assert lib.stub_outstanding() == 0
        assert lib.stub_exited_busy() == 0
        if expected == 0:
            np.testing.assert_array_equal(output, np.repeat((np.arange(12) + 17)[:, None], PAGE, axis=1))
        # A failed completion can preserve the ring; a failed submit/wait invalidates it.
        assert read() == (-errno.EBADF if broken else 0)
        assert lib.stub_outstanding() == 0
    finally:
        lib.table_io_close(handle)
