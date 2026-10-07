"""Cold scattered-row reads, with NumPy and Linux I/O only (no torch or GPU).

  python scripts/bench_table_io.py --quick --out .scratch/table-io-quick.json
  python scripts/bench_table_io.py --table /path/to/values_q4.bin --out .scratch/table-io.json

Every batch starts with verified cold file pages. Timings include page planning, I/O and row copying;
cache eviction, buffer reservation, plain-read references and checksums are outside the timed interval.
Pages/s and MB/s count requested unique pages, not hardware IOPS; useful row MB/s is reported separately.
"""
import argparse
import ctypes
import errno
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smlm.table_io import PAGE, MmapReader, PreadReader, UringReader  # noqa: E402
from smlm.atomic import write_json  # noqa: E402


def resident_pages(fd, size):
    """mincore observes residency without faulting in the read-only file's contents."""
    count = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    vec = (ctypes.c_ubyte * count)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
    libc.mincore.restype = ctypes.c_int
    # ACCESS_COPY permits obtaining an address; nothing is written, even to the private mapping.
    with mmap.mmap(fd, size, access=mmap.ACCESS_COPY) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        if libc.mincore(address, size, vec):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
    return int(np.count_nonzero(np.ctypeslib.as_array(vec) & 1))


def command_output(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}


def device_info(path):
    mount = command_output(["findmnt", "-J", "-T", str(path), "-o", "SOURCE,FSTYPE,OPTIONS,TARGET"])
    if isinstance(mount, str):
        mount = json.loads(mount)
    source = (mount.get("filesystems") or [{}])[0].get("source", "")
    name = os.path.basename(source.split("[")[0])
    match = re.match(r"(nvme\d+n\d+)", name)
    if match:
        name = match.group(1)
    else:
        block = Path("/sys/class/block") / name
        if (block / "partition").exists():
            name = block.resolve().parent.name
    info = {"name": name, "mount": mount, "model": None, "scheduler": None}
    for key, relative in (("model", "device/model"), ("scheduler", "queue/scheduler")):
        try:
            info[key] = (Path("/sys/class/block") / name / relative).read_text().strip()
        except OSError:
            pass
    return info


def disk_stat(name):
    try:
        fields = (Path("/sys/class/block") / name / "stat").read_text().split()
        return {"reads": int(fields[0]), "read_bytes": int(fields[2]) * 512}
    except (OSError, IndexError, ValueError):
        return None


def plain_checksum(fd, rows, row_bytes):
    """Independent reference: no shared page planning or row-gather code."""
    digest = hashlib.sha256()
    for row in rows.tolist():
        data = os.pread(fd, row_bytes, row * row_bytes)
        if len(data) != row_bytes:
            raise OSError("short plain reference read")
        digest.update(data)
    return digest.hexdigest()


def configurations(args):
    configs = []
    for size in args.batch_sizes:
        configs.append({"method": "mmap", "batch_rows": size})
        configs.extend({"method": "pread", "threads": n, "batch_rows": size} for n in args.threads)
        configs.extend({"method": "uring", "queue_depth": n, "batch_rows": size} for n in args.queue_depths)
    # Avoid systematically favouring the last method after the drive has warmed up.
    np.random.default_rng(args.seed).shuffle(configs)
    return configs


def measure(config, args, batches, references, fd, size, device, deadline):
    result = {**config, "status": "ok", "samples": []}
    start = time.perf_counter()
    if config["method"] == "mmap":
        reader = MmapReader(args.table, args.row_bytes)
    elif config["method"] == "pread":
        reader = PreadReader(args.table, args.row_bytes, threads=config["threads"])
    else:
        reader = UringReader(args.table, args.row_bytes, queue_depth=config["queue_depth"])
    try:
        if hasattr(reader, "reserve"):
            reader.reserve(min((size + PAGE - 1) // PAGE, 2 * config["batch_rows"]))
        result["setup_s"] = time.perf_counter() - start
        for batch, expected in zip(batches, references):
            if time.perf_counter() >= deadline:
                result["status"] = "time_limit"
                break
            reader.drop_os_cache()
            before = resident_pages(fd, size)
            if before:
                result.update(status="not_cold", resident_pages_before=before)
                break
            d0 = disk_stat(device)
            t0 = time.perf_counter()
            data = reader.fetch(batch)
            elapsed = time.perf_counter() - t0
            d1 = disk_stat(device)
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise RuntimeError(f"checksum mismatch: {config}, expected {expected}, got {actual}")
            after = resident_pages(fd, size)
            sample = {"seconds": elapsed, "rows": len(batch), "pages": reader.last_pages,
                      "ranges": reader.last_ranges, "sha256": actual, "correct": True,
                      "resident_pages_before": before, "resident_pages_after": after}
            if d0 is not None and d1 is not None:
                sample["shared_device_delta"] = {k: d1[k] - d0[k] for k in d0}
            result["samples"].append(sample)
    finally:
        reader.close()
    samples = result["samples"]
    if samples:
        seconds = sum(s["seconds"] for s in samples)
        rows = sum(s["rows"] for s in samples)
        pages = sum(s["pages"] for s in samples)
        result.update(seconds=seconds, batches=len(samples), rows_per_s=rows / seconds,
                      pages_per_s=pages / seconds, mb_per_s=pages * PAGE / seconds / 1e6,
                      useful_mb_per_s=rows * args.row_bytes / seconds / 1e6,
                      batch_latency_p50_ms=float(np.percentile([s["seconds"] for s in samples], 50) * 1000),
                      batch_latency_p99_ms=float(np.percentile([s["seconds"] for s in samples], 99) * 1000),
                      correct=True,
                      cache_bypass_observed=(all(s["resident_pages_after"] == 0 for s in samples)
                                             if config["method"] != "mmap" else None))
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="/home/leon/smlm-tables/B-16M/values_q4.bin")
    ap.add_argument("--row-bytes", type=int, default=192)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[200, 2000, 20000])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--queue-depths", type=int, nargs="+", default=[1, 32, 128, 256])
    ap.add_argument("--batches", type=int, help="batches per configuration (quick: 5, otherwise: 30)")
    ap.add_argument("--quick", action="store_true", help="short smoke run; p99 has very few samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, help="wall-time budget (quick: 90, otherwise: 540)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.batches = args.batches if args.batches is not None else (5 if args.quick else 30)
    args.max_seconds = args.max_seconds if args.max_seconds is not None else (90 if args.quick else 540)
    if not 0 < args.row_bytes <= PAGE or args.batches < 1 or args.max_seconds <= 0:
        ap.error("row-bytes must be in 1..4096, batches and max-seconds must be positive")
    if any(n < 1 for n in args.batch_sizes + args.threads + args.queue_depths):
        ap.error("batch sizes, thread counts and queue depths must be positive")
    args.table = str(Path(args.table).resolve())
    if (args.out.resolve() == Path(args.table) or
            args.out.exists() and os.path.samefile(args.out, args.table)):
        ap.error("--out must not overwrite the input table")
    return args


def main(argv=None):
    args = parse_args(argv)
    start = time.perf_counter()
    deadline = start + args.max_seconds
    size = os.stat(args.table).st_size
    rows, remainder = divmod(size, args.row_bytes)
    if not rows or remainder or max(args.batch_sizes) > rows:
        raise ValueError("table must contain whole rows and be at least as large as every batch")
    dev = device_info(args.table)
    res = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "command": shlex.join([sys.executable, *sys.argv]), "device": dev,
           "kernel": platform.release(), "platform": platform.platform(), "python": sys.version,
           "numpy": np.__version__, "liburing": command_output(["pkg-config", "--modversion", "liburing"]),
           "gcc": command_output(["gcc", "-dumpfullversion"]), "table_bytes": size, "table_rows": rows,
           "page_bytes": PAGE, "os_page_bytes": mmap.PAGESIZE,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "methodology": {"cold": "DONTNEED + mincore == 0 before every batch",
                           "rows": "uniform without replacement, sorted; identical batches for every method",
                           "timed": "page planning + reads + row copy; persistent readers, reserved buffers",
                           "scope": "raw table row I/O only; no scales, hot-row cache, FIFO or GPU transfers",
                           "excluded": "setup, reference, eviction, mincore, checksums",
                           "mb": "decimal MB of requested unique 4096-byte pages; not physical device bytes",
                           "device_counters": "whole device; other processes can contribute",
                           "p99": "NumPy linear percentile; smoke samples do not establish tail latency"},
           "results": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        res["wall_seconds"] = time.perf_counter() - start
        write_json(args.out, res, indent=2)

    fd = os.open(args.table, os.O_RDONLY)
    try:
        rng = np.random.default_rng(args.seed)
        batches = {n: [np.sort(rng.choice(rows, n, replace=False)) for _ in range(args.batches)]
                   for n in dict.fromkeys(args.batch_sizes)}
        t0 = time.perf_counter()
        references = {}
        for n, group in batches.items():
            references[n] = [plain_checksum(fd, batch, args.row_bytes) for batch in group]
        res["reference_seconds"] = time.perf_counter() - t0
        res["reference_sha256"] = references
        for config in configurations(args):
            if time.perf_counter() >= deadline:
                result = {**config, "status": "time_limit", "samples": []}
            else:
                try:
                    result = measure(config, args, batches[config["batch_rows"]],
                                     references[config["batch_rows"]], fd, size, dev["name"], deadline)
                except Exception as exc:
                    code = getattr(exc, "errno", None)
                    unavailable = code in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM, errno.EACCES)
                    result = {**config, "status": "unavailable" if unavailable else "error",
                              "errno": code, "error_type": type(exc).__name__, "error": str(exc)}
            res["results"].append(result)
            save()
            label = f"{config['method']:5s} n={config['batch_rows']:5d} "
            label += f"threads={config.get('threads', '-')} qd={config.get('queue_depth', '-')}"
            if "rows_per_s" in result:
                print(f"{label:42s} {result['rows_per_s']:10.0f} rows/s {result['pages_per_s']:10.0f} pages/s "
                      f"{result['mb_per_s']:8.1f} MB/s p50={result['batch_latency_p50_ms']:.3f} ms "
                      f"p99={result['batch_latency_p99_ms']:.3f} ms [{result['status']}]", flush=True)
            else:
                print(f"{label}: {result['status']} {result.get('error', '')}", flush=True)
        res["complete"] = all(r["status"] == "ok" for r in res["results"])
    finally:
        # Leave no benchmark-warmed clean pages behind, including reference reads.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        res["resident_pages_at_exit"] = resident_pages(fd, size)
        os.close(fd)
        save()
    print(f"JSON: {args.out}; {res['wall_seconds']:.2f} s; complete={res.get('complete', False)}")
    return 0 if res.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
