"""Step 2: split a trained memory checkpoint into the small rest and the value table as flat files (CPU only).

  python scripts/convert_table.py --ckpt runs/cloud/B-16M-s0/model.pt --out data/tables/B-16M   (ideally on an NVMe drive)

Writes into --out (the NVMe):
  rest.pt           model_config + every weight except the value table (a few hundred MB at most)
  values_bf16.bin   table in bf16, row-major (rows x dim x 2 bytes), round-to-nearest-even as torch .to(bfloat16)
  values_q4.bin     table in 4 bit, row-major (rows x dim/2 bytes), kernels.quantize_q4 (same format as kernel 4)
  scales_q4.bin     fp16 scale per row (rows x 2 bytes)
  hot_rows.npy      row ids sorted by reads during training (most read first; from mem_access_train.npy)
  meta.json         shapes, sha256 of every file and of the source checkpoint
The fp32 table stays in the checkpoint (variant b-fp32 reads it from there). The checkpoint is opened with mmap;
the table is processed in chunks, so the RAM peak is about one chunk.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.kernels import quantize_q4  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=1 << 20)
    ap.add_argument("--check_gpu_rows", type=int, default=1 << 20,
                    help="compare the CPU quantisation with quantize_q4 on the GPU for this many rows (0: skip)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    torch.set_num_threads(os.cpu_count())
    ck = torch.load(args.ckpt, mmap=True, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    table_keys = [k for k in sd if k.endswith("values.weight")]
    table = sd[table_keys[0]]
    assert all(sd[k].untyped_storage().data_ptr() == table.untyped_storage().data_ptr() for k in table_keys), \
        "expected one shared table"
    rows, dim = table.shape
    rest = {k: v.clone() for k, v in sd.items() if k not in table_keys}
    torch.save({"model_config": ck["model_config"], "state_dict": rest, "table_keys": table_keys,
                "table_shape": [rows, dim]}, os.path.join(args.out, "rest.pt"))
    print(f"rest.pt: {sum(v.numel() for v in rest.values()) / 1e6:.1f} M parameters; table {rows} x {dim}", flush=True)

    paths = {n: os.path.join(args.out, n) for n in ("values_bf16.bin", "values_q4.bin", "scales_q4.bin")}
    with open(paths["values_bf16.bin"], "wb") as fb, open(paths["values_q4.bin"], "wb") as fq, \
            open(paths["scales_q4.bin"], "wb") as fs:
        for a in range(0, rows, args.chunk):
            w = table[a:a + args.chunk]
            fb.write(w.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
            packed, scales = quantize_q4(w)
            fq.write(packed.numpy().tobytes())
            fs.write(scales.view(torch.int16).numpy().tobytes())
            if (a // args.chunk) % 4 == 0:
                print(f"  {a + w.shape[0]:,} / {rows:,} rows, {time.time() - t0:.0f} s", flush=True)

    check = None
    if args.check_gpu_rows and torch.cuda.is_available():
        n = min(args.check_gpu_rows, rows)
        pg, sg = quantize_q4(table[:n].cuda())
        q = np.fromfile(paths["values_q4.bin"], dtype=np.uint8, count=n * dim // 2).reshape(n, dim // 2)
        s = np.fromfile(paths["scales_q4.bin"], dtype=np.int16, count=n)
        check = {"rows": n, "packed_equal": bool((pg.cpu().numpy() == q).all()),
                 "scales_equal": bool((sg.view(torch.int16).cpu().numpy() == s).all())}
        print("CPU vs GPU quantisation:", check, flush=True)

    acc = os.path.join(os.path.dirname(args.ckpt), "mem_access_train.npy")
    if os.path.exists(acc):
        counts = np.load(acc)
        np.save(os.path.join(args.out, "hot_rows.npy"), np.argsort(-counts, kind="stable").astype(np.int64))
    meta = {"source": os.path.abspath(args.ckpt), "rows": rows, "dim": dim, "table_keys": table_keys,
            "q4": "kernels.quantize_q4: per row d = fp16(max-abs value with sign / -8), code = clamp(round(w/d),-8,7)+8,"
                  " low nibble = dims 0..dim/2-1, high nibble = dims dim/2..dim-1",
            "cpu_vs_gpu_q4_check": check, "seconds": round(time.time() - t0, 1),
            "sha256": {n: sha256(os.path.join(args.out, n)) for n in sorted(os.listdir(args.out))
                       if n.endswith((".bin", ".pt", ".npy"))}}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
