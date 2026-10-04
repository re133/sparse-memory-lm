"""Step 2: value table outside the GPU (smlm/offload.py, scripts/convert_table.py)."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from smlm.kernels import quantize_q4
from smlm.model import ModelConfig, Transformer
from smlm.offload import HostTable, MmapQ4Table, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA = torch.cuda.is_available()


def _q4_file(tmp_path, rows=5000, dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, dim, generator=g)
    packed, scales = quantize_q4(w)
    q, s = tmp_path / "q.bin", tmp_path / "s.bin"
    packed.numpy().tofile(q)
    scales.numpy().tofile(s)
    return str(q), str(s), packed, scales


@pytest.mark.parametrize("cache,fifo", [(0, 0), (700, 0), (0, 300), (500, 200)])
def test_mmap_fetch_matches_table(tmp_path, cache, fifo):
    q, s, packed, scales = _q4_file(tmp_path)
    rows, width = packed.shape
    hot = np.random.default_rng(1).permutation(rows)
    t = MmapQ4Table(q, s, rows, width * 2, hot_rows=hot, cache_rows=cache, fifo_rows=fifo, device="cpu")
    rng = np.random.default_rng(2)
    for it in range(30):
        u = np.unique(rng.integers(0, rows, size=rng.integers(1, 900)))
        out = torch.empty(len(u), width, dtype=torch.uint8)
        out_s = torch.empty(len(u), dtype=torch.float16)
        t.fetch(u, out, out_s)
        assert torch.equal(out, packed[u]) and torch.equal(out_s, scales[u])
        if it == 10:
            t.drop_os_cache()
        # cache bookkeeping stays consistent
        filled = t.slot2row >= 0
        assert (t.row2slot[t.slot2row[filled]] == np.flatnonzero(filled)).all()
        assert (t.row2slot >= 0).sum() == filled.sum()
    assert t.stats["hits"] + t.stats["misses"] == t.stats["hits"] + t.stats["misses"] > 0
    if cache == 0 and fifo == 0:
        assert t.stats["hits"] == 0
    t.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, "q4"])
def test_host_fetch_matches_table(dtype):
    w = torch.randn(3000, 64)
    if dtype == "q4":
        p, sc = quantize_q4(w)
        t = HostTable(p, sc, device="cpu")
    else:
        p, sc = w.to(dtype), None
        t = HostTable(p, device="cpu")
    u = np.unique(np.random.default_rng(0).integers(0, 3000, 500))
    out = torch.empty(len(u), p.shape[1], dtype=p.dtype)
    out_s = torch.empty(len(u), dtype=torch.float16) if sc is not None else None
    t.fetch(u, out, out_s)
    assert torch.equal(out, p[u])
    if sc is not None:
        assert torch.equal(out_s, sc[u])


@pytest.mark.skipif(not CUDA, reason="kernel 4 / Triton on the GPU")
def test_offload_model_bit_identical(tmp_path):
    """b) and c) give exactly the logits of the same table in GPU memory (a), for bf16 and 4 bit."""
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=3, n_heads=2, ffn_hidden=128, max_seq_len=64,
                      mem_layers=[0, 2], mem_n_keys=64, mem_heads=2, mem_knn=8, mem_k_dim=32,
                      mem_share_values=True, mem_impl="triton")
    torch.manual_seed(0)
    ref = Transformer(cfg).cuda()
    with torch.no_grad():
        ref.memory_layers()[0].values.weight.normal_(0, 0.5)          # a table that matters
    for m in ref.memory_layers():
        m.query_norm.eval()
    ck = tmp_path / "model.pt"
    torch.save({"model_config": cfg.to_dict(), "state_dict": ref.state_dict()}, ck)
    out = tmp_path / "tables"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "convert_table.py"), "--ckpt", str(ck), "--out",
                    str(out), "--chunk", "1000"], check=True, capture_output=True)
    rows, dim = 64 * 64, 64
    x = torch.randint(0, 256, (2, 48), device="cuda")
    ref.eval()

    def logits(model):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x).float()

    ref.set_memory_inference_table("bf16")
    want_bf16 = logits(ref)
    ref.set_memory_inference_table("q4")
    want_q4 = logits(ref)
    bf = torch.from_numpy(np.fromfile(out / "values_bf16.bin", dtype=np.int16)).view(torch.bfloat16).view(rows, dim)
    p = torch.from_numpy(np.fromfile(out / "values_q4.bin", dtype=np.uint8)).view(rows, dim // 2)
    s = torch.from_numpy(np.fromfile(out / "scales_q4.bin", dtype=np.float16))
    rest = str(out / "rest.pt")
    assert torch.equal(logits(load_model(rest, (bf.cuda(), None))), want_bf16)
    assert torch.equal(logits(load_model(rest, HostTable(bf))), want_bf16)
    assert torch.equal(logits(load_model(rest, (p.cuda(), s.cuda()))), want_q4)
    assert torch.equal(logits(load_model(rest, HostTable(p, s))), want_q4)
    c = MmapQ4Table(str(out / "values_q4.bin"), str(out / "scales_q4.bin"), rows, dim,
                    hot_rows=np.arange(rows), cache_rows=500, fifo_rows=300)
    assert torch.equal(logits(load_model(rest, c)), want_q4)
    assert c.stats["misses"] > 0 and c.stats["hits"] > 0
