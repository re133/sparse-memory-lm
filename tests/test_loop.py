"""Step 12 (LOOP): a memory layer that reads the shared table a second time (ModelConfig.mem_round_layers).

CPU: structure, default unchanged, round 2 adds exactly its own read, row-sparse gradient with four readers against
the dense gradient, saving and loading. GPU: Triton against PyTorch path, decode graphs bit-identical.
"""
import json
import os

import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.train import MODELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU = torch.cuda.is_available()


def tiny(rounds=(), seed=0, value_grad="dense", impl="torch", device="cpu"):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[0, 2, 3], mem_round_layers=list(rounds), mem_n_keys=16, mem_heads=2, mem_knn=4,
                      mem_k_dim=16, mem_share_values=True, mem_value_grad=value_grad, mem_impl=impl)
    torch.manual_seed(seed)
    m = Transformer(cfg)
    return m.double() if device == "cpu" else m.to(device)


def test_default_has_no_second_read():
    m = tiny()
    assert m.memory_sites() == ["L0", "L2", "L3"] and len(m.memory_layers()) == 3
    assert not any("_r2" in k for k in m.state_dict())
    assert all(b.ffn_r2 is None for b in m.layers)


def test_old_config_loads_with_default():
    """Configs written before the field existed (every checkpoint so far) load as before."""
    with open(os.path.join(ROOT, "runs", "hop2", "B-1M-hop2-s0", "run-info.json")) as f:
        old = json.load(f)["model_config"]
    assert "mem_round_layers" not in old
    assert ModelConfig(**old).mem_round_layers == []


def test_preset():
    p = MODELS["B-1M-r2-sparse"]
    base = MODELS["B-1M-sparse"]
    assert p["mem_round_layers"] == [10] and {k: v for k, v in p.items() if k != "mem_round_layers"} == base


def test_round2_structure():
    m = tiny(rounds=[3])
    mems = m.memory_layers()
    assert m.memory_sites() == ["L0", "L2", "L3", "L3r2"] and len(mems) == 4
    assert mems[3] is m.layers[3].ffn_r2
    assert all(x.values is mems[0].values for x in mems)                 # one table
    assert len({id(x.keys) for x in mems}) == 4                           # own keys / query per read
    assert len({id(x.query_proj.weight) for x in mems}) == 4
    pc, pc0 = m.param_counts(), tiny().param_counts()
    assert pc["memory_values"] == pc0["memory_values"] == 16 ** 2 * 32   # counted once
    r2 = sum(p.numel() for p in mems[3].parameters() if p is not mems[0].values.weight)
    norm = sum(p.numel() for p in m.layers[3].ffn_norm_r2.parameters())
    assert pc["dense_body"] == pc0["dense_body"] + r2 + norm
    assert pc["active_non_embedding_per_token"] == pc["dense_body"] + 4 * 2 * 4 * 32
    assert m.macs_per_token()["memory"] == 4 * mems[0].macs_per_token()


def test_round2_adds_exactly_its_read():
    """With round 2's output projection zeroed, the model equals the same weights without round 2; otherwise the
    third layer's output is first read + second read on the updated stream."""
    m2 = tiny(rounds=[3])
    m1 = tiny()
    sd = {k: v for k, v in m2.state_dict().items() if "_r2" not in k}
    m1.load_state_dict(sd)
    x = torch.randint(0, 128, (2, 9), generator=torch.Generator().manual_seed(3))
    m1.eval(), m2.eval()
    with torch.no_grad():
        assert not torch.equal(m1(x), m2(x))
        r2 = m2.layers[3].ffn_r2
        r2.value_proj.weight.zero_()
        r2.value_proj.bias.zero_()
        assert torch.equal(m1(x), m2(x))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if GPU else []))
def test_row_sparse_gradient_four_readers(device):
    """Row-sparse accumulation of the shared table (four readers, two micro-batches) equals the dense gradient;
    every other gradient is the same."""
    dense = tiny(rounds=[3], value_grad="dense", device=device)
    sparse = tiny(rounds=[3], value_grad="row_sparse", device=device)
    sparse.load_state_dict(dense.state_dict())
    gen = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 128, (2, 17), generator=gen).to(device) for _ in range(2)]
    for m in (dense, sparse):
        for b in batches:
            _, loss = m(b[:, :-1], b[:, 1:])
            (loss / 2).backward()
    table_d = dense.memory_layers()[0].values.weight
    table_s = sparse.memory_layers()[0].values.weight
    tol = dict(rtol=1e-10, atol=1e-12) if device == "cpu" else dict(rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(table_s.row_store.acc.to(table_d.grad.dtype), table_d.grad, **tol)
    read = table_d.grad.abs().sum(-1) > 0
    assert torch.equal(table_s.row_store.touched, read)
    for (n, p0), p1 in zip(dense.named_parameters(), sparse.parameters()):
        if p0 is table_d:
            continue
        torch.testing.assert_close(p1.grad, p0.grad, **tol, msg=n)


def test_save_and_load(tmp_path):
    m = tiny(rounds=[3]).eval()
    path = tmp_path / "model.pt"
    torch.save({"model_config": m.cfg.to_dict(), "state_dict": m.state_dict()}, path)
    ck = torch.load(path, weights_only=True)
    m2 = Transformer(ModelConfig(**ck["model_config"])).double().eval()
    m2.load_state_dict(ck["state_dict"])
    assert m2.memory_sites() == m.memory_sites()
    assert all(x.values is m2.memory_layers()[0].values for x in m2.memory_layers())
    x = torch.randint(0, 128, (2, 9), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))


@pytest.mark.skipif(not GPU, reason="Triton kernels need a GPU")
def test_triton_equals_torch_round2():
    ref = tiny(rounds=[3], value_grad="row_sparse", impl="torch", device="cuda")
    ker = tiny(rounds=[3], value_grad="row_sparse", impl="triton", device="cuda")
    ker.load_state_dict(ref.state_dict())
    gen = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 128, (2, 17), generator=gen).cuda() for _ in range(2)]
    for m in (ref, ker):
        for b in batches:
            _, loss = m(b[:, :-1], b[:, 1:])
            (loss / 2).backward()
    for (n, p0), p1 in zip(ref.named_parameters(), ker.parameters()):
        if p0.grad is not None:
            torch.testing.assert_close(p1.grad, p0.grad, rtol=1e-5, atol=1e-5, msg=n)
    t0, t1 = ref.memory_layers()[0].values.weight, ker.memory_layers()[0].values.weight
    torch.testing.assert_close(t1.row_store.acc, t0.row_store.acc, rtol=1e-5, atol=1e-5)
    assert torch.equal(t1.row_store.touched, t0.row_store.touched)


@pytest.mark.skipif(not GPU, reason="graphs need a GPU")
@pytest.mark.parametrize("kind", ["fp32", "q4"])
def test_decode_graph_round2(kind):
    ker = tiny(rounds=[3], value_grad="row_sparse", impl="triton", device="cuda").eval()
    ker.set_memory_inference_table(kind)
    x = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(2)).cuda()

    def decode():
        caches = [dict() for _ in range(ker.cfg.n_layers)]
        outs = [ker(x[:, :4], kv_caches=caches, pos0=0).float()]
        for i in range(4, 8):
            outs.append(ker(x[:, i:i + 1], kv_caches=caches, pos0=i).float())
        return torch.cat(outs, 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = decode()
        ker.set_memory_decode_graphs(True)
        graph = decode()
        graph2 = decode()
        ker.set_memory_decode_graphs(False)
    assert torch.equal(graph, eager) and torch.equal(graph2, eager)
