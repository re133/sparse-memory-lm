"""Engram-style n-gram memory (smlm/engram.py): token compression, hashing, causality, KV cache, row-sparse training."""
import pytest
import torch

from smlm.engram import NgramHash, canonical_ids
from smlm.model import ModelConfig, Transformer
from smlm.optim import build_optimizer, clip_grads

CUDA = torch.cuda.is_available()


def tiny(impl="torch", seed=0, head_dim=8, **kw):
    cfg = ModelConfig(vocab_size=50304, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      eng_layers=[0, 2], eng_heads=2, eng_head_dim=head_dim, eng_rows=101, eng_impl=impl, **kw)
    torch.manual_seed(seed)
    return Transformer(cfg)


def test_canonical_ids():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    canon, n = canonical_ids(50304)
    same = [(" The", " the"), ("THE", "the"), ("ﬁ", "fi")]            # case and NFKC (ligature)
    for a, b in same:
        ia, ib = enc.encode_ordinary(a), enc.encode_ordinary(b)
        if len(ia) == len(ib) == 1:
            assert canon[ia[0]] == canon[ib[0]], (a, b)
    assert canon[enc.encode_ordinary(" cat")[0]] != canon[enc.encode_ordinary(" dog")[0]]
    assert 30000 < n < 50304 and int(canon.max()) == n - 1
    # byte pieces that aren't valid UTF-8 alone keep their own ids
    pieces = [i for i in range(256) if len(enc.decode_single_token_bytes(i)) == 1 and enc.decode_single_token_bytes(i)[0] >= 0x80]
    assert len({int(canon[i]) for i in pieces}) == len(pieces)


def test_hash_range_heads_and_determinism():
    h = NgramHash(50304, orders=(2, 3), heads=4, rows=101, seed=0)
    c = torch.randint(0, h.pad, (3, 2 + 50))
    r = h(c)
    assert r.shape == (3, 50, 8)
    for j in range(8):                                     # head j uses its own block of rows
        assert bool(((r[..., j] >= j * 101) & (r[..., j] < (j + 1) * 101)).all())
    assert torch.equal(r, NgramHash(50304, orders=(2, 3), heads=4, rows=101, seed=0)(c))
    # a bigram head only depends on tokens t-1, t
    c2 = c.clone()
    c2[:, 0] = (c2[:, 0] + 1) % h.pad                      # token t-2 of the first position
    r2 = h(c2)
    assert torch.equal(r2[:, 0, :4], r[:, 0, :4]) and not torch.equal(r2[:, 0, 4:], r[:, 0, 4:])


@pytest.mark.skipif(not CUDA, reason="GPU")
def test_hash_cpu_equals_gpu():
    h = NgramHash(50304, seed=0)
    c = torch.randint(0, h.pad, (4, 2 + 1024))
    assert torch.equal(h(c), h.cuda()(c.cuda()).cpu())


@pytest.mark.parametrize("train", [False, True])
def test_causal(train):
    m = tiny().double()
    m.train(train)
    x = torch.randint(0, 50257, (2, 40))
    y = x.clone()
    y[:, 25:] = torch.randint(0, 50257, (2, 15))
    with torch.no_grad():
        torch.testing.assert_close(m(x)[:, :25], m(y)[:, :25])


def test_kv_cache_and_chunks_match_full_forward():
    m = tiny().double().eval()
    with torch.no_grad():                                  # a non-zero conv, so its cache is tested too
        for e in m.engram_layers():
            e.conv.weight.normal_(std=0.3)
    x = torch.randint(0, 50257, (2, 30))
    with torch.no_grad():
        full = m(x)
        caches = [dict() for _ in range(m.cfg.n_layers)]
        steps = [m(x[:, a:a + n], kv_caches=caches, pos0=a) for a, n in ((0, 7), (7, 1), (8, 1), (9, 12), (21, 9))]
    torch.testing.assert_close(torch.cat(steps, 1), full)


def test_conv_starts_as_identity():
    """Zero-initialised conv: the module output is exactly the gated value v~ at the start."""
    m = tiny().double()
    e = m.engram_layers()[0]
    h = torch.randn(2, 10, 32, dtype=torch.float64)
    rows = m.ngram(torch.cat([torch.full((2, 2), m.ngram.pad), m.ngram.canon[torch.randint(0, 50257, (2, 10))]], 1))
    y = e(h, rows)
    flat = rows.reshape(-1)
    ev = e.values.weight[flat].view(2, 10, -1)
    k, v = e.w_k(ev), e.w_v(ev)
    alpha = torch.sigmoid((e.h_norm(h) * e.k_norm(k)).sum(-1, keepdim=True) / 32 ** 0.5)
    torch.testing.assert_close(y, alpha * v)


def test_param_counts():
    m = tiny()
    pc = m.param_counts()
    assert pc["memory_values"] == 2 * 4 * 101 * 8           # 2 modules x (2 orders x 2 heads) x 101 rows x 8
    assert pc["active_non_embedding_per_token"] == pc["dense_body"] + 2 * 4 * 8
    assert m.macs_per_token()["engram"] > 0


def test_only_read_rows_change_and_frozen_rest():
    """One optimiser step: rows that weren't read keep their values (lazy Adam), the others move."""
    m = tiny()
    e = m.engram_layers()[0]
    before = e.values.weight.detach().clone()
    opt = build_optimizer(m, 1e-2, 3e-3, 0.1)
    x = torch.randint(0, 50257, (2, 17))
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    clip_grads(m, 1.0)
    opt.step()
    c = torch.cat([torch.full((2, 2), m.ngram.pad), m.ngram.canon[x[:, :-1]]], 1)
    read = torch.zeros(e.values.weight.shape[0], dtype=torch.bool)
    read[m.ngram(c).reshape(-1)] = True
    after = e.values.weight.detach()
    assert torch.equal(after[~read], before[~read])
    assert not torch.equal(after[read], before[read])


@pytest.mark.skipif(not CUDA, reason="Triton kernels on the GPU")
@pytest.mark.parametrize("head_dim", [8, 24])
def test_triton_equals_torch(head_dim):
    """Lookups of 1 row x head_dim (24 in E-1M) through the row-sparse Triton kernels: same gradients and table
    accumulator as the PyTorch reference."""
    ref, ker = tiny("torch", head_dim=head_dim).cuda(), tiny("triton", head_dim=head_dim).cuda()
    ker.load_state_dict(ref.state_dict())
    x = torch.randint(0, 50257, (2, 33), device="cuda")
    for m in (ref, ker):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
    for (n, p0), p1 in zip(ref.named_parameters(), ker.parameters()):
        if p0.grad is not None:
            torch.testing.assert_close(p1.grad, p0.grad, rtol=1e-5, atol=1e-6, msg=n)
    for e0, e1 in zip(ref.engram_layers(), ker.engram_layers()):
        torch.testing.assert_close(e1.values.weight.row_store.acc, e0.values.weight.row_store.acc, rtol=1e-5, atol=1e-6)
        assert torch.equal(e1.values.weight.row_store.touched, e0.values.weight.row_store.touched)


def test_both_memories_get_their_own_learning_rate():
    """BE: product-key table and Engram tables in separate lazy Adams with their own learning rates; one step runs."""
    cfg = ModelConfig(vocab_size=50304, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[1, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16, mem_share_values=True,
                      mem_value_grad="row_sparse", eng_layers=[0, 2], eng_heads=2, eng_head_dim=8, eng_rows=101,
                      eng_impl="torch")
    torch.manual_seed(0)
    m = Transformer(cfg)
    opt = build_optimizer(m, 1e-3, 2.4e-3, 0.1, eng_value_lr=3e-3)
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["memory_values"] == 2.4e-3 and lrs["engram_values"] == 3e-3
    assert m.param_counts()["memory_values"] == 16 ** 2 * 32 + 2 * 4 * 101 * 8
    x = torch.randint(0, 50257, (2, 17))
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    clip_grads(m, 1.0)
    opt.step()
    assert torch.isfinite(loss)


def test_dense_variant_same_forward_and_plain_adam():
    """eng_value_grad="dense": same outputs as the row-sparse version; the table gets an ordinary gradient and sits in
    AdamW's Engram value group (eng_value_lr, default value_lr, no weight decay), i.e. plain Adam on all rows as in
    the paper."""
    sp, de = tiny(), tiny(eng_value_grad="dense")
    de.load_state_dict(sp.state_dict())
    x = torch.randint(0, 50257, (2, 17))
    torch.testing.assert_close(de(x)[:, :5], sp(x)[:, :5])
    opt = build_optimizer(de, 1e-2, 3e-3, 0.1)
    groups = {g["name"]: g for g in opt.param_groups}
    tab = de.engram_layers()[0].values.weight
    assert any(p is tab for p in groups["engram_values"]["params"]) and groups["engram_values"]["weight_decay"] == 0
    assert groups["engram_values"]["lr"] == 3e-3
    # --eng_value_lr reaches the dense Engram table too (it used to be ignored there)
    groups = {g["name"]: g for g in build_optimizer(de, 1e-2, 3e-3, 0.1, eng_value_lr=5e-3).param_groups}
    assert groups["engram_values"]["lr"] == 5e-3 and any(p is tab for p in groups["engram_values"]["params"])
    _, loss = de(x[:, :-1], x[:, 1:])
    loss.backward()
    assert tab.grad is not None and tab.grad.shape == tab.shape
