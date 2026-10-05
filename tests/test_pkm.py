import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.pkm import ProductKeyMemory

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def small_pkm(device, value_impl="embedding_bag", query_norm="batchnorm", n_keys=16, heads=2, knn=8, k_dim=16):
    torch.manual_seed(0)
    return ProductKeyMemory(d_in=24, d_out=24, n_keys=n_keys, heads=heads, knn=knn, k_dim=k_dim,
                            query_norm=query_norm, value_impl=value_impl).double().to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_keys,knn", [(16, 8), (16, 16), (32, 5), (8, 1)])
def test_product_key_topk_equals_bruteforce(device, n_keys, knn):
    """Product-key search must return exactly the top-k of a brute-force search over all n_keys^2 keys."""
    mem = small_pkm(device, n_keys=n_keys, knn=knn, heads=3)
    torch.manual_seed(1)
    q = torch.randn(200, mem.heads, mem.k_dim, dtype=torch.float64, device=device)
    scores, indices = mem.get_indices(q)

    half = mem.k_dim // 2
    for h in range(mem.heads):
        k1, k2 = mem.keys[h, 0].detach(), mem.keys[h, 1].detach()      # (n_keys, half)
        # explicit full key table: key (i, j) = concat(k1[i], k2[j]) at flat index i * n_keys + j
        full_keys = torch.cat([k1.repeat_interleave(n_keys, 0), k2.repeat(n_keys, 1)], dim=1)
        assert full_keys.shape == (n_keys ** 2, mem.k_dim)
        bf_scores, bf_idx = (q[:, h] @ full_keys.T).topk(knn, dim=-1)
        torch.testing.assert_close(scores[:, h], bf_scores, rtol=0, atol=1e-12)
        assert torch.equal(indices[:, h].sort(-1).values, bf_idx.sort(-1).values)
        # and each returned score belongs to its index
        recomputed = (q[:, h].unsqueeze(1) * full_keys[indices[:, h]]).sum(-1)
        torch.testing.assert_close(scores[:, h], recomputed, rtol=0, atol=1e-12)
    assert half * 2 == mem.k_dim


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value_impl", ["embedding_bag", "gather"])
def test_gradient_only_reaches_read_values(device, value_impl):
    """values.grad is non-zero exactly on the rows that were read, and zero everywhere else."""
    mem = small_pkm(device, value_impl=value_impl)
    torch.manual_seed(2)
    x = torch.randn(4, 5, mem.d_in, dtype=torch.float64, device=device)
    mem.record = True
    out = mem(x)
    (out * torch.randn_like(out)).sum().backward()
    read = torch.unique(mem.last_indices.reshape(-1))
    grad_rows = torch.nonzero(mem.values.weight.grad.abs().sum(-1) > 0).squeeze(-1)
    assert torch.equal(grad_rows, read)
    assert read.numel() < mem.size        # the test is meaningful: some rows were not read
    # keys and query network receive gradient through the softmax weights
    assert mem.keys.grad.abs().sum() > 0
    assert mem.query_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_value_read_implementations_agree(device):
    m1 = small_pkm(device, "embedding_bag")
    m2 = small_pkm(device, "gather")
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(3, 7, m1.d_in, dtype=torch.float64, device=device)
    o1, o2 = m1(x), m2(x)
    torch.testing.assert_close(o1, o2)
    o1.square().sum().backward()
    o2.square().sum().backward()
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        torch.testing.assert_close(p1.grad, p2.grad, msg=n)


def tiny_model(mem=True):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=3, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[1] if mem else [], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16)
    torch.manual_seed(0)
    return Transformer(cfg).double()


@pytest.mark.parametrize("mem", [False, True])
def test_causal_in_eval_mode(mem):
    """Changing future tokens must not change earlier logits (BN uses running stats in eval mode)."""
    model = tiny_model(mem).eval()
    x = torch.randint(0, 128, (2, 40))
    y = x.clone()
    y[:, 25:] = torch.randint(0, 128, (2, 15))
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:, :25], model(y)[:, :25])


@pytest.mark.parametrize("mem", [False, True])
def test_kv_cache_decode_matches_full_forward(mem):
    model = tiny_model(mem).eval()
    x = torch.randint(0, 128, (1, 30))
    with torch.no_grad():
        full = model(x)
        caches = [dict() for _ in range(model.cfg.n_layers)]
        step_logits = [model(x[:, :10], kv_caches=caches, pos0=0)]
        for t in range(10, 30):
            step_logits.append(model(x[:, t:t + 1], kv_caches=caches, pos0=t))
    torch.testing.assert_close(torch.cat(step_logits, 1), full)


@pytest.mark.parametrize("mem", [False, True])
def test_kv_cache_chunks_match_full_forward(mem):
    """Several new tokens at once after a filled cache (chunked prefill): each must only see the cache and the
    new tokens before it. Changing the last token of a chunk must not change the earlier ones."""
    model = tiny_model(mem).eval().double()
    x = torch.randint(0, 128, (2, 30))
    with torch.no_grad():
        full = model(x)
        caches = [dict() for _ in range(model.cfg.n_layers)]
        chunks = [model(x[:, a:a + n], kv_caches=caches, pos0=a) for a, n in ((0, 10), (10, 5), (15, 1), (16, 14))]
        torch.testing.assert_close(torch.cat(chunks, 1), full)
        y = x.clone()
        y[:, 14] = (y[:, 14] + 1) % 128
        caches = [dict() for _ in range(model.cfg.n_layers)]
        model(y[:, :10], kv_caches=caches, pos0=0)
        torch.testing.assert_close(model(y[:, 10:15], kv_caches=caches, pos0=10)[:, :4], chunks[1][:, :4])


def test_param_counts_and_macs():
    model = tiny_model(True)
    pc = model.param_counts()
    assert pc["memory_values"] == 16 ** 2 * 32
    assert pc["non_embedding"] == pc["dense_body"] + pc["memory_values"]
    assert pc["active_non_embedding_per_token"] == pc["dense_body"] + 2 * 4 * 32


def test_triton_rejects_unsupported_shapes():
    """The Triton selection kernel needs power-of-two n_keys (<= 65536) and knn: other sizes fail early with a
    clear message instead of deep inside the kernel; the PyTorch path takes any size."""
    for n_keys, knn in ((24, 4), (16, 5)):
        with pytest.raises(ValueError, match="powers of two"):
            ProductKeyMemory(32, 32, n_keys=n_keys, knn=knn, k_dim=16, impl="triton")
        ProductKeyMemory(32, 32, n_keys=n_keys, knn=knn, k_dim=16, impl="torch")
