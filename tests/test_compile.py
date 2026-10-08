"""Dense compilation parity, eager memory boundaries, accumulated gradients and cache fallback.

Run: OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 python -m pytest -q tests/test_compile.py
"""
import pytest
import torch

from smlm.compile import compile_dense
from smlm.model import ModelConfig, Transformer
from smlm.sparse_values import row_sparse_tables, row_store


@pytest.fixture(autouse=True)
def fresh_dynamo():
    """Dynamo keeps compiled variants per code object across tests; without a reset the shared dense-body function
    hits the recompile limit (8) after a few tests, which fullgraph=True turns into a hard error."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def tiny(kind="dense", impl="torch"):
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=2, ffn_hidden=24, max_seq_len=16)
    if kind == "memory":
        cfg.mem_layers = [1]
        cfg.mem_n_keys, cfg.mem_heads, cfg.mem_knn, cfg.mem_k_dim = 8, 2, 2, 8
        cfg.mem_impl, cfg.mem_value_grad = impl, "row_sparse"
        cfg.eng_layers = [0]
        cfg.eng_heads, cfg.eng_head_dim, cfg.eng_rows = 1, 8, 17
        cfg.eng_impl = impl
    torch.manual_seed(17)
    return Transformer(cfg)


def _assert_eager(module, args):
    assert not torch.compiler.is_compiling(), type(module).__name__


def _unexpected_compile(*args):
    raise AssertionError("KV-cache calls must not enter compiled regions")


def _parity(device, kind, bf16=False, fused_ce=False):
    torch.set_num_threads(4)
    impl = "triton" if device == "cuda" else "torch"
    ref = tiny(kind, impl).to(device)
    compiled = tiny(kind, impl).to(device)
    compiled.load_state_dict(ref.state_dict())
    names = list(compiled.state_dict())
    params = [id(p) for p in compiled.parameters()]
    for module in compiled.memory_layers() + compiled.engram_layers():
        module.register_forward_pre_hook(_assert_eager)
    compile_dense(compiled)
    assert list(compiled.state_dict()) == names
    assert [id(p) for p in compiled.parameters()] == params
    assert compiled.lm_head.weight is compiled.tok_emb.weight
    # FP32 fusion changes rounding; bf16 also rounds fused intermediates differently.
    tol = dict(rtol=3e-2, atol=3e-3) if bf16 else dict(rtol=3e-4, atol=3e-6)
    for _ in range(2):
        idx = torch.randint(0, ref.cfg.vocab_size, (2, 7), device=device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=bf16):
            logits0, loss0 = ref(idx[:, :-1], idx[:, 1:])
            logits1, loss1 = compiled(idx[:, :-1], idx[:, 1:], fused_ce=fused_ce, ce_chunk_size=5)
        if fused_ce:
            assert logits1 is None
        else:
            torch.testing.assert_close(logits1, logits0, **tol)
        torch.testing.assert_close(loss1, loss0, **tol)
        (loss0 / 2).backward()
        (loss1 / 2).backward()
    for (name, p0), (_, p1) in zip(ref.named_parameters(), compiled.named_parameters()):
        if p0.grad is None:
            assert p1.grad is None, name
        else:
            torch.testing.assert_close(p1.grad, p0.grad, msg=name, **tol)
    for p0, p1 in zip(row_sparse_tables(ref), row_sparse_tables(compiled)):
        torch.testing.assert_close(row_store(p1).acc, row_store(p0).acc, **tol)
        torch.testing.assert_close(row_store(p1).touched, row_store(p0).touched)
        assert p1.grad is None
    ref.eval()
    compiled.eval()
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=bf16):
        torch.testing.assert_close(compiled(idx), ref(idx), **tol)
        for block in compiled.layers:
            block._dense_compile["body"] = _unexpected_compile
        caches0 = [dict() for _ in ref.layers]
        caches1 = [dict() for _ in compiled.layers]
        for a, b in ((0, 3), (3, 4), (4, 7)):
            out0 = ref(idx[:, a:b], kv_caches=caches0, pos0=a)
            out1 = compiled(idx[:, a:b], kv_caches=caches1, pos0=a)
            torch.testing.assert_close(out1, out0, **tol)


@pytest.mark.parametrize("kind,bf16", [("dense", False), ("memory", False), ("dense", True)])
def test_inductor_cpu(kind, bf16):
    _parity("cpu", kind, bf16)


@pytest.mark.parametrize("kind", ["dense", "memory"])
def test_inductor_with_fused_ce_cpu(kind):
    _parity("cpu", kind, fused_ce=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
@pytest.mark.parametrize("kind", ["dense", "memory"])
@pytest.mark.parametrize("fused_ce", [False, True])
def test_inductor_gpu_bf16(kind, fused_ce):
    _parity("cuda", kind, bf16=True, fused_ce=fused_ce)


def test_compile_opt_in_and_restore():
    model = tiny()
    idx = torch.randint(0, model.cfg.vocab_size, (1, 5))
    with torch.no_grad():
        before = model(idx)
    assert all("forward" not in block.__dict__ for block in model.layers)
    compile_dense(model, enabled=False)
    assert all("forward" not in block.__dict__ for block in model.layers)
    compile_dense(model)
    callables = [block._dense_compile for block in model.layers]
    compile_dense(model)
    assert all(block._dense_compile is old for block, old in zip(model.layers, callables))
    compile_dense(model, enabled=False)
    assert all("forward" not in block.__dict__ and not hasattr(block, "_dense_compile") for block in model.layers)
    with torch.no_grad():
        torch.testing.assert_close(model(idx), before, rtol=0, atol=0)


def test_compile_graph_reused_across_layers():
    """A full-depth preset must not specialize its dense graph on each block's identity."""
    torch.set_num_threads(4)
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=12, n_heads=2, ffn_hidden=24, max_seq_len=16)
    model = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 6))
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    with torch.no_grad():
        expected = model(idx)
        compile_dense(model, backend=backend)
        actual = model(idx)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(graphs) == 1
