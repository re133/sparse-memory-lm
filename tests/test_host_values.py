"""Host-resident training against the existing row-sparse value path.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m pytest -q tests/test_host_values.py
  TRITON_CACHE_DIR=$PWD/.cache/triton python -m pytest -q tests/test_host_values.py -k cuda

CPU fp32 trajectories use rtol=2e-5, atol=2e-7 for parameters and accumulated gradients;
losses use rtol=2e-6, atol=2e-7. GPU fp32 permits rtol=1e-4, atol=2e-6 for changed reduction order.
Decoded compressed moments allow one storage rounding interval relative to each row's maximum.
Fp32 moments use atol=1e-5 relative to the tensor maximum, except the analytically zero BN bias.
GPU bf16 trajectories allow absolute loss error 0.002 and parameter error below 3% of their movement.
"""
from contextlib import nullcontext
import os

import pytest
import torch

from smlm.host_values import enable_host_values
from smlm.model import ModelConfig, Transformer
from smlm.optim import build_optimizer, clip_grads
from smlm.sparse_values import LazyRowAdam, row_sparse_tables, row_store

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="host transfer tests need a GPU")
BACKENDS = [pytest.param("cpu", "torch", id="cpu-torch"),
            pytest.param("cuda", "torch", marks=CUDA, id="cuda-torch"),
            pytest.param("cuda", "triton", marks=CUDA, id="cuda-triton")]
MODES = ["fp32", "bf16", "int8"]


@pytest.fixture(autouse=True)
def four_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def tiny(impl="torch", share=True, seed=39):
    if impl == "triton":
        pytest.importorskip("triton")
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=4, n_heads=4, ffn_hidden=48, max_seq_len=24,
                      mem_layers=[0, 2, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16,
                      mem_share_values=share, mem_value_grad="row_sparse", mem_impl=impl)
    torch.manual_seed(seed)
    return Transformer(cfg)


def paired(device="cpu", impl="torch", share=True):
    normal, host = tiny(impl, share), tiny(impl, share)
    host.load_state_dict(normal.state_dict())
    enable_host_values(host)
    return normal.to(device), host.to(device)


def table_of(model):
    return model.memory_layers()[0].values.weight


def optimizers(model, mode):
    return build_optimizer(model, lr=3e-4, value_lr=7e-4, weight_decay=0.03, value_state=mode)


def optimizer_for(opt, param):
    return next(o for o in opt.opts if any(p is param for g in o.param_groups for p in g["params"]))


def bits_equal(actual, expected):
    return torch.equal(actual.detach().cpu().contiguous().view(torch.uint8),
                       expected.detach().cpu().contiguous().view(torch.uint8))


def close(actual, expected, device="cpu", **kwargs):
    tol = dict(rtol=2e-5, atol=2e-7) if device == "cpu" else dict(rtol=1e-4, atol=2e-6)
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), **dict(tol, **kwargs))


def decoded(state, mode):
    m, v = state["exp_avg"].float().cpu(), state["exp_avg_sq"].float().cpu()
    if mode == "int8":
        m = m.sign() * (m / 127).square() * state["exp_avg_scale"].cpu()[:, None]
        v = (v / 255 * state["exp_avg_sq_scale"].cpu()[:, None]).square().square()
    return m, v


def compare_states(first, second, normal, host, mode, device):
    for (name, p), (host_name, h) in zip(normal.named_parameters(), host.named_parameters()):
        assert name == host_name
        a, b = optimizer_for(first, p).state[p], optimizer_for(second, h).state[h]
        assert a.keys() == b.keys(), name
        assert int(a["step"]) == int(b["step"]), name
        if getattr(p, "pk_value_param", False) and mode != "fp32":
            # Decode before comparison: equal fp32 results can straddle a quantizer rounding boundary.
            tolerance = 0.008 if mode == "bf16" else 0.018
            for actual, expected in zip(decoded(b, mode), decoded(a, mode)):
                scale = expected.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
                close(actual / scale, expected / scale, device, rtol=1e-4, atol=tolerance, msg=name)
        else:
            for key in ("exp_avg", "exp_avg_sq"):
                if name.endswith("query_proj.bias"):
                    # BatchNorm cancels this bias analytically; relative error magnifies fp32 noise.
                    close(b[key], a[key], device, msg=f"{name}/{key}")
                else:
                    scale = a[key].abs().max().clamp_min(1e-30).cpu()
                    close(b[key].cpu() / scale, a[key].cpu() / scale, device, atol=1e-5,
                          msg=f"{name}/{key}")
        if getattr(h, "host_values", False):
            assert all(v.device.type == "cpu" for v in b.values() if torch.is_tensor(v))


@pytest.mark.parametrize("device,impl", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
def test_shared_training_matches_existing_path(tmp_path, device, impl, mode):
    normal, host = paired(device, impl)
    opts = [optimizers(m, mode) for m in (normal, host)]
    generator = torch.Generator().manual_seed(103)
    for m in (normal, host):
        for memory in m.memory_layers():
            memory.record = True
    for step in range(3):
        for _ in range(3):
            tokens = torch.randint(0, 64, (2, 8), generator=generator).to(device)
            losses, reads = [], []
            for m in (normal, host):
                _, loss = m(tokens[:, :-1], tokens[:, 1:])
                (loss / 3).backward()
                losses.append(loss)
                reads.append([mem.last_indices.cpu() for mem in m.memory_layers()])
            close(losses[1], losses[0], device, rtol=2e-6, atol=2e-7)
            assert all(torch.equal(a, b) for a, b in zip(*reads))
            # Check each partial accumulation, so dropping an earlier layer or micro-batch cannot hide.
            close(row_store(table_of(host)).acc, row_store(table_of(normal)).acc, device)
            assert torch.equal(row_store(table_of(host)).touched, row_store(table_of(normal)).touched.cpu())
        for (name, p), (_, h) in zip(normal.named_parameters(), host.named_parameters()):
            if p is table_of(normal):
                assert p.grad is None and h.grad is None
            else:
                close(h.grad, p.grad, device, msg=name)
        norms = [clip_grads(m, 1e-3) for m in (normal, host)]
        for actual, expected in zip(norms[1], norms[0]):
            close(actual, expected, device)
        assert float(row_store(table_of(host)).grad_scale) < 1.0
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        for (name, p), (_, h) in zip(normal.named_parameters(), host.named_parameters()):
            close(h, p, device, msg=name)
        compare_states(*opts, normal, host, mode, device)
        for m in (normal, host):
            st = row_store(table_of(m))
            assert not st.touched.any() and torch.count_nonzero(st.acc) == 0
            assert st.grad_scale == 1.0
        assert int(optimizer_for(opts[1], table_of(host)).state[table_of(host)]["step"]) == step + 1
    path = tmp_path / "model.pt"
    torch.save(host.state_dict(), path)
    restored = Transformer(host.cfg).to(device).eval()
    restored.load_state_dict(torch.load(path, weights_only=True, map_location=device), strict=True)
    host.eval()
    with torch.no_grad():
        close(restored(tokens[:, :-1]), host(tokens[:, :-1]), device)
    assert not getattr(table_of(restored), "host_values", False)


@pytest.mark.parametrize("device,impl", BACKENDS)
@pytest.mark.parametrize("mode", MODES)
def test_unread_rows_keep_historical_values_and_state(device, impl, mode):
    model = enable_host_values(tiny(impl)).to(device)
    memory, table = model.memory_layers()[0], table_of(model)
    opt = optimizers(model, mode)
    lazy = optimizer_for(opt, table)
    gen = torch.Generator().manual_seed(122)
    visits = [torch.arange(table.shape[0]).view(-1, 1), torch.tensor([[1, 1, 5], [5, 9, 1]]),
              torch.tensor([[3, 3, 9], [11, 3, 11]])]
    for step, indices in enumerate(visits):
        previous = table.detach().clone()
        before_state = {k: v.clone() for k, v in lazy.state[table].items() if k != "step"}
        w = torch.randn(indices.shape, generator=gen, device="cpu").to(device).requires_grad_()
        g = torch.randn(indices.shape[0], table.shape[1], generator=gen).to(device)
        memory.read_values(indices.to(device), w).backward(g)
        unread = torch.ones(table.shape[0], dtype=torch.bool)
        unread[indices.unique()] = False
        assert torch.equal(row_store(table).touched, ~unread)
        clip_grads(model, 0.25)
        opt.step()
        opt.zero_grad(set_to_none=True)
        assert bits_equal(table[unread], previous[unread])
        for key, value in before_state.items():
            assert bits_equal(lazy.state[table][key][unread], value[unread]), key
        assert int(lazy.state[table]["step"]) == step + 1
        assert table.grad is None
        assert not row_store(table).touched.any()
        assert torch.count_nonzero(row_store(table).acc) == 0


@pytest.mark.parametrize("impl", ["torch", "triton"])
@pytest.mark.parametrize("mode", MODES)
@CUDA
def test_cuda_bf16_autocast_matches_bags_and_optimizer(impl, mode):
    normal, host = paired("cuda", impl)
    opts = [optimizers(m, mode) for m in (normal, host)]
    gen = torch.Generator().manual_seed(89)
    for _ in range(3):
        indices = torch.randint(0, table_of(normal).shape[0], (13, 8), generator=gen).cuda()
        weights = torch.randn(13, 8, generator=gen).cuda().to(torch.bfloat16)
        gradient = torch.randn(13, 32, generator=gen).cuda()
        outputs, weight_grads = [], []
        for m in (normal, host):
            w = weights.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = m.memory_layers()[0].read_values(indices, w)
            output.backward(gradient)
            outputs.append(output)
            weight_grads.append(w.grad)
        close(outputs[1], outputs[0], "cuda")
        close(weight_grads[1], weight_grads[0], "cuda", rtol=0, atol=0)
        close(row_store(table_of(host)).acc, row_store(table_of(normal)).acc, "cuda")
        for m, opt in zip((normal, host), opts):
            clip_grads(m, 0.1)
            opt.step()
            opt.zero_grad(set_to_none=True)
        close(table_of(host), table_of(normal), "cuda")
        first = optimizer_for(opts[0], table_of(normal)).state[table_of(normal)]
        second = optimizer_for(opts[1], table_of(host)).state[table_of(host)]
        for a, b in zip(decoded(second, mode), decoded(first, mode)):
            scale = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
            close(a / scale, b / scale, "cuda", atol={"fp32": 1e-5, "bf16": 0.008, "int8": 0.018}[mode])


@pytest.mark.parametrize("impl", ["torch", "triton"])
@pytest.mark.parametrize("mode", ["fp32", "int8"])
@CUDA
def test_cuda_bf16_full_training_trajectory(impl, mode):
    normal, host = paired("cuda", impl)
    opts = [optimizers(m, mode) for m in (normal, host)]
    gen = torch.Generator().manual_seed(31)
    before = {name: p.detach().cpu().clone() for name, p in normal.named_parameters()}
    for _ in range(3):
        for _ in range(2):
            tokens = torch.randint(0, 64, (2, 8), generator=gen).cuda()
            losses = []
            for m in (normal, host):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = m(tokens[:, :-1], tokens[:, 1:])
                (loss / 2).backward()
                losses.append(loss)
            # A bf16 rounding boundary can change a later lookup; this is a short trajectory check.
            close(losses[1], losses[0], "cuda", rtol=0, atol=0.002)
        for m, opt in zip((normal, host), opts):
            clip_grads(m, 0.1)
            opt.step()
            opt.zero_grad(set_to_none=True)
        for (name, p), (_, h) in zip(normal.named_parameters(), host.named_parameters()):
            delta = (h.detach().cpu() - p.detach().cpu()).norm()
            movement = (p.detach().cpu() - before[name]).norm().clamp_min(1e-5)
            assert delta / movement < 0.03, (name, float(delta), float(movement))
        table = table_of(host)
        assert table.device.type == "cpu"
        assert all(v.device.type == "cpu" for v in optimizer_for(opts[1], table).state[table].values())


@pytest.mark.parametrize("device,impl", BACKENDS)
@pytest.mark.parametrize("share", [True, False])
def test_eval_has_no_row_store_and_checkpoint_loads_normal_model(tmp_path, device, impl, share):
    normal, host = paired(device, impl, share)
    normal.eval()
    host.eval()
    tokens = torch.arange(12, device=device).view(2, 6)
    with torch.no_grad():
        close(host(tokens), normal(tokens), device)
    assert all(not hasattr(table, "row_store") for table in row_sparse_tables(host))
    path = tmp_path / "model.pt"
    torch.save(host.state_dict(), path)
    restored = Transformer(host.cfg).to(device).eval()
    restored.load_state_dict(torch.load(path, weights_only=True, map_location=device), strict=True)
    assert restored.state_dict().keys() == normal.state_dict().keys()
    if share:
        assert len(row_sparse_tables(restored)) == 1
    with torch.no_grad():
        close(restored(tokens), host(tokens), device)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected, actual = restored(tokens), host(tokens)
            # The normal Triton eval path fuses the SwiLU multiply and bf16 cast into bag_infer.
            scale = expected.abs().max().clamp_min(1e-30)
            close(actual / scale, expected / scale, "cuda", rtol=0, atol=0.008)
    assert all(not getattr(p, "host_values", False) for p in restored.parameters())


@pytest.mark.parametrize("device,impl", BACKENDS)
@pytest.mark.parametrize("frozen", ["table", "rest"])
def test_frozen_table_and_table_only_training(device, impl, frozen):
    model = enable_host_values(tiny(impl)).to(device)
    table = table_of(model)
    for p in model.parameters():
        p.requires_grad_((p is not table) if frozen == "table" else (p is table))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt = optimizers(model, "fp32")
    tokens = torch.arange(14, device=device).view(2, 7)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with context:
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    clip_grads(model, 0.5)
    opt.step()
    changed = {name for name, p in model.named_parameters() if not bits_equal(p, before[name])}
    if frozen == "table":
        assert "layers.0.ffn.values.weight" not in changed
        assert not hasattr(table, "row_store")
        assert any("query_proj" in name for name in changed)
    else:
        assert changed == {"layers.0.ffn.values.weight"}


@pytest.mark.parametrize("device,impl", BACKENDS)
def test_opt_in_preserves_table_identity_and_placement(device, impl):
    model = tiny(impl)
    table = table_of(model)
    original = table.detach().clone()
    keys = tuple(model.state_dict())
    assert not getattr(table, "host_values", False)
    assert enable_host_values(model) is model
    assert enable_host_values(model) is model
    model.to(device=device, dtype=torch.bfloat16)
    assert table_of(model) is table
    assert all(memory.values.weight is table for memory in model.memory_layers())
    assert table.host_values and table.device.type == "cpu" and table.dtype == torch.float32
    assert bits_equal(table, original)
    assert tuple(model.state_dict()) == keys
    assert model.tok_emb.weight.device.type == device and model.tok_emb.weight.dtype == torch.bfloat16
    assert row_sparse_tables(model) == [table]
    if device == "cuda":
        model.cuda()
        assert table_of(model) is table and table.device.type == "cpu"
    normal = tiny(impl).to(device=device, dtype=torch.bfloat16)
    assert table_of(normal).device.type == device and table_of(normal).dtype == torch.bfloat16
    assert not getattr(table_of(normal), "host_values", False)


def test_staging_growth_reuses_capacity_buckets(monkeypatch):
    memory = enable_host_values(tiny()).memory_layers()[0].values
    empty, allocations = torch.empty, []

    def cpu_empty(shape, **kwargs):
        assert kwargs.pop("pin_memory") is True
        allocations.append(shape)
        return empty(shape, **kwargs)

    # Exercise the pool policy without requiring a CUDA/HIP pinned-memory allocator on the CPU runner.
    monkeypatch.setattr(torch, "empty", cpu_empty)
    first = memory.staging(5)
    second = memory.staging(7)
    assert first.data_ptr() == second.data_ptr()
    third = memory.staging(9)
    assert third.shape == (9, memory.embedding_dim)
    assert allocations == [(8, memory.embedding_dim), (16, memory.embedding_dim)]


def test_host_optimizer_updates_only_touched_chunks(monkeypatch):
    model = enable_host_values(tiny())
    table = table_of(model)
    opt = optimizers(model, "fp32")
    lazy = optimizer_for(opt, table)
    assert isinstance(lazy, LazyRowAdam)
    assert lazy.impl == "torch"
    # The old fp32 torch path invokes a dense inner Adam, which cannot scale to a host table.
    def no_dense_adam(*args, **kwargs):
        raise AssertionError("host table must not use a dense Adam step")
    monkeypatch.setattr(torch.optim.Adam, "step", no_dense_adam)
    st = row_store(table)
    st.acc[3] = 1.0
    st.touched[3] = True
    previous = table.detach().clone()
    lazy.step()
    unread = torch.arange(table.shape[0]) != 3
    assert bits_equal(table[unread], previous[unread])
    assert not bits_equal(table[3], previous[3])


@pytest.mark.parametrize("invalid", ["dense", "dtype", "no_memory", "engram"])
def test_host_mode_rejects_unsupported_models_before_mutation(invalid):
    model = tiny()
    if invalid == "dense":
        model.memory_layers()[-1].value_grad = "dense"
    elif invalid == "dtype":
        model.double()
    elif invalid == "no_memory":
        model = Transformer(ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=4, ffn_hidden=48))
    else:
        cfg = model.cfg
        cfg.eng_layers, cfg.eng_rows, cfg.eng_heads, cfg.eng_head_dim, cfg.eng_impl = [1], 17, 2, 8, "torch"
        model = Transformer(cfg)
    parameters = list(model.parameters())
    with pytest.raises(ValueError, match="row_sparse|fp32|PKM|Engram"):
        enable_host_values(model)
    assert all(p is q for p, q in zip(model.parameters(), parameters))
    assert all(not getattr(p, "host_values", False) for p in model.parameters())


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason="explicit Triton CPU interpreter check")
@pytest.mark.parametrize("mode", MODES)
def test_interpreter_triton_remaps_and_accumulates(mode):
    normal, host = tiny(), enable_host_values(tiny("triton"))
    host.load_state_dict(normal.state_dict())
    opts = [optimizers(m, mode) for m in (normal, host)]
    gen = torch.Generator().manual_seed(57)
    for _ in range(2):
        for _ in range(2):
            indices = torch.randint(0, 13, (8, 8), generator=gen) * 3 + 1
            weights = torch.randn(8, 8, generator=gen)
            gradient = torch.randn(8, 32, generator=gen)
            results = []
            for m in (normal, host):
                w = weights.clone().requires_grad_()
                out = m.memory_layers()[0].read_values(indices, w)
                out.backward(gradient)
                results.append((out, w.grad))
            for actual, expected in zip(results[1], results[0]):
                close(actual, expected, atol=1e-5)
            close(row_store(table_of(host)).acc, row_store(table_of(normal)).acc, atol=1e-5)
        for m, opt in zip((normal, host), opts):
            clip_grads(m, 0.1)
            opt.step()
            opt.zero_grad(set_to_none=True)
        close(table_of(host), table_of(normal))
