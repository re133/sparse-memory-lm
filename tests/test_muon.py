"""Muon mathematics, optimizer routing, and training integration.

Run: python -m pytest -q tests/test_muon.py
"""
import copy
import io
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from smlm.model import ModelConfig, Transformer
from smlm.muon import Muon, optimizer_description, zeropower_via_newtonschulz5
from smlm.optim import build_optimizer, clip_grads, lr_multiplier, set_lr
from smlm.sparse_values import LazyRowAdam, OptimizerSet, row_sparse_tables, row_store
from smlm.train import MODELS

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"))]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def tiny(kind="dense", device="cpu"):
    cfg = dict(vocab_size=32, d_model=16, n_layers=2, n_heads=2, ffn_hidden=32, max_seq_len=16)
    if kind in ("pkm", "both", "dense_values"):
        cfg.update(mem_layers=[0, 1], mem_n_keys=8, mem_heads=2, mem_knn=2, mem_k_dim=8,
                   mem_share_values=True, mem_keys_weight_decay=False, mem_score_scale="learned",
                   mem_value_grad="dense" if kind == "dense_values" else "row_sparse")
    if kind in ("engram", "both"):
        cfg.update(eng_layers=[0], eng_heads=2, eng_head_dim=4, eng_rows=17, eng_impl="torch")
    torch.manual_seed(19)
    return Transformer(ModelConfig(**cfg)).to(device)


def children(opt):
    return opt.opts if isinstance(opt, OptimizerSet) else [opt]


def legacy_optimizer(model, lr, value_lr, weight_decay, betas=(0.9, 0.95), eps=1e-8, eng_value_lr=None):
    """Frozen pre-Muon builder, independent of the new branch and its partition helper."""
    sparse_tables = row_sparse_tables(model)
    sparse_ids = {id(t) for t in sparse_tables}
    decay, no_decay, values = [], [], []
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in sparse_ids:
            continue
        if getattr(p, "pk_value_param", False):
            values.append(p)
        elif p.dim() >= 2 and not getattr(p, "no_weight_decay", False):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay, "base_lr": lr, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "base_lr": lr, "name": "no_decay"},
    ]
    if values:
        groups.append({"params": values, "weight_decay": 0.0, "base_lr": value_lr, "name": "memory_values"})
    for g in groups:
        g["lr"] = g["base_lr"]
    opt = torch.optim.AdamW(groups, betas=betas, eps=eps, fused=True)
    if sparse_tables:
        owners = {id(m.values.weight): m for m in model.modules() if getattr(m, "value_grad", None) == "row_sparse"}
        lazy = []
        for engram in (False, True):
            tabs = [t for t in sparse_tables if getattr(owners[id(t)], "is_engram", False) == engram]
            if not tabs:
                continue
            impl = "triton" if any(getattr(owners[id(t)], "impl", "torch") == "triton" for t in tabs) else "torch"
            tlr = (eng_value_lr or value_lr) if engram else value_lr
            lazy.append(LazyRowAdam(tabs, lr=tlr, betas=betas, eps=eps, impl=impl,
                                    name="engram_values" if engram else "memory_values"))
        return OptimizerSet(opt, *lazy)
    return opt


def reference_ns(matrix):
    """Independent evaluation of the quintic on singular values in fp64."""
    u, s, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    s = s / (s.norm() + 1e-7)
    for _ in range(5):
        s = 3.4445 * s - 4.7750 * s ** 3 + 2.0315 * s ** 5
    return (u * s) @ vh


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(8, 8), (6, 12), (12, 6)])
def test_newton_schulz_singular_band_and_reference(shape, device):
    torch.manual_seed(7)
    rows, cols = shape
    rank = min(shape)
    u = torch.linalg.qr(torch.randn(rows, rank)).Q
    v = torch.linalg.qr(torch.randn(cols, rank)).Q
    matrix = ((u * torch.linspace(0.5, 1.0, rank)) @ v.T).to(device)
    original = matrix.clone()
    result = zeropower_via_newtonschulz5(matrix)
    assert result.dtype == (torch.bfloat16 if device == "cuda" else torch.float32)
    assert result.shape == matrix.shape and torch.equal(matrix, original)
    singular = torch.linalg.svdvals(result.float())
    assert bool(((singular > 0.65) & (singular < 1.2)).all())
    # bf16 on the GPU: the quintic has slope ~-1.5 near s = 0.9, so rounding grows over the five steps; measured on
    # the RX 9070 up to 0.067 per element (3 shapes x 4 seeds). The band above is the property Muon needs.
    tolerance = 0.1 if device == "cuda" else 2e-5
    torch.testing.assert_close(result.double(), reference_ns(matrix), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
def test_newton_schulz_zero_and_ill_conditioned(device):
    zero = torch.zeros(7, 13, device=device)
    assert torch.count_nonzero(zeropower_via_newtonschulz5(zero)) == 0
    # Finite iterations do not turn tiny singular values into an exact polar factor.
    matrix = torch.diag(torch.tensor([1.0, 1e-3, 1e-8, 0.0], device=device))
    result = zeropower_via_newtonschulz5(matrix)
    assert torch.isfinite(result).all()
    tolerance = 0.04 if device == "cuda" else 2e-5
    torch.testing.assert_close(result.double(), reference_ns(matrix), rtol=tolerance, atol=tolerance)
    assert result[-1, -1] == 0 and result[2, 2].abs() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(5, 5), (3, 7), (7, 3)])
def test_muon_matches_independent_update_over_steps(device, shape):
    torch.manual_seed(13)
    parameter = nn.Parameter(torch.randn(shape, device=device))
    expected = parameter.detach().double().clone()
    buffer = torch.zeros_like(expected)
    opt = Muon([parameter], lr=0.003, momentum=0.95, weight_decay=0.1)
    for step in range(4):
        grad = torch.randn_like(parameter)
        parameter.grad = grad.clone()
        original = parameter.grad.clone()
        buffer = 0.95 * buffer + grad.double()
        nesterov = grad.double() + 0.95 * buffer
        expected = expected * (1 - 0.003 * 0.1) - 0.003 * 0.2 * math.sqrt(max(shape)) * reference_ns(nesterov)
        opt.step()
        assert torch.equal(parameter.grad, original), "step must not overwrite accumulated gradients"
        tolerance = 3e-4 if device == "cuda" else 3e-6
        torch.testing.assert_close(parameter.double(), expected, rtol=tolerance, atol=tolerance)


def test_muon_none_gradient_skips_decay_and_zero_gradient_decays():
    parameter = nn.Parameter(torch.ones(3, 5))
    opt = Muon([parameter], lr=0.01, weight_decay=0.1)
    opt.step()
    assert torch.equal(parameter, torch.ones_like(parameter)) and not opt.state
    parameter.grad = torch.zeros_like(parameter)
    opt.step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.999), rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["A", "B-1M-sparse", "E-1M", "D-100M"])
def test_preset_partition_complete_disjoint_and_semantic(preset):
    with torch.device("meta"):
        model = Transformer(ModelConfig(**MODELS[preset]))
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
    names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    members = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(members) == len(set(members)) and set(members) == set(names)
    muon = {names[id(p)] for subopt in children(opt) if isinstance(subopt, Muon)
            for group in subopt.param_groups for p in group["params"]}
    expected = set()
    for i, layer in enumerate(model.layers):
        suffixes = ["attn.wqkv.weight", "attn.wo.weight"]
        suffixes += (["ffn.query_proj.weight", "ffn.value_proj.weight"] if layer.is_memory
                     else ["ffn.w13.weight", "ffn.w2.weight"])
        if layer.engram is not None:
            suffixes += ["engram.w_k.weight", "engram.w_v.weight"]
        expected.update(f"layers.{i}.{suffix}" for suffix in suffixes)
    assert muon == expected
    lazy = {id(p) for subopt in children(opt) if isinstance(subopt, LazyRowAdam)
            for group in subopt.param_groups for p in group["params"]}
    assert lazy == {id(p) for p in row_sparse_tables(model)}
    assert id(model.tok_emb.weight) == id(model.lm_head.weight)
    for group in opt.param_groups:
        assert group["base_lr"] == group["lr"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("kind", ["dense", "pkm", "engram", "both", "dense_values"])
def test_adamw_default_bit_identical_to_legacy(kind, explicit):
    model, reference = tiny(kind), tiny(kind)
    args = (6e-4, 1e-3, 0.1)
    opt = build_optimizer(model, *args, eng_value_lr=0.002, **({"optimizer": "adamw"} if explicit else {}))
    old = legacy_optimizer(reference, *args, eng_value_lr=0.002)
    assert [type(o) for o in children(opt)] == [type(o) for o in children(old)]
    for step in range(3):
        ids = (torch.arange(36).reshape(4, 9) + step) % 32
        losses = []
        for m, optimizer in [(model, opt), (reference, old)]:
            set_lr(optimizer, lr_multiplier(step, 3, 1, 0.1))
            for chunk in ids.chunk(2):
                _, loss = m(chunk[:, :-1], chunk[:, 1:])
                (loss / 2).backward()
            losses.append(loss.detach())
            clip_grads(m, 0.2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert torch.equal(*losses)
        for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
            assert torch.equal(actual, expected), name
        for new_child, old_child in zip(children(opt), children(old)):
            for new_group, old_group in zip(new_child.param_groups, old_child.param_groups):
                assert {k: v for k, v in new_group.items() if k != "params"} == \
                       {k: v for k, v in old_group.items() if k != "params"}
                for new_p, old_p in zip(new_group["params"], old_group["params"]):
                    for key, actual in new_child.state[new_p].items():
                        expected = old_child.state[old_p][key]
                        assert torch.equal(actual, expected) if torch.is_tensor(actual) else actual == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_muon_accumulation_clipping_schedule_and_lazy_rows(device, impl):
    if impl == "triton" and device == "cpu":
        pytest.skip("Triton requires the GPU")
    model = tiny("both", device)
    for memory in model.memory_layers() + model.engram_layers():
        memory.impl = impl
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, eng_value_lr=0.002, optimizer="muon")
    before = {id(p): p.detach().clone() for p in model.parameters()}
    set_lr(opt, lr_multiplier(0, 4, 2, 0.1))
    for group in opt.param_groups:
        assert group["lr"] == 0.5 * group["base_lr"]
    ids = torch.arange(36, device=device).reshape(4, 9) % 32
    for chunk in ids.chunk(2):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(chunk[:, :-1], chunk[:, 1:])
        (loss / 2).backward()
    tables = row_sparse_tables(model)
    read = {id(p): row_store(p).touched.clone() for p in tables}
    assert all(p.grad is None for p in tables)
    norms = clip_grads(model, 0.01)
    assert all(torch.isfinite(norm) and norm > 0 for norm in norms)
    rest = [p.grad.flatten() for p in model.parameters() if p.grad is not None
            and not getattr(p, "pk_value_param", False)]
    assert torch.cat(rest).norm() <= 0.010001
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    for table in tables:
        touched = read[id(table)]
        assert torch.equal(table[~touched], before[id(table)][~touched])
        assert not torch.equal(table[touched], before[id(table)][touched])
        assert not row_store(table).touched.any()
    for subopt in children(opt):
        if isinstance(subopt, Muon):
            assert all(not torch.equal(p, before[id(p)]) for g in subopt.param_groups for p in g["params"])


@pytest.mark.parametrize("device", DEVICES)
def test_muon_state_dict_continuation(device):
    torch.manual_seed(12)
    parameter = nn.Parameter(torch.randn(7, 5, device=device))
    opt = Muon([{"params": [parameter], "base_lr": 0.01}], lr=0.01, weight_decay=0.1)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        opt.step()
    restored = nn.Parameter(parameter.detach().clone())
    resumed = Muon([restored], lr=0.7, momentum=0.5, weight_decay=0.0)
    checkpoint = io.BytesIO()
    torch.save(opt.state_dict(), checkpoint)
    checkpoint.seek(0)
    resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    assert resumed.param_groups[0]["base_lr"] == 0.01
    for step in range(3):
        parameter.grad = torch.randn_like(parameter)
        restored.grad = parameter.grad.clone()
        set_lr(opt, 1 / (step + 1))
        set_lr(resumed, 1 / (step + 1))
        opt.step()
        resumed.step()
        assert torch.equal(parameter, restored)


@pytest.mark.parametrize("kind", ["dense", "dense_values"])
def test_mixed_dense_state_dict_continuation(kind):
    model = tiny(kind)
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
    ids = torch.arange(36).reshape(4, 9) % 32
    for _ in range(2):
        model(ids[:, :-1], ids[:, 1:])[1].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    restored = tiny(kind)
    restored.load_state_dict(model.state_dict())
    resumed = build_optimizer(restored, 0.01, 0.02, 0.2, optimizer="muon")
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for step in range(3):
        for m, optimizer in [(model, opt), (restored, resumed)]:
            set_lr(optimizer, lr_multiplier(step, 3, 1, 0.1))
            m(ids[:, :-1], ids[:, 1:])[1].backward()
            clip_grads(m, 0.1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for (name, actual), expected in zip(model.named_parameters(), restored.parameters()):
            assert torch.equal(actual, expected), name


def test_mixed_lazy_state_dict_rejected_clearly():
    model = tiny("both")
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
    with pytest.raises(NotImplementedError, match="LazyRowAdam|row.sparse|resum"):
        opt.state_dict()
    with pytest.raises(NotImplementedError, match="LazyRowAdam|row.sparse|resum"):
        opt.load_state_dict({})


@pytest.mark.parametrize("frozen", ["body", "tables", "some"])
def test_frozen_parameters_are_not_assigned(frozen):
    model = tiny("both")
    tables = {id(p) for p in row_sparse_tables(model)}
    for name, p in model.named_parameters():
        if frozen == "body":
            p.requires_grad_(id(p) in tables)
        elif frozen == "tables":
            p.requires_grad_(id(p) not in tables)
        elif name in ("layers.0.attn.wqkv.weight", "tok_emb.weight"):
            p.requires_grad_(False)
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
    members = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(members) == len(set(members))
    assert set(members) == {id(p) for p in model.parameters() if p.requires_grad}


def test_embedding_tied_to_hidden_matrix_remains_adamw():
    model = tiny()
    # A hidden projection alias must not accidentally move an embedding to Muon.
    model.layers[0].attn.wo.weight = model.tok_emb.weight
    opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
    muon_ids = {id(p) for subopt in children(opt) if isinstance(subopt, Muon)
                for group in subopt.param_groups for p in group["params"]}
    assert id(model.tok_emb.weight) not in muon_ids
    members = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(members) == len(set(members)) == len(list(model.parameters()))


def test_optimizer_description_lists_actual_groups():
    model = tiny("both")
    opt = build_optimizer(model, 0.002, 0.004, 0.3, eng_value_lr=0.006, optimizer="muon")
    description = optimizer_description(opt)
    for expected in ("Muon", "momentum=0.95", "steps=5", "3.4445", "-4.7750", "2.0315",
                     "0.2*sqrt(max(rows,cols))", "AdamW", "LazyRowAdam", "bf16", "fp32"):
        assert expected in description
    for group in opt.param_groups:
        assert f"{group['name']}: base_lr={group['base_lr']:g}, wd={group['weight_decay']:g}" in description


@pytest.mark.parametrize("selection", [None, "muon"])
def test_cli_defaults_and_run_info_use_selected_optimizer(monkeypatch, tmp_path, selection):
    from smlm import train

    class InfoWritten(Exception):
        pass

    model = tiny()
    model.cuda = lambda: model
    monkeypatch.setattr(train, "Transformer", lambda cfg: model)
    monkeypatch.setattr(train, "load_meta", lambda dataset: {"splits": {"validation": {"n_words": 1}}})
    monkeypatch.setattr(train, "TrainStream", lambda *args, **kw: SimpleNamespace(tokens_per_step=32,
                                                                              steps_per_epoch=1))
    monkeypatch.setattr(train, "git_info", lambda: {})
    monkeypatch.setattr(train, "hardware_info", lambda: {})
    monkeypatch.setattr(train, "data_dir", lambda dataset: "unused")
    command = ["smlm.train", "--model", "A", "--out_dir", str(tmp_path), "--tokens", "32"]
    if selection is not None:
        command += ["--optimizer", selection]
    monkeypatch.setattr(sys, "argv", command)
    original_dump = json.dump

    def write_then_stop(info, file, **kwargs):
        original_dump(info, file, **kwargs)
        raise InfoWritten

    monkeypatch.setattr(train.json, "dump", write_then_stop)
    with pytest.raises(InfoWritten):
        train.main()
    info = json.loads((tmp_path / "run-info.json").read_text())
    config = info["train_config"]
    assert (config["lr"], config["value_lr"], config["weight_decay"]) == (6e-4, 1e-3, 0.1)
    assert (config["warmup_frac"], config["min_lr_ratio"], config["clip"]) == (0.05, 0.1, 1.0)
    if selection == "muon":
        assert "Muon" in config["optimizer"] and "AdamW" in config["optimizer"]
        assert "muon_decay: base_lr=0.0006" in config["optimizer"]
    else:
        assert config["optimizer"] == ("AdamW(fused) betas=(0.9,0.95) eps=1e-8; memory values: "
                                       "lr=value_lr, wd=0, separate clip")


def test_invalid_optimizer_and_nonmatrix_parameters_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(tiny(), 6e-4, 1e-3, 0.1, optimizer="unknown")
    with pytest.raises(ValueError, match="matri"):
        Muon([nn.Parameter(torch.ones(3))])
    with pytest.raises(ValueError, match="matrix"):
        zeropower_via_newtonschulz5(torch.ones(2, 3, 4))


def test_muon_rejects_sparse_gradient_before_updating():
    parameter = nn.Parameter(torch.ones(3, 5))
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.ones(2, 5), (3, 5), check_invariants=True)
    opt = Muon([parameter])
    with pytest.raises(RuntimeError, match="dense gradients"):
        opt.step()
    assert torch.equal(parameter, torch.ones_like(parameter)) and not opt.state


@pytest.mark.parametrize("override", [{"lr": -0.1}, {"weight_decay": float("nan")},
                                      {"momentum": 1.0}, {"ns_steps": 0}])
def test_muon_rejects_invalid_group_overrides(override):
    with pytest.raises(ValueError, match="Muon"):
        Muon([{"params": [nn.Parameter(torch.ones(3, 5))], **override}])


def test_dense_accumulation_matches_full_batch_update():
    full, accumulated = tiny(), tiny()
    ids = torch.arange(36).reshape(4, 9) % 32
    for model, micro in [(full, 4), (accumulated, 2)]:
        opt = build_optimizer(model, 6e-4, 1e-3, 0.1, optimizer="muon")
        for start in range(0, len(ids), micro):
            batch = ids[start:start + micro]
            _, loss = model(batch[:, :-1], batch[:, 1:])
            (loss * micro / len(ids)).backward()
        clip_grads(model, 0.1)
        opt.step()
    for (name, actual), expected in zip(full.named_parameters(), accumulated.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-7, msg=name)


@pytest.mark.parametrize("device", DEVICES)
def test_newton_schulz_precision_survives_outer_autocast(device):
    matrix = torch.eye(4, device=device)
    expected = zeropower_via_newtonschulz5(matrix)
    with torch.autocast(device, dtype=torch.bfloat16):
        actual = zeropower_via_newtonschulz5(matrix)
    assert torch.equal(actual, expected)
    assert actual.dtype == (torch.bfloat16 if device == "cuda" else torch.float32)


def test_mixed_checkpoint_rejects_incompatible_structure_before_loading():
    opt = build_optimizer(tiny(), 6e-4, 1e-3, 0.1, optimizer="muon")
    saved = copy.deepcopy(opt.state_dict())
    changed = copy.deepcopy(saved)
    changed["optimizers"][0]["state_dict"]["param_groups"][0]["lr"] = 0.123
    changed["optimizers"][-1]["state_dict"]["param_groups"].pop()
    with pytest.raises(ValueError, match="groups"):
        opt.load_state_dict(changed)
    assert opt.state_dict() == saved
