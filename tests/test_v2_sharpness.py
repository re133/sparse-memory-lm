"""v2 "sharpness" switches: v2a = no weight decay on the sub-keys, v2b = v2a + learned per-head score scale."""
import math
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from smlm.model import ModelConfig, Transformer
from smlm.optim import build_optimizer
from smlm.pkm import ProductKeyMemory
from smlm.train import MODELS, MemoryStats

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
V2_FIELDS = ("mem_keys_weight_decay", "mem_score_scale", "mem_score_scale_init")


def pkm(device="cpu", seed=0, **kw):
    torch.manual_seed(seed)
    return ProductKeyMemory(d_in=24, d_out=24, n_keys=16, heads=2, knn=8, k_dim=16, **kw).double().to(device)


def tiny_model(**mem_kw):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=3, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[1], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16, **mem_kw)
    torch.manual_seed(0)
    return Transformer(cfg).double()


def group_of(opt, param):
    return next(g["name"] for g in opt.param_groups if any(p is param for p in g["params"]))


# ---- defaults are exactly the stage-1 (v1) behaviour --------------------------------------------------

def test_defaults_are_v1_and_old_configs_still_load():
    cfg = ModelConfig()
    assert (cfg.mem_keys_weight_decay, cfg.mem_score_scale, cfg.mem_score_scale_init) == (True, "none", 1.0)
    # a model_config saved by a stage-1 run (no v2 fields) still constructs, with v1 behaviour
    old = {k: v for k, v in asdict(ModelConfig(mem_layers=[6])).items() if k not in V2_FIELDS}
    assert ModelConfig(**old).mem_score_scale == "none"
    m = tiny_model()
    mem = m.memory_layers()[0]
    assert mem.log_score_scale is None
    assert not any("log_score_scale" in n for n, _ in m.named_parameters())
    assert not getattr(mem.keys, "no_weight_decay", False)
    assert group_of(build_optimizer(m, 1e-3, 1e-3, 0.1), mem.keys) == "decay"


def test_presets():
    assert MODELS["B-v2a"] == dict(MODELS["B"], mem_keys_weight_decay=False)
    assert MODELS["B-v2b"] == dict(MODELS["B"], mem_keys_weight_decay=False, mem_score_scale="learned")


# ---- v2a: no weight decay on the sub-keys ------------------------------------------------------------

def test_v2a_keys_in_no_decay_group_rest_unchanged():
    m = tiny_model(mem_keys_weight_decay=False)
    mem = m.memory_layers()[0]
    opt = build_optimizer(m, 1e-3, 1e-3, 0.1)
    assert group_of(opt, mem.keys) == "no_decay"
    assert group_of(opt, mem.values.weight) == "memory_values"
    assert group_of(opt, mem.query_proj.weight) == "decay"          # other matrices are still decayed
    assert group_of(opt, m.layers[0].attn.wqkv.weight) == "decay"
    assert {g["name"]: g["weight_decay"] for g in opt.param_groups} == \
        {"decay": 0.1, "no_decay": 0.0, "memory_values": 0.0}


@pytest.mark.parametrize("keys_wd", [True, False])
def test_v2a_optimizer_step_does_not_shrink_keys(keys_wd):
    """With zero gradients only weight decay moves parameters: v1 shrinks the keys, v2a leaves them alone."""
    m = tiny_model(mem_keys_weight_decay=keys_wd).float()
    mem = m.memory_layers()[0]
    opt = build_optimizer(m, 1e-2, 1e-3, 0.1)
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    before = mem.keys.detach().clone()
    opt.step()
    if keys_wd:
        torch.testing.assert_close(mem.keys.detach(), before * (1 - 1e-2 * 0.1))
    else:
        assert torch.equal(mem.keys.detach(), before)


# ---- v2b: learned per-head score scale ---------------------------------------------------------------

def test_v2b_init_reproduces_v1_exactly():
    v1 = pkm(score_scale="none")
    v2 = pkm(score_scale="learned", score_scale_init=1.0)
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert missing == ["log_score_scale"] and unexpected == []
    x = torch.randn(3, 5, 24, dtype=torch.float64)
    torch.testing.assert_close(v1(x), v2(x), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_v2b_scale_sharpens_weights_without_changing_selection(device):
    ref = pkm(device, score_scale="learned")
    sharp = pkm(device, score_scale="learned")
    sharp.load_state_dict(ref.state_dict())
    with torch.no_grad():
        sharp.log_score_scale.copy_(torch.tensor([math.log(5.0), math.log(0.5)], dtype=torch.float64))
    x = torch.randn(4, 6, 24, dtype=torch.float64, device=device)
    for m in (ref, sharp):
        m.record = True
        m(x)
    assert torch.equal(ref.last_indices, sharp.last_indices)       # positive scale: same top-k entries
    torch.testing.assert_close(ref.last_scores, sharp.last_scores)
    expected = F.softmax(sharp.last_scores.float() * torch.tensor([5.0, 0.5], device=device).view(1, 2, 1), -1)
    torch.testing.assert_close(sharp.last_weights, expected)
    ent = lambda w: -(w * w.log()).sum(-1)                           # noqa: E731
    assert (ent(sharp.last_weights[:, 0]) < ent(ref.last_weights[:, 0])).all()   # scale 5 -> sharper
    assert (ent(sharp.last_weights[:, 1]) > ent(ref.last_weights[:, 1])).all()   # scale 0.5 -> flatter


def test_v2b_scale_learns_and_is_not_decayed():
    m = tiny_model(mem_keys_weight_decay=False, mem_score_scale="learned", mem_score_scale_init=2.0)
    mem = m.memory_layers()[0]
    torch.testing.assert_close(mem.score_scale(), torch.full((2,), 2.0, dtype=torch.float64))
    x = torch.randint(0, 128, (2, 20))
    _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert mem.log_score_scale.grad is not None and mem.log_score_scale.grad.abs().sum() > 0
    opt = build_optimizer(m, 1e-3, 1e-3, 0.1)
    assert group_of(opt, mem.log_score_scale) == "no_decay"
    assert group_of(opt, mem.keys) == "no_decay"


# ---- stage-1 guarantees still hold for both variants -------------------------------------------------

VARIANTS = {"v2a": dict(keys_weight_decay=False),
            "v2b": dict(keys_weight_decay=False, score_scale="learned", score_scale_init=3.0)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variants_gradient_only_reaches_read_values(device, variant):
    mem = pkm(device, **VARIANTS[variant])
    x = torch.randn(4, 5, 24, dtype=torch.float64, device=device)
    mem.record = True
    out = mem(x)
    (out * torch.randn_like(out)).sum().backward()
    read = torch.unique(mem.last_indices.reshape(-1))
    grad_rows = torch.nonzero(mem.values.weight.grad.abs().sum(-1) > 0).squeeze(-1)
    assert torch.equal(grad_rows, read)
    assert read.numel() < mem.size


@pytest.mark.parametrize("variant", ["B-v2a", "B-v2b"])
def test_variants_causal_and_kv_cache(variant):
    kw = {k: v for k, v in MODELS[variant].items() if k != "mem_layers"}
    m = tiny_model(**kw)
    if m.memory_layers()[0].log_score_scale is not None:
        with torch.no_grad():
            m.memory_layers()[0].log_score_scale.fill_(math.log(4.0))
    m.eval()
    x = torch.randint(0, 128, (1, 30))
    y = x.clone()
    y[:, 20:] = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        full = m(x)
        torch.testing.assert_close(full[:, :20], m(y)[:, :20])
        caches = [dict() for _ in range(m.cfg.n_layers)]
        steps = [m(x[:, :10], kv_caches=caches, pos0=0)] + \
                [m(x[:, t:t + 1], kv_caches=caches, pos0=t) for t in range(10, 30)]
    torch.testing.assert_close(torch.cat(steps, 1), full)


def test_memory_stats_use_the_scaled_weights():
    """Sharpness metrics must reflect the weights actually used in the forward pass."""
    res = {}
    for scale in (1.0, 8.0):
        mem = pkm(score_scale="learned", score_scale_init=scale)
        mem.eval()
        mem.record = True
        with torch.no_grad():
            mem(torch.randn(64, 24, dtype=torch.float64))
        st = MemoryStats(mem.size, "cpu")
        st.add(mem)
        res[scale] = st.summary()
        assert 1.0 <= res[scale]["eff_entries_per_head"] <= mem.knn
    assert res[8.0]["eff_entries_per_head"] < res[1.0]["eff_entries_per_head"]
    assert res[8.0]["top1_weight"] > res[1.0]["top1_weight"]
