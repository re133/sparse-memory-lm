"""Step 1 "Gegenwert der Tabelle": dense presets D-50M ... D-400M and the budget gate of scripts/run_dense.py."""
import os
import sys

import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.train import MICRO_BS, MODELS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_dense  # noqa: E402

DENSE = {"D-50M": 50e6, "D-100M": 100e6, "D-200M": 200e6, "D-400M": 400e6}


@pytest.mark.parametrize("name", list(DENSE))
def test_dense_preset_shape(name):
    cfg = ModelConfig(**MODELS[name])
    assert not cfg.mem_layers
    assert cfg.d_model // cfg.n_heads == 64 and cfg.d_model % cfg.n_heads == 0
    assert cfg.ffn_hidden % 64 == 0 and abs(cfg.ffn_hidden / cfg.d_model - 8 / 3) < 0.1
    a = ModelConfig(**MODELS["A"])
    assert (cfg.vocab_size, cfg.max_seq_len, cfg.rope_theta, cfg.norm_eps, cfg.init_std) == \
        (a.vocab_size, a.max_seq_len, a.rope_theta, a.norm_eps, a.init_std)
    with torch.device("meta"):
        m = Transformer(cfg)
    pc = m.param_counts()
    assert abs(pc["non_embedding"] / DENSE[name] - 1) < 0.02
    assert pc["memory_values"] == 0
    assert 32 % MICRO_BS[name] == 0


def _tiny(seed=0):
    torch.manual_seed(seed)
    return Transformer(ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64,
                                   max_seq_len=32)).double()


def test_micro_batch_only_changes_memory():
    """Gradient accumulation over 2 x 4 sequences == one pass over 8 (train.py: (loss / accum).backward())."""
    x = torch.randint(0, 97, (8, 17))
    grads = []
    for micro in (8, 4, 2):
        m = _tiny()
        accum = 8 // micro
        for i in range(accum):
            mb = x[i * micro:(i + 1) * micro]
            _, loss = m(mb[:, :-1], mb[:, 1:])
            (loss / accum).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(grads[0], grads[2], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", list(DENSE))
def test_dense_forward_backward(name):
    if name != "D-50M" and not torch.cuda.is_available():
        pytest.skip("large presets only on the GPU")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    m = Transformer(ModelConfig(max_seq_len=64, **MODELS[name])).to(dev)
    x = torch.randint(0, 50257, (2, 65), device=dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        _, loss = m(x[:, :-1], x[:, 1:])
    loss.backward()
    assert torch.isfinite(loss) and abs(float(loss) - 10.83) < 0.5      # ~ ln(50304) at initialisation
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_budget_gate():
    order = ["D-50M-s0", "D-100M-s0", "D-200M-s0", "D-400M-s0"]
    fast = dict(zip(order, [2e6, 1e6, 6e5, 3e5]))
    adm, drop, proj = run_dense.admit(fast, elapsed_h=0.6, cap_h=5.0, order=order)
    assert adm == order and not drop
    work = (500e6 + 50 * run_dense.VAL_TOKENS / 3) * run_dense.FACTOR / 3600
    assert proj == pytest.approx(0.6 + sum(work / s for s in fast.values()) + run_dense.END_H)
    # 400M does not fit: dropped, the smaller ones run
    adm, drop, _ = run_dense.admit(fast, elapsed_h=0.6, cap_h=1.6, order=order)
    assert adm == order[:3] and [d[0] for d in drop] == ["D-400M-s0"]
    # once a size is over budget, the larger ones are dropped too
    slow200 = {**fast, "D-200M-s0": 1e5}
    adm, drop, _ = run_dense.admit(slow200, elapsed_h=0.6, cap_h=1.6, order=order)
    assert adm == order[:2] and [d[0] for d in drop] == order[2:]
    # a failed preflight only drops that size
    failed = {**fast, "D-100M-s0": None}
    adm, drop, _ = run_dense.admit(failed, elapsed_h=0.6, cap_h=5.0, order=order)
    assert adm == ["D-50M-s0", "D-200M-s0", "D-400M-s0"] and drop[0] == ("D-100M-s0", "preflight failed")
