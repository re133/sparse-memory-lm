"""Step 3: memory add-on on a frozen Qwen3.5 (tiny random config, CPU). Needs transformers (.venv-qwen).
The PyTorch reference of Gated DeltaNet is forced (flash-linear-attention, if installed, only runs on the GPU);
the real model with the fast kernels is checked on the GPU by scripts/check_qwen_addon_gpu.py."""
import sys

import pytest
import torch

sys.modules["fla"] = None              # import fla -> ImportError -> transformers uses its PyTorch implementation

transformers = pytest.importorskip("transformers")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from smlm.optim import build_optimizer  # noqa: E402
from smlm.qwen_memory import AddOnConfig, attach, memory_macs, trainable_parameters  # noqa: E402


def tiny_qwen(seed=0):
    cfg = Qwen3_5TextConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_num_key_heads=4, linear_num_value_heads=4,
                            full_attention_interval=4, max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(seed)
    m = Qwen3_5ForCausalLM(cfg)
    return m


def addon_cfg(kind):
    return AddOnConfig(kind=kind, layers=[1, 2], n_keys=16, heads=2, knn=4, k_dim=16, impl="torch")


@pytest.mark.parametrize("kind", ["memory", "dense"])
@pytest.mark.parametrize("train_mode", [False, True])
def test_gate_zero_is_bit_identical(kind, train_mode):
    m = tiny_qwen()
    x = torch.randint(0, 300, (2, 24))
    m.train(train_mode)
    with torch.no_grad():
        want = m(x).logits
    attach(m, addon_cfg(kind))
    m.train(train_mode)
    with torch.no_grad():
        got = m(x).logits
    assert torch.equal(got, want)
    # the add-on really runs (nonzero output once the gate opens)
    with torch.no_grad():
        for a in m.addons:
            a.gate.fill_(0.5)
        assert not torch.allclose(m(x).logits, want)


@pytest.mark.parametrize("kind", ["memory", "dense"])
def test_only_addons_train_and_first_step_moves_only_gates(kind):
    m = tiny_qwen()
    frozen = {n: p.detach().clone() for n, p in m.named_parameters()}
    attach(m, addon_cfg(kind))
    m.train()
    params = trainable_parameters(m)
    assert params and all(p.requires_grad for p in params)
    assert all(not p.requires_grad for n, p in m.named_parameters() if not n.startswith("addons."))
    opt = build_optimizer(m.addons, lr=1e-2, value_lr=1e-2, weight_decay=0.0)
    before = {n: p.detach().clone() for n, p in m.addons.named_parameters()}
    x = torch.randint(0, 300, (2, 24))
    for step in range(2):
        out = m(x[:, :-1], labels=x[:, :-1])
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        changed = {n for n, p in m.addons.named_parameters() if not torch.equal(p, before[n])}
        if step == 0:
            # gates were 0: the add-on bodies get no gradient in the first step, only the gates move
            assert changed == {n for n in before if n.endswith("gate")}, changed
        before = {n: p.detach().clone() for n, p in m.addons.named_parameters()}
    assert changed - {n for n in before if n.endswith("gate")}, "bodies train from the second step on"
    for n, p in m.named_parameters():
        if not n.startswith("addons."):
            assert torch.equal(p, frozen[n]), n


def test_dense_control_has_memory_macs():
    c = AddOnConfig()
    d = 1024
    m = memory_macs(d, c)
    assert m == 1024 * 1024 + 4 * 2 * 1024 * 128 + 4 * 32 * 1024 + 2 * 1024 * 1024
    hidden = max(64, round(m / (3 * d) / 64) * 64)
    assert hidden == 1408 and abs(3 * d * hidden / m - 1) < 0.03


def test_fact_scoring_needs_the_whole_number():
    """Exact fill-in: "12" is wrong when the model writes "12.5"; punctuation after the answer is fine."""
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from eval_fact_cloze import correct
    for out, ans, ok in [(" 12.5 tons", "12", False), (" 10,000 people", "10", False), (" 123_456", "123", False),
                         (" 1998s", "1998", False), (" 12. Then", "12", True), (" 10, and", "10", True),
                         (" 1998", "1998", True), (" John Smith.", "John Smith", True), (" Johnson", "John", False)]:
        assert correct(out, ans) == ok, (out, ans)
