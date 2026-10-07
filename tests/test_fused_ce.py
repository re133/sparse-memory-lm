"""Chunked CE parity and bounded-logit tests. Usage: python -m pytest -q tests/test_fused_ce.py

FP32 tolerances: rtol=2e-5, atol=2e-6. BF16 gradients: rtol=2e-2 and atol=4e-3 times
the reference's largest gradient (at least 1e-3), plus relative L2 error <=5e-3. The absolute
allowance is about half a bf16 ULP at that scale: each chunk rounds before fp32 accumulation.
"""
import pytest
import torch
import torch.nn.functional as F

from smlm.fused_ce import chunked_cross_entropy
from smlm.model import ModelConfig, Transformer


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU unavailable"))]


def close_gradient(actual, expected, amp):
    if amp:
        scale = max(float(expected.abs().max()), 1e-3)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-3 * scale)
        assert float((actual - expected).norm()) <= 5e-3 * float(expected.norm()) + 1e-7
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("chunk_size", [1, 5, 64])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_loss_and_gradients(device, amp, chunk_size, reduction):
    torch.manual_seed(0)
    hidden = torch.randn(2, 7, 32, device=device) * 0.2
    weight = torch.randn(97, 32, device=device) * 0.2
    targets = torch.randint(97, (2, 7), device=device)
    targets.reshape(-1)[5:10] = -100               # includes a whole ignored chunk
    results = []
    for chunked in (False, True):
        h, w = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            if chunked:
                loss = chunked_cross_entropy(h, w, targets, chunk_size, reduction)
            else:
                logits = F.linear(h, w)
                loss = F.cross_entropy(logits.float().reshape(-1, 97), targets.reshape(-1),
                                       reduction=reduction)
        (loss * 0.375).backward()                # upstream loss scaling, as in gradient accumulation
        results.append((loss.detach(), h.grad, w.grad))
    reference, actual = results
    torch.testing.assert_close(actual[0], reference[0], rtol=2e-5, atol=2e-6)
    for got, expected in zip(actual[1:], reference[1:]):
        close_gradient(got, expected, amp)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_tied_embedding_and_accumulation(device, amp):
    torch.manual_seed(1)
    weight = torch.randn(97, 32, device=device) * 0.1
    inputs = [torch.randint(97, (2, 7), device=device) for _ in range(2)]
    targets = [torch.randint(97, (2, 7), device=device) for _ in range(2)]
    results = []
    for chunked in (False, True):
        w = weight.clone().requires_grad_()
        losses = []
        for ids, labels in zip(inputs, targets):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
                hidden = F.embedding(ids, w).sin()
                if chunked:
                    loss = chunked_cross_entropy(hidden, w, labels, chunk_size=5)
                else:
                    loss = F.cross_entropy(F.linear(hidden, w).float().reshape(-1, 97), labels.reshape(-1))
            (loss / len(inputs)).backward()
            losses.append(loss.detach())
        results.append((torch.stack(losses), w.grad))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)
    close_gradient(results[1][1], results[0][1], amp)


def test_saved_tensors_and_projection_chunks(monkeypatch):
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(31, 8, requires_grad=True)
    targets = torch.randint(31, (2, 7))
    saved, projected = [], []
    linear = F.linear

    def project(x, w):
        projected.append(x.shape[0])
        return linear(x, w)

    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(F, "linear", project)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = chunked_cross_entropy(hidden, weight, targets, chunk_size=5)
    assert saved == [tuple(hidden.shape), tuple(weight.shape), tuple(targets.shape)]
    assert projected == [5, 5, 4]
    loss.backward()
    assert projected == [5, 5, 4, 5, 5, 4]


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_all_ignored_and_empty(reduction):
    for count in (0, 7):
        hidden = torch.randn(count, 8, requires_grad=True)
        weight = torch.randn(31, 8, requires_grad=True)
        targets = torch.full((count,), -100)
        loss = chunked_cross_entropy(hidden, weight, targets, chunk_size=3, reduction=reduction)
        ref = F.cross_entropy(F.linear(hidden, weight).float(), targets, reduction=reduction)
        torch.testing.assert_close(loss, ref, equal_nan=True)
        loss.backward()
        assert torch.equal(hidden.grad, torch.zeros_like(hidden))
        assert torch.equal(weight.grad, torch.zeros_like(weight))


@pytest.mark.parametrize("train_hidden,train_weight", [(False, True), (True, False)])
def test_frozen_inputs_and_noncontiguous_hidden(train_hidden, train_weight):
    torch.manual_seed(2)
    hidden = torch.randn(2, 8, 7).transpose(1, 2).requires_grad_(train_hidden)
    weight = torch.randn(31, 8).requires_grad_(train_weight)
    targets = torch.randint(31, (2, 7))
    loss = chunked_cross_entropy(hidden, weight, targets, chunk_size=5)
    ref = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 31), targets.reshape(-1))
    inputs = [p for p in (hidden, weight) if p.requires_grad]
    for got, expected in zip(torch.autograd.grad(loss, inputs), torch.autograd.grad(ref, inputs)):
        close_gradient(got, expected, False)


def test_no_grad_evaluation():
    hidden, weight, targets = torch.randn(2, 7, 8), torch.randn(31, 8), torch.randint(31, (2, 7))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        loss = chunked_cross_entropy(hidden, weight, targets, chunk_size=5, reduction="sum")
        ref = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 31), targets.reshape(-1),
                              reduction="sum")
    torch.testing.assert_close(loss, ref, rtol=2e-5, atol=2e-6)
    assert not loss.requires_grad


def test_fused_forward_does_not_pollute_autocast_weight_cache():
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    weight = torch.randn(31, 8, requires_grad=True)
    targets = torch.randint(31, (2, 7))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        chunked_cross_entropy(hidden, weight, targets, chunk_size=5)
        full_loss = F.cross_entropy(F.linear(hidden, weight).float().reshape(-1, 31), targets.reshape(-1))
    full_loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"chunk_size": -1}, {"chunk_size": 1.5},
                                    {"reduction": "none"}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        chunked_cross_entropy(torch.randn(7, 8), torch.randn(31, 8), torch.randint(31, (7,)), **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_model_default_and_fused_loss(device, amp):
    torch.manual_seed(3)
    cfg = ModelConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=2, ffn_hidden=64, max_seq_len=16)
    reference = Transformer(cfg).to(device)
    actual = Transformer(cfg).to(device)
    actual.load_state_dict(reference.state_dict())
    ids = torch.randint(97, (2, 7), device=device)
    targets = torch.randint(97, (2, 7), device=device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        logits, loss_ref = reference(ids, targets)
        logits_default, loss_default = reference(ids, targets, fused_ce=False)
        logits_fused, loss_actual = actual(ids, targets, fused_ce=True, ce_chunk_size=5)
        logits_generation = actual(ids, fused_ce=True, ce_chunk_size=5)
    assert logits_fused is None
    assert torch.equal(logits, logits_default)
    assert torch.equal(loss_ref, loss_default)
    assert torch.equal(logits, logits_generation)
    torch.testing.assert_close(loss_actual, loss_ref, rtol=2e-5, atol=2e-6)
    loss_ref.backward()
    loss_actual.backward()
    assert actual.lm_head.weight is actual.tok_emb.weight
    for expected, got in zip(reference.parameters(), actual.parameters()):
        close_gradient(got.grad, expected.grad, amp)
