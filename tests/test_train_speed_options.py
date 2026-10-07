"""Training flags and token-weighted chunked evaluation.

  python -m pytest -q tests/test_train_speed_options.py
"""
import sys

import numpy as np
import pytest
import torch

from smlm import train
from smlm.model import ModelConfig, Transformer


@pytest.mark.parametrize("option,value", [("--micro_bs", "0"), ("--micro_bs", "-1"),
                                         ("--micro_bs", "3"), ("--ce_chunk_size", "0")])
def test_invalid_batch_options_fail_before_io(monkeypatch, tmp_path, option, value):
    out = tmp_path / "unused"
    monkeypatch.setattr(sys, "argv", ["train", "--model", "A", "--out_dir", str(out),
                                     "--tokens", "32768", option, value])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU"))])
def test_evaluate_chunked_weights_partial_windows(monkeypatch, device):
    torch.manual_seed(11)
    model = Transformer(ModelConfig(vocab_size=37, d_model=16, n_heads=2, n_layers=1,
                                    ffn_hidden=32, max_seq_len=6)).to(device)
    tokens = np.arange(30, dtype=np.uint16) % 37
    monkeypatch.setattr(train, "load_split", lambda split, dataset: tokens)
    if device == "cpu":
        # evaluate is a GPU entry point; redirect only transfer/autocast to test its real window loop locally.
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
        autocast = torch.autocast
        monkeypatch.setattr(torch, "autocast", lambda *a, **kw: autocast("cpu", **kw))
    eager = train.evaluate(model, "validation", 6, n_words=10, batch=3)
    fused = train.evaluate(model, "validation", 6, n_words=10, batch=3, fused_ce=True, ce_chunk_size=4)
    assert eager["n_tokens"] == fused["n_tokens"] == len(tokens) - 1
    for field in ("loss", "ppl", "word_ppl"):
        assert fused[field] == pytest.approx(eager[field], rel=3e-6, abs=2e-6)
    assert model.training


def test_logits_generation_ignores_loss_options(monkeypatch):
    import smlm.fused_ce as fused
    torch.manual_seed(13)
    model = Transformer(ModelConfig(vocab_size=37, d_model=16, n_heads=2, n_layers=1,
                                    ffn_hidden=32, max_seq_len=8)).eval()
    tokens = torch.arange(8).view(1, -1)

    def unexpected(*args, **kwargs):
        raise AssertionError("plain logits must not call the chunked loss")

    monkeypatch.setattr(fused, "chunked_cross_entropy", unexpected)
    with torch.no_grad():
        logits = model(tokens)
        torch.testing.assert_close(model(tokens, fused_ce=True), logits, rtol=0, atol=0)
        with_targets, loss = model(tokens, tokens, fused_ce=False)
        torch.testing.assert_close(with_targets, logits, rtol=0, atol=0)
        expected = torch.nn.functional.cross_entropy(logits.float().view(-1, 37), tokens.reshape(-1))
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
