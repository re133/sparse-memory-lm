"""Tiny paired learning check: python -m pytest -q tests/test_muon_sanity.py."""
import math

import pytest
import torch

from smlm.bench_muon import compare_linear_optimizers, compare_optimizers, main


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP unavailable"))])
def test_muon_tiny_transformer_learns(device):
    result = compare_optimizers(device=device)
    adamw = result["results"]["adamw"]["validation_loss"]
    muon = result["results"]["muon"]["validation_loss"]
    assert math.isfinite(adamw) and math.isfinite(muon)
    assert adamw < math.log(result["config"]["vocab_size"])
    assert muon < math.log(result["config"]["vocab_size"])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP unavailable"))])
def test_muon_full_rank_linear_teacher(device):
    result = compare_linear_optimizers(device=device)
    adamw = result["results"]["adamw"]["loss"]
    muon = result["results"]["muon"]["loss"]
    assert math.isfinite(adamw) and math.isfinite(muon)
    assert muon <= adamw < result["initial_loss"]


def test_projection_benchmark_refuses_cpu(monkeypatch):
    monkeypatch.setattr("sys.argv", ["bench_muon", "--task", "projections", "--preset", "D-100M",
                                    "--device", "cpu", "--steps", "30", "--timing-warmup", "5"])
    with pytest.raises(ValueError, match="GPU-only"):
        main()
