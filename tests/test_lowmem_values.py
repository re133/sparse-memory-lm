"""Compressed lazy Adam state: CPU toys and optional Triton agreement.

  OMP_NUM_THREADS=1 python -m pytest -q tests/test_lowmem_values.py
  TRITON_INTERPRET=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_lowmem_values.py
"""
import os

import pytest
import torch

from scripts.check_lowmem import DIM, MODES, ROWS, apply_gradient, regression, state_bytes, tracking
from smlm.sparse_values import LazyRowAdam, row_store

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cpu" if INTERPRET or not torch.cuda.is_available() else "cuda"
BACKENDS = ["torch", "triton"]


def require_backend(impl):
    if impl == "triton":
        if not INTERPRET and not torch.cuda.is_available():
            pytest.skip("Triton kernels need a GPU or TRITON_INTERPRET=1")
        pytest.importorskip("triton")


def row_state(optimizer, table):
    return {key: value.detach().clone() for key, value in optimizer.state[table].items()
            if key != "step" and torch.is_tensor(value)}


def same_bits(first, second):
    return torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8))


def decoded_state(optimizer, table):
    state = optimizer.state[table]
    m, v = state["exp_avg"].float(), state["exp_avg_sq"].float()
    if optimizer.state_dtype == "int8":
        m = m.sign() * (m / 127).square() * state["exp_avg_scale"][:, None]
        v = (v / 255 * state["exp_avg_sq_scale"][:, None]).square().square()
    return m, v


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("impl", BACKENDS)
def test_unread_rows_preserve_every_state_tensor(mode, impl):
    require_backend(impl)
    generator = torch.Generator().manual_seed(71)
    table = torch.nn.Parameter(torch.randn(19, 13, generator=generator).to(DEVICE))
    optimizer = LazyRowAdam([table], lr=0.01, state_dtype=mode, impl=impl)
    # Seed all rows first, so unread-state checks include nonzero historical moments and scales.
    visits = [list(range(19)), [1, 5, 8], [2, 9], [1, 5, 18]]
    for step, visited in enumerate(visits):
        rows = torch.tensor(visited, device=DEVICE)
        before = table.detach().clone()
        previous = row_state(optimizer, table)
        gradient = torch.randn(len(visited), 13, generator=generator).to(DEVICE)
        apply_gradient(table, optimizer, rows, gradient, torch.tensor(0.25, device=DEVICE))
        unread = torch.ones(19, dtype=torch.bool, device=DEVICE)
        unread[rows] = False
        assert same_bits(table.detach()[unread], before[unread])
        current = row_state(optimizer, table)
        for key in previous:
            assert same_bits(current[key][unread], previous[key][unread]), key
        assert int(optimizer.state[table]["step"]) == step + 1
        assert table.grad is None
        assert not row_store(table).touched.any()
        assert torch.count_nonzero(row_store(table).acc) == 0
        assert row_store(table).grad_scale == 1.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("impl", BACKENDS)
def test_empty_step_keeps_existing_backend_semantics(mode, impl):
    require_backend(impl)
    table = torch.nn.Parameter(torch.ones(7, 11, device=DEVICE))
    optimizer = LazyRowAdam([table], lr=0.01, state_dtype=mode, impl=impl)
    optimizer.step()
    assert not optimizer.state[table] if impl == "torch" else optimizer.state[table]["step"] == 1
    apply_gradient(table, optimizer, torch.tensor([3], device=DEVICE), torch.ones(1, 11, device=DEVICE))
    previous_step = int(optimizer.state[table]["step"])
    before, previous = table.detach().clone(), row_state(optimizer, table)
    optimizer.step()
    assert same_bits(before, table.detach())
    for key, value in previous.items():
        assert same_bits(value, optimizer.state[table][key]), key
    assert int(optimizer.state[table]["step"]) == previous_step + (impl == "triton")


@pytest.mark.parametrize("mode,bytes_per_row", [("fp32", 8 * DIM), ("bf16", 4 * DIM), ("int8", 2 * DIM + 8)])
@pytest.mark.parametrize("impl", BACKENDS)
def test_state_storage_matches_budget(mode, bytes_per_row, impl):
    require_backend(impl)
    table = torch.nn.Parameter(torch.zeros(ROWS, DIM, device=DEVICE))
    optimizer = LazyRowAdam([table], lr=0.01, state_dtype=mode, impl=impl)
    apply_gradient(table, optimizer, torch.tensor([0], device=DEVICE), torch.zeros(1, DIM, device=DEVICE))
    assert state_bytes(optimizer, table) == ROWS * bytes_per_row
    state = optimizer.state[table]
    expected = {"fp32": (torch.float32, torch.float32), "bf16": (torch.bfloat16, torch.bfloat16),
                "int8": (torch.int8, torch.uint8)}[mode]
    assert (state["exp_avg"].dtype, state["exp_avg_sq"].dtype) == expected
    if mode == "int8":
        for key in ("exp_avg_scale", "exp_avg_sq_scale"):
            assert state[key].dtype == torch.float32 and state[key].shape == (ROWS,)
            assert torch.count_nonzero(state[key]) == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compressed_states_track_sparse_fp32_trajectory(seed):
    result = tracking(seed)
    # Relative movement avoids hiding optimizer error behind the initial parameter values.
    # These permit small quantization drift; a material directional error must fail.
    for mode, overall, worst in (("bf16", 0.02, 0.05), ("int8", 0.05, 0.15)):
        assert result[mode]["relative_value_l2"] < overall, result[mode]
        assert result[mode]["relative_movement_l2"] < overall, result[mode]
        assert result[mode]["worst_row_relative_movement_l2"] < worst, result[mode]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compressed_states_converge_on_sparse_regression(seed):
    result = regression(seed)
    for mode, values in result["modes"].items():
        assert values["tail_mean_loss"] < result["initial_loss"] * 0.02, (mode, values)
        # Averaging the tail keeps the comparison insensitive to one noisy final observation.
        assert values["tail_loss_ratio_to_fp32"] < 1.10, (mode, values)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dim", [1, 23, 384])
def test_triton_matches_torch_in_each_state_mode(mode, dim):
    require_backend("triton")
    generator = torch.Generator().manual_seed(13)
    initial = torch.randn(17, dim, generator=generator).to(DEVICE) * 0.05
    tables = {impl: torch.nn.Parameter(initial.clone()) for impl in BACKENDS}
    opts = {impl: LazyRowAdam([table], lr=0.002, state_dtype=mode, impl=impl)
            for impl, table in tables.items()}
    row_scales = torch.logspace(-7, 3, 17).to(DEVICE)[:, None]
    for step in range(12):
        rows = torch.randperm(17, generator=generator)[:7].to(DEVICE)
        gradients = torch.randn(7, dim, generator=generator).to(DEVICE) * row_scales[rows]
        for impl in BACKENDS:
            apply_gradient(tables[impl], opts[impl], rows, gradients, 0.25 if step % 3 == 0 else 1.0)
    torch.testing.assert_close(tables["triton"], tables["torch"], rtol=1e-5, atol=1e-5)
    for actual, expected in zip(decoded_state(opts["triton"], tables["triton"]),
                                decoded_state(opts["torch"], tables["torch"])):
        # Stored codes may sit on different sides of a rounding boundary after fp32 arithmetic.
        per_row_scale = expected.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        torch.testing.assert_close(actual / per_row_scale, expected / per_row_scale, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["bf16", "int8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_compressed_states_require_fp32_parameters(mode, dtype):
    with pytest.raises(ValueError, match="fp32|float32"):
        LazyRowAdam([torch.nn.Parameter(torch.zeros(4, 7, dtype=dtype))], lr=0.01, state_dtype=mode)


def test_invalid_state_dtype_fails_loudly():
    with pytest.raises(ValueError, match="state_dtype"):
        LazyRowAdam([torch.nn.Parameter(torch.zeros(4, 7))], lr=0.01, state_dtype="fp16")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("options", [{"lr": float("nan")}, {"eps": float("nan")},
                                   {"lr": -0.1}, {"eps": -0.1}, {"betas": (0.9, 1.0)}])
def test_adam_hyperparameter_validation_is_preserved(mode, options):
    with pytest.raises(ValueError):
        LazyRowAdam([torch.nn.Parameter(torch.zeros(4, 7))], state_dtype=mode, **dict({"lr": 0.01}, **options))


@pytest.mark.parametrize("mode", ["bf16", "int8"])
@pytest.mark.parametrize("shape", [(5,), (2, 3, 4), (2, 0)])
def test_compressed_states_require_nonempty_matrix_width(mode, shape):
    with pytest.raises(ValueError, match="contiguous fp32 tables"):
        LazyRowAdam([torch.nn.Parameter(torch.zeros(shape))], lr=0.01, state_dtype=mode)


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_compressed_states_require_contiguous_tables(mode):
    table = torch.nn.Parameter(torch.zeros(7, 11).t())
    with pytest.raises(ValueError, match="contiguous fp32 tables"):
        LazyRowAdam([table], lr=0.01, state_dtype=mode)


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_torch_compact_update_covers_multiple_scratch_chunks(mode):
    table = torch.nn.Parameter(torch.zeros(257, 4096))
    optimizer = LazyRowAdam([table], lr=0.01, state_dtype=mode)
    apply_gradient(table, optimizer, torch.arange(257), torch.ones_like(table))
    torch.testing.assert_close(table, torch.full_like(table, -0.01))
    for value in decoded_state(optimizer, table):
        assert bool((value > 0).all())
    assert torch.count_nonzero(row_store(table).acc) == 0
    assert not row_store(table).touched.any()


@pytest.mark.parametrize("impl", BACKENDS)
def test_int8_cancellation_retains_small_positive_variance(impl):
    require_backend(impl)
    tables = {mode: torch.nn.Parameter(torch.zeros(1, 2, device=DEVICE)) for mode in ("fp32", "int8")}
    opts = {mode: LazyRowAdam([table], lr=0.003, state_dtype=mode, impl=impl)
            for mode, table in tables.items()}
    for values in ([1000.0, 0.0], [-900.0, 1.0], [0.0, 0.0]):
        for mode, table in tables.items():
            apply_gradient(table, opts[mode], torch.tensor([0], device=DEVICE),
                           torch.tensor([values], device=DEVICE))
    state = opts["int8"].state[tables["int8"]]
    assert state["exp_avg_sq"][0, 1] > 0
    assert torch.isfinite(tables["int8"]).all()
    # Losing a tiny denominator after cancellation used to produce a five-digit parameter jump.
    torch.testing.assert_close(tables["int8"][:, 1], tables["fp32"][:, 1], rtol=0.05, atol=1e-6)


@pytest.mark.parametrize("impl", BACKENDS)
def test_int8_positive_variance_never_rounds_to_zero(impl):
    require_backend(impl)
    table = torch.nn.Parameter(torch.zeros(1, 3, device=DEVICE))
    optimizer = LazyRowAdam([table], lr=0.003, state_dtype="int8", impl=impl)
    apply_gradient(table, optimizer, torch.tensor([0], device=DEVICE),
                   torch.tensor([[1e6, 1e-6, 0.0]], device=DEVICE))
    variance = optimizer.state[table]["exp_avg_sq"]
    assert variance[0, 0] == 255
    assert variance[0, 1] == 1
    assert variance[0, 2] == 0
