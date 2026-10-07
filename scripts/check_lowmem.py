"""CPU toy checks for compressed lazy Adam states; no model training or GPU work.

  OMP_NUM_THREADS=1 python scripts/check_lowmem.py [--seeds 0 1 2] [--steps 320]
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.sparse_values import LazyRowAdam, row_store  # noqa: E402

MODES = ("fp32", "bf16", "int8")
ROWS, DIM, STEPS = 48, 32, 320


def apply_gradient(table, optimizer, rows, gradient, scale=1.0):
    store = row_store(table)
    store.acc[rows] = gradient
    store.touched[rows] = True
    store.grad_scale = scale
    optimizer.step()


def sparse_rows(generator):
    # Popular rows appear every step; the other rows retain state across long gaps.
    return torch.cat((torch.arange(8), torch.randperm(ROWS - 8, generator=generator)[:8] + 8))


def state_bytes(optimizer, table):
    state = optimizer.state[table]
    return sum(value.numel() * value.element_size() for key, value in state.items()
               if key != "step" and torch.is_tensor(value))


def errors(table, reference, initial):
    delta = table.detach() - reference.detach()
    movement = reference.detach() - initial
    return {"relative_value_l2": float(delta.norm() / reference.detach().norm().clamp_min(1e-12)),
            "relative_movement_l2": float(delta.norm() / movement.norm().clamp_min(1e-12)),
            "worst_row_relative_movement_l2": float((delta.norm(dim=1)
                                                     / movement.norm(dim=1).clamp_min(1e-12)).max()),
            "max_absolute_value_error": float(delta.abs().max())}


class Bf16RoundingExperiment:
    """Matched RN/SR equations for the rounding study, kept outside the production optimizer."""

    def __init__(self, table, lr, stochastic, seed):
        self.table, self.lr, self.stochastic = table, lr, stochastic
        self.generator = torch.Generator().manual_seed(seed)
        self.m = torch.zeros_like(table, dtype=torch.bfloat16)
        self.v = torch.zeros_like(table, dtype=torch.bfloat16)
        self.steps = 0

    def round(self, value):
        if not self.stochastic:
            return value.to(torch.bfloat16)
        # Uniform low mantissa bits choose neighboring bf16 values without persistent RNG state per row.
        noise = torch.randint(0, 1 << 16, value.shape, generator=self.generator, dtype=torch.int32)
        bits = (value.contiguous().view(torch.int32) + noise).bitwise_and(-65536)
        return bits.view(torch.float32).to(torch.bfloat16)

    @torch.no_grad()
    def step(self):
        store = row_store(self.table)
        rows = store.touched.nonzero().squeeze(1)
        if not rows.numel():
            return
        self.steps += 1
        gradient = store.acc[rows] * store.grad_scale
        m, v = self.m[rows].float(), self.v[rows].float()
        m = 0.9 * m + (1 - 0.9) * gradient
        v = 0.95 * v + (1 - 0.95) * gradient * gradient
        denominator = v.sqrt() / (1 - 0.95 ** self.steps) ** 0.5 + 1e-8
        self.table[rows] -= (self.lr / (1 - 0.9 ** self.steps)) * m / denominator
        self.m[rows], self.v[rows] = self.round(m), self.round(v)
        store.reset(rows)


def optimizers(initial, lr, seed, rounding_study):
    tables = {mode: torch.nn.Parameter(initial.clone()) for mode in MODES}
    opts = {mode: LazyRowAdam([table], lr=lr, state_dtype=mode) for mode, table in tables.items()}
    if rounding_study:
        for name, stochastic in (("bf16_rn_experiment", False), ("bf16_sr_experiment", True)):
            tables[name] = torch.nn.Parameter(initial.clone())
            opts[name] = Bf16RoundingExperiment(tables[name], lr, stochastic, seed + 10000)
    return tables, opts


def tracking(seed=0, steps=STEPS, rounding_study=False):
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(ROWS, DIM, generator=generator) * 0.05
    tables, opts = optimizers(initial, 0.002, seed, rounding_study)
    scales = torch.logspace(-7, 3, ROWS).unsqueeze(1)
    channels = torch.logspace(-2, 0, DIM).unsqueeze(0)
    direction = torch.randn(ROWS, DIM, generator=generator)
    for step in range(steps):
        rows = sparse_rows(generator)
        noise = torch.randn(len(rows), DIM, generator=generator)
        gradient = (0.6 * direction[rows] + 0.4 * noise) * scales[rows] * channels
        scale = 0.25 if step % 11 == 0 else 1.0
        for mode in tables:
            apply_gradient(tables[mode], opts[mode], rows, gradient, scale)
    result = {mode: errors(table, tables["fp32"], initial) for mode, table in tables.items()}
    for mode in MODES:
        result[mode]["state_bytes"] = state_bytes(opts[mode], tables[mode])
        result[mode]["state_bytes_per_row"] = result[mode]["state_bytes"] / ROWS
        result[mode]["state_bytes_per_parameter"] = result[mode]["state_bytes"] / (ROWS * DIM)
        result[mode]["table_accumulator_state_bytes_per_parameter"] = (
            result[mode]["state_bytes_per_parameter"] + 8)
    if rounding_study:
        result["bf16_rn_experiment"]["production_max_abs_difference"] = float(
            (tables["bf16_rn_experiment"].detach() - tables["bf16"].detach()).abs().max())
    return result


def regression(seed=0, steps=STEPS, rounding_study=False):
    generator = torch.Generator().manual_seed(seed + 1000)
    target = torch.randn(ROWS, DIM, generator=generator) * 0.2
    initial = torch.zeros_like(target)
    tables, opts = optimizers(initial, 0.015, seed, rounding_study)
    scales = torch.logspace(-7, 3, ROWS).unsqueeze(1)
    losses = {mode: [] for mode in tables}
    for step in range(steps):
        rows = sparse_rows(generator)
        observed = target[rows] + 0.05 * torch.randn(len(rows), DIM, generator=generator)
        for mode, table in tables.items():
            gradient = (table.detach()[rows] - observed) * scales[rows]
            apply_gradient(table, opts[mode], rows, gradient)
            losses[mode].append(float((table.detach() - target).square().mean()))
    result = {"initial_loss": float((initial - target).square().mean()), "modes": {}}
    reference_tail = sum(losses["fp32"][-40:]) / min(40, steps)
    for mode, table in tables.items():
        tail = sum(losses[mode][-40:]) / min(40, steps)
        result["modes"][mode] = {"final_loss": losses[mode][-1], "tail_mean_loss": tail,
                                 "tail_loss_ratio_to_fp32": tail / reference_tail,
                                 **errors(table, tables["fp32"], initial)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    report = {"torch": torch.__version__, "device": "cpu", "rows": ROWS, "dim": DIM,
              "steps": args.steps, "row_scale_min": 1e-7, "row_scale_max": 1e3,
              "tracking_channel_scale_min": 1e-2, "tracking_channel_scale_max": 1.0,
              "seeds": {str(seed): {"tracking": tracking(seed, args.steps, True),
                                    "regression": regression(seed, args.steps, True)} for seed in args.seeds}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
