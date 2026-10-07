"""Paired optimizer sanity check on a tiny synthetic next-token task; no corpus or run files.

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/bench_muon.py --device cpu
python scripts/bench_muon.py --task linear --device cpu
python scripts/bench_muon.py --device cuda --steps 240
python scripts/bench_muon.py --task projections --preset A --device cuda --steps 30 --timing-warmup 5
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from smlm.model import ModelConfig, Transformer  # noqa: E402
from smlm.muon import Muon, muon_parameters  # noqa: E402
from smlm.optim import build_optimizer, clip_grads, lr_multiplier, set_lr  # noqa: E402


def _batches(count, batch_seqs, seq_len, vocab_size, generator, permutation):
    starts = torch.randint(vocab_size, (count, batch_seqs, 1), generator=generator)
    indices = (starts + torch.arange(seq_len + 1)) % vocab_size
    return permutation[indices]


def compare_optimizers(device="cpu", steps=240, seed=0, lr=6e-4, batch_seqs=8, seq_len=16,
                       eval_batches=8, timing_warmup=10):
    """Identical initialization and batches; validation averages fresh draws from a fixed token cycle.

    The timing includes forward/backward, clipping and optimizer updates after startup steps. This
    tiny task checks learning and wiring, not language-model quality or representative throughput.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is unavailable; use --device cpu")
    if steps <= 0 or batch_seqs <= 0 or seq_len <= 0 or eval_batches <= 0:
        raise ValueError("steps, batch_seqs, seq_len and eval_batches must be positive")
    if not 0 <= timing_warmup < steps:
        raise ValueError("timing_warmup must be nonnegative and smaller than steps")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    torch.set_num_threads(4)
    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=1, n_heads=2, ffn_hidden=64,
                      max_seq_len=seq_len)
    generator = torch.Generator().manual_seed(seed + 1)
    permutation = torch.randperm(cfg.vocab_size, generator=generator)
    batches = _batches(steps, batch_seqs, seq_len, cfg.vocab_size, generator, permutation).to(device)
    validation = _batches(eval_batches, batch_seqs, seq_len, cfg.vocab_size, generator, permutation).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = Transformer(cfg)
        state = copy.deepcopy(initial.state_dict())
    warmup_steps = max(1, math.ceil(steps * 0.05))
    result = {
        "device": device, "torch": torch.__version__, "seed": seed, "steps": steps,
        "batch_seqs": batch_seqs, "seq_len": seq_len, "eval_batches": eval_batches,
        "timing_warmup": timing_warmup, "config": cfg.to_dict(), "parameters": initial.param_counts(),
        "lr": lr, "weight_decay": 0.1, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8,
        "warmup_steps": warmup_steps, "min_lr_ratio": 0.1, "max_grad_norm": 1.0,
        "autocast": "bfloat16" if device == "cuda" else "disabled",
        "task": "randomly relabelled deterministic token cycle; fresh validation sequence draws",
        "results": {},
    }

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    for name in ("adamw", "muon"):
        model = Transformer(cfg).to(device)
        model.load_state_dict(state)
        opt = build_optimizer(model, lr=lr, value_lr=1e-3, weight_decay=0.1, optimizer=name)
        losses = []
        started = None
        for step, tokens in enumerate(batches):
            if step == timing_warmup:
                synchronize()
                started = time.perf_counter()
            set_lr(opt, lr_multiplier(step, steps, warmup_steps, 0.1))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(tokens[:, :-1], tokens[:, 1:])
            loss.backward()
            clip_grads(model, 1.0)
            opt.step()
            losses.append(loss.detach())
        synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            val_losses = [model(tokens[:, :-1], tokens[:, 1:])[1] for tokens in validation]
        final_loss = float(torch.stack(val_losses).mean())
        final_train_loss = float(torch.stack(losses[-20:]).mean())
        if not math.isfinite(final_loss) or not math.isfinite(final_train_loss):
            raise RuntimeError(f"{name} produced a non-finite loss")
        timed_steps = steps - timing_warmup
        result["results"][name] = {
            "validation_loss": final_loss, "final_train_loss_mean": final_train_loss,
            "final_train_loss_steps": min(20, steps), "timed_steps": timed_steps,
            "training_seconds": elapsed, "milliseconds_per_step": 1000 * elapsed / timed_steps,
            "tokens_per_second": timed_steps * batch_seqs * seq_len / elapsed,
        }
        del opt, model
    result["muon_minus_adamw_loss"] = (result["results"]["muon"]["validation_loss"]
                                      - result["results"]["adamw"]["validation_loss"])
    result["muon_no_worse"] = result["muon_minus_adamw_loss"] <= 0
    return result


def compare_linear_optimizers(device="cpu", steps=240, lr=6e-4, timing_warmup=10):
    """Fit the identity from zero on a full orthogonal basis; a deliberately simple matrix check.

    Equal diagonal singular directions isolate the shape-scaled orthogonal update. This is not
    evidence that Muon beats AdamW on language models, nor a search for a winning seed or rate.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is unavailable; use --device cpu")
    if steps <= 0 or not 0 <= timing_warmup < steps:
        raise ValueError("steps must be positive and timing_warmup smaller than steps")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    torch.set_num_threads(4)
    dim = 64
    basis = torch.eye(dim, device=device)
    warmup_steps = max(1, math.ceil(steps * 0.05))
    result = {
        "device": device, "torch": torch.__version__, "steps": steps, "dimension": dim,
        "parameters": dim * dim, "initialization": "zero matrix", "target": "identity matrix",
        "lr": lr, "weight_decay": 0.1, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8,
        "warmup_steps": warmup_steps, "min_lr_ratio": 0.1, "max_grad_norm": 1.0,
        "timing_warmup": timing_warmup, "task": "linear identity fit on the full orthogonal basis",
        "initial_loss": float(basis.square().mean()), "minimum_data_loss": 0.0, "results": {},
    }
    for name in ("adamw", "muon"):
        model = torch.nn.Linear(dim, dim, bias=False, device=device)
        with torch.no_grad():
            model.weight.zero_()
        groups = [{"params": list(model.parameters()), "base_lr": lr}]
        opt = (Muon(groups, lr=lr, weight_decay=0.1) if name == "muon" else
               torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True))
        losses = []
        started = None
        for step in range(steps):
            if step == timing_warmup:
                if device == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
            set_lr(opt, lr_multiplier(step, steps, warmup_steps, 0.1))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = F.mse_loss(model(basis).float(), basis)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach())
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            final_loss = float(F.mse_loss(model(basis), basis))
        if not math.isfinite(final_loss):
            raise RuntimeError(f"{name} produced a non-finite loss")
        result["results"][name] = {
            "loss": final_loss, "final_train_loss_mean": float(torch.stack(losses[-20:]).mean()),
            "final_train_loss_steps": min(20, steps), "training_seconds": elapsed,
            "timed_steps": steps - timing_warmup,
            "milliseconds_per_step": 1000 * elapsed / (steps - timing_warmup),
        }
    result["muon_minus_adamw_loss"] = (result["results"]["muon"]["loss"]
                                      - result["results"]["adamw"]["loss"])
    result["muon_no_worse"] = result["muon_minus_adamw_loss"] <= 0
    return result


def compare_projection_optimizers(device="cuda", preset="A", steps=30, seed=0, lr=6e-4,
                                  timing_warmup=5):
    """Time optimizer steps on real projection shapes with fixed gradients; never train a model."""
    if device != "cuda":
        raise ValueError("The real-shape projection benchmark is GPU-only; use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP is unavailable; projection timings require a GPU")
    if preset not in ("A", "D-100M"):
        raise ValueError("projection preset must be A or D-100M")
    if steps <= 0 or not 0 <= timing_warmup < steps:
        raise ValueError("steps must be positive and timing_warmup smaller than steps")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    from .train import MODELS

    torch.set_num_threads(4)
    with torch.device("meta"):
        model = Transformer(ModelConfig(**MODELS[preset]))
    shapes = [tuple(p.shape) for p in muon_parameters(model)]
    del model
    # Both optimizers get the same matrices. No sampling, cloning, forward or backward is timed.
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = [torch.randn(shape, device=device, dtype=torch.float32, generator=generator) * 0.02
               for shape in shapes]
    gradients = [torch.randn(shape, device=device, dtype=torch.float32, generator=generator) * 0.01
                 for shape in shapes]
    result = {
        "task": "optimizer steps only on actual hidden projection shapes; fixed synthetic gradients",
        "preset": preset, "device": device, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "hip": torch.version.hip, "seed": seed,
        "steps": steps, "timing_warmup": timing_warmup, "lr": lr, "weight_decay": 0.1,
        "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8, "muon_momentum": 0.95,
        "muon_ns_steps": 5, "muon_ns_dtype": "bfloat16", "parameter_dtype": "float32",
        "gradient_dtype": "float32", "matrix_count": len(shapes),
        "elements": sum(math.prod(shape) for shape in shapes),
        "shapes": [{"shape": list(shape), "matrices": count, "elements_each": math.prod(shape)}
                   for shape, count in sorted(Counter(shapes).items())],
        "memory_scope": "retained initial weights and gradients, optimized weights, optimizer state and temporaries",
        "results": {},
    }
    for name in ("adamw", "muon"):
        params = [torch.nn.Parameter(weight.clone()) for weight in weights]
        for p, gradient in zip(params, gradients):
            p.grad = gradient
        opt = (Muon(params, lr=lr, weight_decay=0.1) if name == "muon" else
               torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, fused=True))
        for _ in range(timing_warmup):
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        start_event.record()
        for _ in range(steps - timing_warmup):
            opt.step()
        end_event.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        event_ms = start_event.elapsed_time(end_event)
        timed_steps = steps - timing_warmup
        result["results"][name] = {
            "timed_steps": timed_steps, "wall_seconds": elapsed,
            "wall_milliseconds_per_step": 1000 * elapsed / timed_steps,
            "gpu_event_milliseconds_per_step": event_ms / timed_steps,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        del opt, params, p
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("transformer", "linear", "projections"), default="transformer")
    parser.add_argument("--preset", choices=("A", "D-100M"), default="A", help="projection benchmark only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--batch-seqs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--timing-warmup", type=int, default=10)
    args = parser.parse_args()
    if args.task == "projections":
        result = compare_projection_optimizers(args.device, args.preset, args.steps, args.seed,
                                              args.lr, args.timing_warmup)
    elif args.task == "linear":
        result = compare_linear_optimizers(args.device, args.steps, args.lr, args.timing_warmup)
    else:
        kwargs = vars(args)
        kwargs.pop("task")
        kwargs.pop("preset")
        result = compare_optimizers(**kwargs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
